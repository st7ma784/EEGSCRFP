#!/usr/bin/env bash
# Launch a W&B hyperparameter sweep for EEGSCRFP on Lancaster HEC.
# Run this on the LOGIN NODE (not via sbatch).
#
# Usage:
#   ./scripts/hc_sweep_launch.sh patches                      # 10 agents, 1 trial each
#   ./scripts/hc_sweep_launch.sh encoder --agents 20          # 20 agent jobs
#   ./scripts/hc_sweep_launch.sh patches --agents 5 --count 2 # 5 jobs × 2 trials
#   ./scripts/hc_sweep_launch.sh patches --resume abc123def   # add agents to existing sweep
#   ./scripts/hc_sweep_launch.sh configs/sweep/my.yaml        # custom config file
#   ./scripts/hc_sweep_launch.sh patches --time 02:00:00      # override wall-time
#   ./scripts/hc_sweep_launch.sh patches --dry-run            # print commands, don't submit
#
# Prerequisites:
#   - WANDB_API_KEY set in environment (e.g. in ~/.bashrc on HEC)
#   - WANDB_ENTITY set (e.g. export WANDB_ENTITY=st7ma784)
#   - wandb and sbatch on PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Sweep → config mapping ────────────────────────────────────────────────────
declare -A SWEEP_CONFIGS=(
    [patches]="configs/sweep/patch_simulation.yaml"
    [encoder]="configs/sweep/encoder_architecture.yaml"
    [lora]="configs/sweep/lora_participants.yaml"
    [objectives]="configs/sweep/training_objectives.yaml"
)

declare -A SWEEP_NAMES=(
    [patches]="Patch simulation (sensor count degradation)"
    [encoder]="Encoder architecture sweep (linear/MLP/transformer)"
    [lora]="LoRA participant generalisation sweep"
    [objectives]="Training objective weights sweep"
)

# ── Defaults ──────────────────────────────────────────────────────────────────
N_AGENTS=10
TRIALS_PER_AGENT=1
SWEEP_ID=""
WALL_TIME="04:00:00"
PARTITION="gpu-medium"
DRY_RUN=false

# ── Parse arguments ───────────────────────────────────────────────────────────
SWEEP_KEY=${1:?"Usage: $0 <patches|encoder|lora|objectives|path/to/sweep.yaml> [--agents N] [--count N] [--resume ID] [--time HH:MM:SS] [--dry-run]"}
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --agents)    N_AGENTS="$2";         shift 2 ;;
        --count)     TRIALS_PER_AGENT="$2"; shift 2 ;;
        --resume)    SWEEP_ID="$2";         shift 2 ;;
        --time)      WALL_TIME="$2";        shift 2 ;;
        --partition) PARTITION="$2";        shift 2 ;;
        --dry-run)   DRY_RUN=true;          shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Resolve config path ───────────────────────────────────────────────────────
if [[ -v "SWEEP_CONFIGS[$SWEEP_KEY]" ]]; then
    CONFIG="${PROJECT_ROOT}/${SWEEP_CONFIGS[$SWEEP_KEY]}"
    LABEL="${SWEEP_NAMES[$SWEEP_KEY]}"
else
    # Treat as a direct path to a YAML file
    CONFIG="$SWEEP_KEY"
    LABEL="$(basename "$CONFIG" .yaml)"
fi

cd "$PROJECT_ROOT"
[[ -f "$CONFIG" ]] || { echo "ERROR: sweep config not found: $CONFIG" >&2; exit 1; }

# ── Pre-flight checks ─────────────────────────────────────────────────────────
command -v python >/dev/null 2>&1 \
    || { echo "ERROR: python not on PATH. Activate your conda env first." >&2; exit 1; }
command -v sbatch >/dev/null 2>&1 \
    || { echo "ERROR: sbatch not on PATH. Run this on the HEC login node." >&2; exit 1; }

[[ -z "${WANDB_API_KEY:-}" ]] && {
    echo "ERROR: WANDB_API_KEY is not set." >&2
    echo "  export WANDB_API_KEY=<your-key>" >&2
    exit 1
}
[[ -z "${WANDB_ENTITY:-}" ]] && {
    echo "ERROR: WANDB_ENTITY is not set." >&2
    echo "  export WANDB_ENTITY=st7ma784   # or your W&B username" >&2
    exit 1
}

mkdir -p "${PROJECT_ROOT}/logs"

# ── Create or resume sweep ────────────────────────────────────────────────────
if [[ -z "$SWEEP_ID" ]]; then
    echo "Creating W&B sweep: $LABEL"
    echo "  Config: $CONFIG"
    echo ""

    if $DRY_RUN; then
        echo "[dry-run] python -c 'wandb.sweep(...)' $CONFIG"
        SWEEP_ID="dry-run-id"
    else
        SWEEP_ID=$(python - "$CONFIG" "${WANDB_ENTITY}" <<'PYEOF'
import sys, yaml, wandb, contextlib
cfg     = yaml.safe_load(open(sys.argv[1], encoding='utf-8'))
entity  = sys.argv[2]
project = cfg.get("project", "eegscrfp")
with contextlib.redirect_stdout(sys.stderr):
    sweep_id = wandb.sweep(cfg, entity=entity, project=project)
parts = sweep_id.split("/")
if len(parts) == 1:
    sweep_id = f"{entity}/{project}/{sweep_id}"
elif len(parts) == 2:
    sweep_id = f"{entity}/{sweep_id}"
print(sweep_id, flush=True)
PYEOF
)
        [[ -z "$SWEEP_ID" ]] && {
            echo "ERROR: sweep creation failed. Check WANDB_API_KEY / WANDB_ENTITY." >&2
            echo "Re-run with --resume <id> once you have a sweep ID." >&2
            exit 1
        }
        echo "  Created: $SWEEP_ID"
    fi
else
    echo "Resuming existing sweep: $SWEEP_ID"
fi

TOTAL_TRIALS=$(( N_AGENTS * TRIALS_PER_AGENT ))

echo ""
echo "=== Sweep Summary ============================================="
echo "  Sweep:        $LABEL"
echo "  Sweep ID:     $SWEEP_ID"
echo "  Agents:       $N_AGENTS jobs × $TRIALS_PER_AGENT trial(s) = $TOTAL_TRIALS total trials"
echo "  Wall-time:    $WALL_TIME per job"
echo "  Partition:    $PARTITION"
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    echo "  Dashboard:    https://wandb.ai/${WANDB_ENTITY}/eegscrfp/sweeps/${SWEEP_ID##*/}"
fi
echo "==============================================================="
echo ""

$DRY_RUN && {
    echo "[dry-run] Would submit $N_AGENTS agent jobs:"
    echo "  sbatch --time=$WALL_TIME --partition=$PARTITION \\"
    echo "    scripts/hc_sweep_agent.sh $SWEEP_ID $TRIALS_PER_AGENT"
    exit 0
}

# ── Submit agent jobs ─────────────────────────────────────────────────────────
SUBMITTED_IDS=()
SHORT_ID="${SWEEP_ID##*/}"   # strip entity/project/ prefix
SHORT_ID="${SHORT_ID:0:8}"   # first 8 chars keeps job names readable

for i in $(seq 1 "$N_AGENTS"); do
    JOB_ID=$(sbatch \
        --job-name="scrfp-${SHORT_ID}" \
        --time="$WALL_TIME" \
        --partition="$PARTITION" \
        --output="logs/sweep_${SHORT_ID}_%j.out" \
        --error="logs/sweep_${SHORT_ID}_%j.err" \
        --parsable \
        "${SCRIPT_DIR}/hc_sweep_agent.sh" "$SWEEP_ID" "$TRIALS_PER_AGENT")
    SUBMITTED_IDS+=("$JOB_ID")
    printf "  Submitted agent %2d / %d -> job %s\n" "$i" "$N_AGENTS" "$JOB_ID"
done

echo ""
echo "All $N_AGENTS agents queued."
echo ""
echo "Useful commands:"
echo "  Watch queue:      squeue -u \$USER"
echo "  Cancel all:       scancel ${SUBMITTED_IDS[*]}"
echo "  Follow first log: tail -f logs/sweep_${SHORT_ID}_${SUBMITTED_IDS[0]}.out"
echo ""
echo "Add more agents later:"
echo "  $0 ${SWEEP_KEY} --resume ${SWEEP_ID} --agents 5 --time ${WALL_TIME}"
