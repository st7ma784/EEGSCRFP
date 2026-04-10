#!/usr/bin/env bash
# Launch all EEGSCRFP hyperparameter sweeps locally.
# Creates W&B sweep IDs for all four sweeps, then runs agents sequentially.
#
# Usage:
#   cd EEGSCRFP
#   ./scripts/run_sweeps_local.sh                      # all sweeps, default trial counts
#   ./scripts/run_sweeps_local.sh --only encoder       # single sweep
#   ./scripts/run_sweeps_local.sh --dry-run            # print sweep IDs, don't run agents
#   ./scripts/run_sweeps_local.sh --encoder 40 --objectives 20 --patch 15 --lora 10
#
# Sweep order (sequential — each finishes before the next starts):
#   1. encoder_architecture   (Bayesian, fastest per trial)
#   2. training_objectives    (Grid, 64 combos)
#   3. patch_simulation       (Grid, sensor count degradation)
#   4. lora_participants      (Bayesian, most expensive per trial)
#
# To run sweeps in parallel across tmux panes instead:
#   tmux new-session -d -s sweeps
#   tmux send-keys -t sweeps "wandb agent --count 30 <ENCODER_ID>" Enter
#   tmux split-window -h
#   tmux send-keys -t sweeps "wandb agent --count 20 <PATCH_ID>" Enter
#   ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Default trial counts ───────────────────────────────────────────────────────
N_ENCODER=30
N_OBJECTIVES=30
N_PATCH=20
N_LORA=15
ONLY=""
DRY_RUN=false

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --encoder)    N_ENCODER="$2";    shift 2 ;;
        --objectives) N_OBJECTIVES="$2"; shift 2 ;;
        --patch)      N_PATCH="$2";      shift 2 ;;
        --lora)       N_LORA="$2";       shift 2 ;;
        --only)       ONLY="$2";         shift 2 ;;
        --dry-run)    DRY_RUN=true;      shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Environment ───────────────────────────────────────────────────────────────
export WANDB_PROJECT="${WANDB_PROJECT:-eegscrfp}"
export WANDB_ENTITY="${WANDB_ENTITY:-PGNTeam}"
# Feature/output dirs — fall back to local paths if env vars not set
export EEGSCRFP_DATA_DIR="${EEGSCRFP_DATA_DIR:-}"
export EEGSCRFP_OUTPUT_DIR="${EEGSCRFP_OUTPUT_DIR:-${PROJECT_ROOT}/outputs}"

# Activate conda env if not already active
if [[ "$(conda info --envs 2>/dev/null | grep -c '\*')" -eq 0 ]] 2>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate opence 2>/dev/null || true
fi

PYTHON="${PYTHON:-python}"

# ── Helper: create one sweep, print ID ───────────────────────────────────────
create_sweep() {
    local config="$1"
    local id
    id=$($PYTHON - "$config" "$WANDB_ENTITY" <<'PYEOF'
import sys, yaml, wandb, contextlib

cfg    = yaml.safe_load(open(sys.argv[1], encoding='utf-8'))
entity = sys.argv[2]
project = cfg.get("project", "eegscrfp")
with contextlib.redirect_stdout(sys.stderr):
    sweep_id = wandb.sweep(cfg, entity=entity, project=project)
# Emit fully-qualified ID
parts = sweep_id.split("/")
if len(parts) == 1:
    import os
    project = os.environ.get("WANDB_PROJECT", "eegscrfp")
    sweep_id = f"{entity}/{project}/{sweep_id}"
elif len(parts) == 2:
    sweep_id = f"{entity}/{sweep_id}"
print(sweep_id, flush=True)
PYEOF
    )
    echo "$id"
}

# ── Create all sweeps upfront ─────────────────────────────────────────────────
echo "================================================================"
echo " EEGSCRFP — Full sweep deployment"
echo "  Entity:  $WANDB_ENTITY"
echo "  Project: $WANDB_PROJECT"
echo "================================================================"
echo ""

declare -A SWEEP_IDS

for sweep_key in encoder objectives patch lora; do
    [[ -n "$ONLY" && "$sweep_key" != "$ONLY" ]] && continue

    case $sweep_key in
        encoder)    config="configs/sweep/encoder_architecture.yaml"  ; n=$N_ENCODER    ;;
        objectives) config="configs/sweep/training_objectives.yaml"   ; n=$N_OBJECTIVES ;;
        patch)      config="configs/sweep/patch_simulation.yaml"      ; n=$N_PATCH      ;;
        lora)       config="configs/sweep/lora_participants.yaml"     ; n=$N_LORA       ;;
    esac

    printf "  Creating %-12s sweep from %s ... " "$sweep_key" "$config"
    if $DRY_RUN; then
        echo "[dry-run]"
        SWEEP_IDS[$sweep_key]="dry-run/$sweep_key"
    else
        id=$(create_sweep "$config")
        SWEEP_IDS[$sweep_key]="$id"
        echo "$id"
    fi
done

echo ""
echo "================================================================"
echo " Sweep IDs"
echo "================================================================"
for k in "${!SWEEP_IDS[@]}"; do
    short="${SWEEP_IDS[$k]##*/}"
    echo "  $k: ${SWEEP_IDS[$k]}"
    echo "     https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/sweeps/${short}"
done
echo ""

$DRY_RUN && { echo "[dry-run] Exiting without running agents."; exit 0; }

# ── Run agents sequentially ───────────────────────────────────────────────────
# Each `wandb agent --count N` blocks until N trials are complete, then exits.
echo "================================================================"
echo " Running agents (sequential — each finishes before the next)"
echo "================================================================"
echo ""

for sweep_key in encoder objectives patch lora; do
    [[ -n "$ONLY" && "$sweep_key" != "$ONLY" ]] && continue
    [[ -z "${SWEEP_IDS[$sweep_key]+x}" ]] && continue

    sid="${SWEEP_IDS[$sweep_key]}"

    case $sweep_key in
        encoder)    n=$N_ENCODER    ;;
        objectives) n=$N_OBJECTIVES ;;
        patch)      n=$N_PATCH      ;;
        lora)       n=$N_LORA       ;;
    esac

    echo "──────────────────────────────────────────────────────────────"
    echo " Starting $sweep_key agent: $n trials"
    echo " Sweep: $sid"
    echo "──────────────────────────────────────────────────────────────"

    WANDB_PROJECT="$WANDB_PROJECT" \
    EEGSCRFP_DATA_DIR="$EEGSCRFP_DATA_DIR" \
    EEGSCRFP_OUTPUT_DIR="$EEGSCRFP_OUTPUT_DIR" \
    $PYTHON -m wandb agent --count "$n" "$sid"

    echo ""
    echo " ✓  $sweep_key agent finished ($n trials)"
    echo ""
done

echo "================================================================"
echo " All sweeps complete."
echo " Dashboard: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/sweeps"
echo "================================================================"
