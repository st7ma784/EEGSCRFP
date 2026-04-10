#!/bin/bash
# SBATCH job template for a single W&B sweep agent — EEGSCRFP subproject.
# Do NOT submit this directly -- use hc_sweep_launch.sh instead.
#
# Each job runs one (or more) W&B sweep trials, then exits.

#SBATCH --job-name=scrfp-sweep
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --signal=USR1@60
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=s.mander3@lancaster.ac.uk
#SBATCH --partition=gpu-medium

set -euo pipefail

SWEEP_ID=${1:?SWEEP_ID required (passed by hc_sweep_launch.sh)}
TRIALS_PER_AGENT=${2:-1}

PROJECT_ROOT=$global_storage/EEGViewer/EEGSCRFP

mkdir -p "${PROJECT_ROOT}/logs"

# ── Conda environment ──────────────────────────────────────────────────────────
module add opence

CONDA_BASE=$(conda info --base 2>/dev/null) || CONDA_BASE="$global_storage/miniconda"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$global_storage/opence"

# ── Project paths ──────────────────────────────────────────────────────────────
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── Data directory ────────────────────────────────────────────────────────────
# EEGSCRFP_DATA_DIR is read by sensor_count.py and other experiment scripts
# as the default for --data-dir.  Point at the narrative EEG data on scratch.
if [[ -n "${global_scratch:-}" && -d "${global_scratch}/EEG/data/data/narrative" ]]; then
    export EEGSCRFP_DATA_DIR="${global_scratch}/EEG/data/data/narrative"
elif [[ -n "${global_storage:-}" && -d "${global_storage}/EEG/data/data/narrative" ]]; then
    export EEGSCRFP_DATA_DIR="${global_storage}/EEG/data/data/narrative"
fi

if [[ -n "${EEGSCRFP_DATA_DIR:-}" ]]; then
    echo "EEGSCRFP_DATA_DIR=${EEGSCRFP_DATA_DIR}"
    ls "${EEGSCRFP_DATA_DIR}" 2>/dev/null | head -5 \
        || echo "  (directory empty or inaccessible)"
else
    echo "EEGSCRFP_DATA_DIR not set — experiment will use synthetic prompts"
fi

export EEGSCRFP_OUTPUT_DIR="${global_scratch:-/tmp}/eegscrfp_outputs"
mkdir -p "${EEGSCRFP_OUTPUT_DIR}"
echo "EEGSCRFP_OUTPUT_DIR=${EEGSCRFP_OUTPUT_DIR}"

export PYTHONUNBUFFERED=1

# ── W&B credentials check ─────────────────────────────────────────────────────
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    if ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
        echo "ERROR: No W&B credentials found." >&2
        echo "  Set WANDB_API_KEY in ~/.bashrc, or add to ~/.netrc:" >&2
        echo "    machine api.wandb.ai login user password <your-key>" >&2
        exit 1
    fi
    echo "W&B credentials: found in ~/.netrc"
else
    echo "W&B credentials: WANDB_API_KEY is set"
fi

# ── Timing + exit trap ────────────────────────────────────────────────────────
JOB_START=$(date +%s)
trap 'RC=$?; ELAPSED=$(( $(date +%s) - JOB_START ));
      echo "";
      echo "=== Job finished: exit $RC  |  elapsed $(( ELAPSED/3600 ))h$(( ELAPSED%3600/60 ))m$(( ELAPSED%60 ))s ===";
      if [[ $RC -ne 0 ]]; then
          echo "FAILED -- check lines above for the traceback.";
      fi' EXIT

# ── Header ────────────────────────────────────────────────────────────────────
echo "================================================================"
echo " EEGSCRFP Sweep Agent"
echo "================================================================"
echo " Job ID:      $SLURM_JOB_ID"
echo " Node:        $SLURM_NODELIST"
echo " Sweep:       $SWEEP_ID"
echo " Trials:      $TRIALS_PER_AGENT"
echo " Started:     $(date)"
echo " Python:      $(which python)  $(python --version 2>&1)"
echo " Conda env:   ${CONDA_DEFAULT_ENV:-unknown}"
echo " Working dir: $(pwd)"
echo "----------------------------------------------------------------"
echo " GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader 2>/dev/null \
    | sed 's/^/   /' || echo "   (nvidia-smi unavailable)"
echo "----------------------------------------------------------------"
echo " Disk (global_storage):"
df -h "$global_storage" 2>/dev/null | tail -1 \
    | awk '{printf "   %s used of %s (%s free)\n", $3, $2, $4}' \
    || echo "   (df unavailable)"
echo " RAM:"
free -h 2>/dev/null \
    | awk '/^Mem:/{printf "   %s used of %s (%s free)\n", $3, $2, $4}' \
    || echo "   (free unavailable)"
echo "================================================================"

# ── Import sanity check ────────────────────────────────────────────────────────
echo "Checking key imports..."
python - <<'PYEOF'
import sys, importlib
for mod in ("torch",
            "src.model.sparse_attention",
            "src.metrics.network_patches",
            "src.metrics.cka_metrics"):
    try:
        importlib.import_module(mod)
        print(f"  [ok] {mod}")
    except Exception as e:
        print(f"  [FAIL] {mod}: {e}", file=sys.stderr)
        sys.exit(1)
import torch
print(f"  [ok] CUDA available: {torch.cuda.is_available()}  "
      f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
PYEOF
echo ""

# ── Run the agent ──────────────────────────────────────────────────────────────
echo "Starting wandb agent (sweep: $SWEEP_ID, trials: $TRIALS_PER_AGENT)..."
srun wandb agent --count "${TRIALS_PER_AGENT}" "${SWEEP_ID}"
