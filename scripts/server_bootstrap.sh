#!/usr/bin/env bash
# Runbook for setting up WearSeizure-1D on SERVER-02 (see docs/SERVER_INVENTORY.md).
#
# THIS FILE IS NOT MEANT TO BE PIPED STRAIGHT INTO A SHELL. It is a
# step-by-step reference: read each section, run the commands that fit your
# situation, and in particular *look at* the output of the review steps
# before running anything destructive. The cleanup section is written so a
# `bash server_bootstrap.sh` alone still requires a positional argument
# ("confirm-delete") to actually delete anything -- running it with no
# arguments only prints what *would* be reviewed/deleted and does nothing.
set -euo pipefail

step() { echo; echo "=== $1 ==="; }

# -----------------------------------------------------------------------
# 1. Review (and only if you're sure, delete) old failed experiment dirs.
#    Scope is exactly these three -- nothing else under ~/Manh is touched.
# -----------------------------------------------------------------------
OLD_DIRS=(
  "$HOME/Manh/1D-CNN-Accelerator-for-EEG_Detection"
  "$HOME/Manh/1D-CNN-Accelerator-for-EEG_Detection-q1"
  "$HOME/Manh/1D-CNN-Accelerator-for-EEG_Detection-main"
)

step "Reviewing old experiment directories (read-only)"
for d in "${OLD_DIRS[@]}"; do
  if [ -d "$d" ]; then
    echo "--- $d ---"
    du -sh "$d" 2>/dev/null || true
    ls -la "$d" | head -20
  else
    echo "--- $d --- (not found, skipping)"
  fi
done

if [ "${1:-}" = "confirm-delete" ]; then
  step "Deleting old experiment directories (you passed confirm-delete)"
  for d in "${OLD_DIRS[@]}"; do
    if [ -d "$d" ]; then
      echo "Removing $d"
      rm -rf -- "$d"
    fi
  done
else
  step "Skipping deletion"
  echo "Re-run as: bash server_bootstrap.sh confirm-delete"
  echo "...once you've confirmed the listing above is exactly what should go."
fi

# -----------------------------------------------------------------------
# 2. Clone the repo (replace with the actual URL once created; see the
#    session's "gh repo create" output).
# -----------------------------------------------------------------------
step "Clone (run manually, URL depends on the created GitHub repo)"
cat <<'EOF'
  cd ~/Manh
  git clone https://github.com/NADUNGVN/WearSeizure-1D.git
  cd WearSeizure-1D
  # Discipline: pin to a specific reviewed commit, never a floating branch tip
  # (README "Training server" / point 3). Replace <sha> with the commit you
  # were told to use.
  git checkout <sha>
EOF

# -----------------------------------------------------------------------
# 3. Conda env + PyTorch matching this box's CUDA driver.
# -----------------------------------------------------------------------
step "Environment setup (run manually)"
cat <<'EOF'
  nvidia-smi   # check the "CUDA Version" field (max supported by the driver)

  # chbmit-cnn is the environment actually in use on SERVER-02 (confirmed
  # 2026-08); it already has PyTorch 2.5.1+cu121 working against the Quadro
  # RTX 8000. Only create it if it does not exist -- `conda env list` first.
  conda create -n chbmit-cnn python=3.11 -y   # skip if it already exists
  conda activate chbmit-cnn

  # Pick a CUDA build <= what nvidia-smi reported. All four servers run
  # drivers from 580.x-595.x, which comfortably support CUDA 12.1+, so cu121
  # is a safe default; use cu124 if nvidia-smi reports a newer CUDA version
  # and you want the newer toolkit.
  pip install torch --index-url https://download.pytorch.org/whl/cu121

  pip install -r environment/requirements-common.txt
  pip install -e .
EOF

# -----------------------------------------------------------------------
# 4. Environment variables (see docs/SERVER_INVENTORY.md for why these
#    exact paths).
# -----------------------------------------------------------------------
step "Environment variables (run manually, or add to ~/.bashrc / .env)"
cat <<'EOF'
  export CHBMIT_RAW_DIR=~/Manh/datasets/CHB-MIT/1.0.0
  export WEARSEIZURE_ARTIFACTS_DIR=~/Manh/WearSeizure-1D-artifacts
EOF

# -----------------------------------------------------------------------
# 5. Validate the real data ingests correctly -- read-only, no training.
#    Expect: 13 subjects (Appendix A), ~77 seizure events, ~599.5h total.
# -----------------------------------------------------------------------
step "Validate manifest + splits against real CHB-MIT data (run manually)"
cat <<'EOF'
  python scripts/make_manifest.py profile=server data=chbmit
  python scripts/make_splits.py profile=server data=chbmit split=patient_specific_loso_edf
  python scripts/make_splits.py profile=server data=chbmit split=zero_shot_loso_subject
EOF

echo
echo "Done printing the runbook. Nothing beyond the directory review (and"
echo "deletion, if you passed confirm-delete) was actually executed."
