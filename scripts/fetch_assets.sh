#!/usr/bin/env bash
# Downloads the Franka Panda model from MuJoCo Menagerie (Apache 2.0) into assets/.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=assets/mujoco_menagerie
if [ -f "$DEST/franka_emika_panda/panda.xml" ]; then
  echo "already present: $DEST/franka_emika_panda"
  exit 0
fi
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie "$DEST"
git -C "$DEST" sparse-checkout set franka_emika_panda
echo "done: $DEST/franka_emika_panda"
