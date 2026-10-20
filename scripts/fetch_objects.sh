#!/usr/bin/env bash
# Downloads the chosen Google Scanned Objects (CC-BY 4.0), as MJCF from kevinzakka/mujoco_scanned_objects (MIT),
# into assets/. The folder names come from src/plan_and_act/scenes/objects.txt, one name on each line.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=assets/scanned_objects
LIST=src/plan_and_act/scenes/objects.txt
mapfile -t NAMES < <(grep -v '^\s*$' "$LIST")
missing=()
for n in "${NAMES[@]}"; do
  [ -f "$DEST/models/$n/model.xml" ] || missing+=("$n")
done
if [ ${#missing[@]} -eq 0 ]; then
  echo "already present: ${#NAMES[@]} objects in $DEST/models"
  exit 0
fi
if [ ! -d "$DEST/.git" ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/kevinzakka/mujoco_scanned_objects "$DEST"
fi
git -C "$DEST" sparse-checkout set --no-cone "${NAMES[@]/#//models/}"
echo "done: ${#NAMES[@]} objects in $DEST/models"
