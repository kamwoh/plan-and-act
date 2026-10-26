#!/usr/bin/env bash
# Downloads the robot models into assets/: the arms and grippers from MuJoCo Menagerie (each folder has its own
# license, mostly Apache 2.0 or BSD), and the YAM Ultra from the I2RT SDK (MIT).
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=assets/mujoco_menagerie
FOLDERS=(
  franka_emika_panda franka_fr3 universal_robots_ur5e universal_robots_ur10e robotiq_2f85 kinova_gen3
  kuka_iiwa_14 rethink_robotics_sawyer flexiv_rizon4 robotis_omy_3m ufactory_xarm7 ufactory_lite6
  agilex_piper arx_l5 i2rt_yam trossen_vx300s trossen_wx250s trossen_wxai
)
if [ ! -d "$DEST/.git" ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie "$DEST"
fi
git -C "$DEST" sparse-checkout set "${FOLDERS[@]}"
echo "done: ${#FOLDERS[@]} folders in $DEST"

I2RT=assets/i2rt
if [ ! -d "$I2RT/.git" ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/i2rt-robotics/i2rt "$I2RT"
fi
git -C "$I2RT" sparse-checkout set i2rt/robot_models/arm/yam_ultra i2rt/robot_models/gripper/linear_4310
echo "done: the YAM Ultra in $I2RT"
