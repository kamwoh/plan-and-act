"""arm-mind: an LLM controls a simulated robot arm."""
import os

# MuJoCo reads this when it is first imported. EGL renders off screen without a window.
os.environ.setdefault("MUJOCO_GL", "egl")
