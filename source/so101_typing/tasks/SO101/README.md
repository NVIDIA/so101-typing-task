# SO-101 robot configuration

The keyboard simulation loads the pinned Workshop SysID USD built as
`/opt/so101/assets/workshop_so101.usdc` by [the Docker build](../../../../docker/typing-public.Dockerfile).
`SO101_TYPING_USD` selects the robot asset for every keyboard task; `so101_cfg.py`
and `actuator_profiles.py` configure its placement and drives. The older
`SO101_ROBOT_USD` remains available for archived recording and deployment
identity checks. It does not select the keyboard task robot.

The Workshop asset has different body frames and physics from the older robot
USD. Existing keyboard poses, jaw-tip offsets, calibration data and policy
checkpoints were measured with the older model and need revalidation before
using them for physical alignment or deployment. The robot is rotated 90 degrees
about Z in the task so its forward direction points toward the keyboard.

The original part meshes, Onshape metadata and URDF/MJCF sources are available
at the pinned [upstream model revision](https://github.com/TheRobotStudio/SO-ARM100/tree/eecbe3e0a9ebb23e25ad7b2759b03884c6660903/Simulation/SO101).
See [third-party attribution](../../../../THIRD_PARTY_NOTICES.md#so-101-robot-assets).
For robot setup and deployment, follow
[hardware preparation](../../../../docs/release/HARDWARE_PREPARATION.md).

The current default actuator profile is `gtc_berlin`, copied exactly from the
October 12 SysID result in `data/gtc_berlin.yaml`. The prior `anchorbench`
profile and its registered task IDs remain available. The fitted YAML specifies
stiffness, drive damping, armature, dry friction, and viscous friction for each
joint; motor effort and velocity limits retain the Workshop values (3.35 and
30 rad/s) because the fit does not specify them. The `./so101` CLI selects
`gtc_berlin` for fresh runs unless `--actuator` is supplied.
