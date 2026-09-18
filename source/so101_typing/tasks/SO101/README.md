# SO-101 robot configuration

The simulator loads one assembled SO-101 USD, downloaded and checksum-verified
by [the Docker build](../../../../docker/typing-public.Dockerfile).
`SO101_ROBOT_USD` selects that asset; `so101_cfg.py` and `actuator_profiles.py`
configure the robot and its drives.

The original part meshes, Onshape metadata and URDF/MJCF sources are available
at the pinned [upstream model revision](https://github.com/TheRobotStudio/SO-ARM100/tree/eecbe3e0a9ebb23e25ad7b2759b03884c6660903/Simulation/SO101).
See [third-party attribution](../../../../THIRD_PARTY_NOTICES.md#so-101-robot-assets).
For robot setup and deployment, follow
[hardware preparation](../../../../docs/release/HARDWARE_PREPARATION.md).
