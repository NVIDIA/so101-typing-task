# SO-101 robot model sources

For robot setup, training and deployment, follow the [main guide](../../../../README.md).
The typing runtime loads the pinned robot USD selected by `SO101_ROBOT_USD` in
`docker/typing-public.Dockerfile`. The local URDF, MJCF, meshes and converted USD
preserve model sources and conversion provenance.

`scene.xml` loads `so101_new_calib.xml`, whose virtual joint zeros are at the
middle of each joint's range. Editing these source files does not switch the
runtime asset or calibrate the physical robot. Use the documented
[LeRobot calibration and scene-fitting workflow](../../../../docs/release/HARDWARE_PREPARATION.md).
LeRobot reports the gripper as 0–100 percent; the deployment adapter converts
that value to the model's angular joint coordinates.

The robot descriptions were generated from Onshape using
[onshape-to-robot](https://github.com/Rhoban/onshape-to-robot), with relative mesh
paths and base collision meshes removed. STS3215 motor properties are adapted
from [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini).
See [third-party attribution](../../../../THIRD_PARTY_NOTICES.md#so-101-robot-assets)
for the original robot design, upstream source and Apache-2.0 license.
