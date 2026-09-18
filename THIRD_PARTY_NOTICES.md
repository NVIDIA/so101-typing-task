# Third-party notices

Third-party material retains the licenses identified below. The repository's
[BSD-3-Clause license](LICENSE) does not replace these asset or dependency terms.

## Isaac Lab

The Isaac Lab-derived task code in `source/so101_typing/tasks/`, training and
evaluation scripts in `scripts/reinforcement_learning/rsl_rl/`, and
`runtime/benchmark-runtime.patch` retain their upstream copyright and
BSD-3-Clause license notices. File headers identify the applicable copyright
years and The Isaac Lab Project Developers.

The runtime source ancestor is
[`ooctipus/IsaacLab` at `6f991d4becf764b151e0a1c775561ddd9c406f72`](https://github.com/ooctipus/IsaacLab/tree/6f991d4becf764b151e0a1c775561ddd9c406f72).
Its [license](https://github.com/ooctipus/IsaacLab/blob/6f991d4becf764b151e0a1c775561ddd9c406f72/LICENSE)
is reproduced in the root `LICENSE`. The patch contains local runtime changes;
its scope and source snapshot are recorded in [runtime/README.md](runtime/README.md).

## SO-101 robot assets

The meshes and `.part` files in `source/so101_typing/tasks/SO101/assets/`,
the `so101_new_calib` URDF/MJCF descriptions,
`joints_properties.xml`, `scene.xml`, and the upstream portions of that
directory's README come from
[TheRobotStudio/SO-ARM100, `Simulation/SO101`](https://github.com/TheRobotStudio/SO-ARM100/tree/eecbe3e0a9ebb23e25ad7b2759b03884c6660903/Simulation/SO101).
The SO-101 design is credited upstream to The Robot Studio in collaboration
with Hugging Face. These assets are licensed under
[Apache License 2.0](licenses/Apache-2.0.txt).

The bundled mesh, `.part`, URDF, and XML files match that upstream revision.
The README has local usage changes. The typing runtime uses the separately
downloaded, pinned USD described below.

The robot description's upstream attribution to
[Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini) for STS3215 motor
properties is retained. Open Duck Mini's
[license](https://github.com/apirrone/Open_Duck_Mini/blob/b23317a485b3cec7d8417f352478778b3475173c/LICENSE)
is also Apache-2.0; the same full license text is included above. The
`onshape-to-robot` generator and the motor-gain reference remain credited in
the model README and XML comments.

## Logitech MX Keys visual asset

The geometry in
`source/so101_typing/tasks/keyboards/data/mx_keys/mx_keys_asset.npz` and the
textures in that directory's `textures/` subdirectory derive from
["Logitech MX Keys" by ArkinRS](https://sketchfab.com/3d-models/logitech-mx-keys-67901d82781f48349f1f61ce78a6b6e0),
published on Sketchfab under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The full license, including its warranty disclaimer, is reproduced in
[licenses/CC-BY-4.0.txt](licenses/CC-BY-4.0.txt).

Creator: [ArkinRS](https://sketchfab.com/ArkinRS). The creator credits Poligon
for surface imperfections. The original USDZ embeds this creator, model URL,
and license attribution.

Local modifications convert the USDZ to NPZ, segment the merged keycap mesh
into individual keys, change coordinates and units, map keys into the task's
slot order, extract the case meshes and textures, and modify the key-label
base-color texture. The converter and `manifest.json` describe the conversion.
This attribution concerns the visual model and textures, not the separately
recorded calibration measurements in the same directory.

## Components installed separately

The Docker build and hardware setup install upstream packages rather than
vendoring their source into this repository. The exact runtime pins are in
[docker/typing-public.Dockerfile](docker/typing-public.Dockerfile) and
[runtime/hardware.lock](runtime/hardware.lock). Principal components include:

| Component | Upstream license |
| --- | --- |
| Newton | [Apache-2.0](https://github.com/ooctipus/newton/blob/5205aa49fb900e124f06122bf7e82ed7686a1fde/LICENSE.md) |
| NVIDIA Warp | [Apache-2.0](https://github.com/NVIDIA/warp/blob/main/LICENSE.md) |
| MuJoCo / MuJoCo Warp | [Apache-2.0](https://github.com/google-deepmind/mujoco/blob/main/LICENSE) / [Apache-2.0](https://github.com/google-deepmind/mujoco_warp/blob/main/LICENSE) |
| RSL-RL | [BSD-3-Clause](https://pypi.org/project/rsl-rl-lib/5.4.1/) |
| LeRobot | [Apache-2.0 and included third-party notices](https://github.com/huggingface/lerobot/blob/b6ec0060779550c0a157ae34feb89e0cf86012a8/LICENSE) |

These links summarize the named projects; the licenses and notices shipped
with the installed versions govern those packages and their dependencies.
The NVIDIA Isaac Lab base image also carries its own component licenses and
Isaac Sim terms. This source repository's BSD license does not license the
container's entire software stack.

## Downloaded robot asset and benchmark artifacts

The Docker build downloads `robot_assets/so101_no_camera_new_calib.usd` from
[`nvidia/Anchor-Lab` revision `647edd5787cd764cdc041103ad282dc59214d919`](https://huggingface.co/datasets/nvidia/Anchor-Lab/tree/647edd5787cd764cdc041103ad282dc59214d919).
The pinned dataset card declares Apache-2.0. The SO-ARM100 attribution above
also identifies the underlying robot design. This USD is downloaded during
the build and is separate from the converted USD files bundled in this tree.

Checkpoints and evaluation bundles fetched by
`scripts/download_benchmark_artifacts.sh` are published separately in
[`VigneshBhavan/so101-keyboard-typing-benchmark`](https://huggingface.co/datasets/VigneshBhavan/so101-keyboard-typing-benchmark).
The dataset card at the revision recorded in
[docs/release/artifact_revision.json](docs/release/artifact_revision.json)
does not declare a license for those artifacts. This repository's source
license does not establish a license for the downloaded weights or data.
