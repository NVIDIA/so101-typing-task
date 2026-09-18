# SO-101 deployment tools

For new users, follow [hardware preparation](../../../docs/release/HARDWARE_PREPARATION.md)
and use `./so101 prepare-deployment` followed by `./so101 deploy`. Both training
for your own placement and the optional reference checkpoint use that workflow.
You calibrate your own robot with LeRobot; export generates the matching reset
pose from the saved training geometry and your chosen encoder convention.

The shared runner validates the checkpoint, environment, target length, geometry,
reset pose and timing contract before connecting. The public deployment command
also checks the local calibration file on every dry run and execution. Hardware
motion requires the explicit `--execute` option and operator confirmation.

For scene calibration, use [record and fit](../../../docs/release/REAL2SIM.md), then
[qualify alignment](../../../docs/release/ALIGNMENT.md). The modules here implement
measured-motion recording, fitting, replay, FK probes, export and policy execution.
Keep run outputs and local investigations outside this source directory.
