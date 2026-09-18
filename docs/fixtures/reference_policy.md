# Historical 19k policy: fixture and joint mapping

The default download contains the AnchorBench `model_19000.pt` checkpoint and
its matching environment. It uses the original SO-101 fixed typing jaw, a closed
moving jaw, the Logitech MX keyboard and a **4.44 mm keyboard mat**. Its
**89/100 physical six-letter trials** apply to that original robot and fixture.
Simulation playback needs no physical robot; follow the [README](../../README.md#workflow-2-optionally-try-a-released-checkpoint).

## Frozen simulation contract

| Quantity | Value |
| --- | --- |
| Keyboard asset origin XYZ, robot-base frame | `[0.2625651584, 0.0121353200, 0.0017813943]` m |
| Asset roll / pitch / yaw | `[-5.0, 0.0506258799, -88.6058757454]` degrees |
| Joint order | shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll |
| Encoder-to-model offsets | `[0, 1.9336272481, -10.7735791533, 1.3647648917, 0]` degrees |

```text
Read:    q_model = q_LeRobot + offset
Command: q_LeRobot_goal = q_model_goal - offset
```

These values were selected together using recorded physical contacts and
simulation replay before policy training. The final scene passed **78/78
reset-per-key first-contact checks** on the original recordings. This does not
establish clean continuous replay, release or clearance for those recordings.
The final numeric constants are preserved, but an exact promoted optimizer
output is unavailable; the archived diagnostic fits were not the released result.

The pose is an effective simulation correction, not physical ruler instructions.
In particular, its roll does not establish a five-degree tilt of the physical
keyboard. The [reference drawing](reference-birdseye.svg) shows that model.

## Use with another robot

Calibrate your own robot with LeRobot and [qualify its alignment](../release/ALIGNMENT.md)
against this policy's trained scene. Do not copy another robot's calibration or
assume that its offsets transfer. Automatic mapping in software-only export
recognizes only this exact released checkpoint/environment pair; physical
execution still requires the documented alignment or supervised experimental route.

For a different fixture, [record and fit your scene](../release/REAL2SIM.md), then
train a policy for it. The September no-mat example uses a different fitted pose
and mapping; see its [physical reference](physical_reference_20260915.md) and
[results](../release/RESULTS.md).
