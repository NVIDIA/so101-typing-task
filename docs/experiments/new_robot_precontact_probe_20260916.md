# New-robot pre-contact probe — 2026-09-16

> Dated investigation record. Commands and results describe this workstation at the time; follow [the portable workflow](../../FIXTURE_SETUP.md) for a new setup. Raw recordings and `output/` evidence are local artifacts, not included in the source release.

Three torque-off recordings were collected on the September development follower with the
physical keyboard held fixed. Each recording contains one Q, one G, and one P
press. The historical analysis and replay below used the **earlier photo
candidate**, `output/fixture-photos-20260915/candidate-task.json`, SHA-256
`f4bd971e1add76d28442d5e7653bdf00f03d7d017f087fbcc4e3cdedb30d337f`.
That differs from the subsequently frozen Ctrl-aligned reference. The encoder
sample preceding receipt of each USB key-down was compared with a modeled key
target; this timing does not independently locate first physical contact.

Recordings:

```text
output/fixture-photos-20260915/manual-recordings/20260916T003449.479156Z/recording.json
output/fixture-photos-20260915/manual-recordings/20260916T003732.245749Z/recording.json
output/fixture-photos-20260915/manual-recordings/20260916T003804.510719Z/recording.json
```

Mean pre-contact position error across the three repetitions:

| key | direct conversion | historical July offsets |
| --- | ---: | ---: |
| Q | 38.22 mm | 13.61 mm |
| G | 38.46 mm | 14.82 mm |
| P | 39.23 mm | 15.25 mm |

The direct convention leaves the tip approximately 18–25 mm below the model
target. The historical offsets reduce the vertical error to approximately
5.1 mm (Q), 1.9 mm (G), and 2.4 mm (P). The result is repeatable across these
three hand-guided captures and is consistent with the improved physical reach
observed with the offset benchmark profile. It does not identify the cause by
itself.

This is a pre-contact geometry diagnostic, not an independently metered
encoder-to-model calibration: the target comes from the candidate keyboard model,
and hand placement/contact timing contribute residual error. It supports
keeping the historical values as an explicit benchmark option for this setup;
it does not authorize copying them to another SO-101 or replacing the robot's
native LeRobot calibration.

## Simulation versus hardware boundary (2026-09-15 local / 2026-09-16 UTC)

On 2026-09-15 local time, the policy-free FK trajectories for Q, G, and P were
generated from the earlier photo candidate and passed the Newton MJWarp simulation
gate. Those reports use model-space joint commands with
`joint_zero_offset_deg = [0, 0, 0, 0, 0]`. This is the correct simulator
contract; adding the historical offsets inside the simulator would double-count
the correction.

That result checks internal simulated FK/contact consistency. It does not prove
the physical keyboard placement matches, nor does it prove
that raw model targets can be sent directly as physical LeRobot commands. A
physical replay is a separate encoder-conversion test: the hardware adapter must
either use the direct convention or explicitly subtract the historical offsets.
The exact Q/G/P trajectories have not been used to certify direct hardware
replay. The historical-offset mode remains an explicit hardware diagnostic for
this robot and fixture.

## Real-trajectory replay result (2026-09-16 UTC)

The three real Q/G/P recordings were replayed through the earlier candidate with
prescribed robot poses. Direct conversion produced unrelated contacts (A,
V/SPACE, and L neighbourhoods) in all three runs. Historical offsets improved
the result but did not establish parity: Q matched once with an extra TAB
contact, G mapped to F, and P mapped to O (with additional adjacent contacts in
some runs). These replays therefore fail the intended Q/G/P scene-parity gate.

All six reports have `key_sequence_matches=false`. These results do not validate
the latest frozen placement. Keep the dated Ctrl-aligned reference fixed while
investigating the encoder/model mapping; do not move the keyboard to improve the
score.

## Recheck against the actual frozen reference — 2026-09-16 UTC

A fresh CPU-only comparison uses
`configs/tasks/physical-reference-20260915.json` (SHA-256 `3690b63ec8711698bbe3af39f4ae9236c1c5715e0fd9880bcce6a36121d62feb`),
the exact task selected for the new training run. No keyboard or joint parameters
were fitted. Mean FK tip position over each physical down/up interval was compared
with the model key centre depressed by an assumed 1.8 mm. This differs from the
single pre-event sample protocol above.

Across the two A–Z recordings (52 presses), mean 3D residual decreases from
41.44 mm direct to 12.63 mm with historical offsets. Mean signed vertical residual
(model tip minus modeled pressed-key centre) changes from -19.59 mm to -3.27 mm.
All three Q/G/P recordings show the same improvement: approximately 40–42 mm
direct versus 13 mm with historical offsets. This establishes a repeatable
improvement under the fixed scene, not a validated physical/model mapping.

Exact timestamp, per-key residuals, assumptions and input hashes are saved in
`frozen-offset-comparison-20260916.json` (local artifact: `output/fixture-photos-20260915/frozen-offset-comparison-20260916.json`).
This is a geometric calculation, not a new Newton contact replay or hardware test.
