# Powered leader recording — September 16, 2026

> Dated investigation record. Commands and results describe this workstation at the time; follow [the portable workflow](../../FIXTURE_SETUP.md) for a new setup. Raw recordings and `output/` evidence are local artifacts, not included in the source release.

Compare powered follower motion with the existing torque-off recordings while
keeping the keyboard/base placement, model, fixed jaw and follower calibration
unchanged. Replay uses **measured follower positions**, never leader positions or
sent commands. There are no added joint offsets in recording or default replay.
This tests whether hand support/loading affects scene correspondence; it does not
assume that teleoperation fixes the mismatch.

The original workstation launcher inferred the leader from attached devices.
The current portable launcher requires explicit follower/leader ports and
calibration IDs; see `./scripts/leader_scene_check.sh --help`. The shorthand
commands below describe that historical session, not complete new-user commands.
Use [the portable workflow](../../FIXTURE_SETUP.md) for your own setup.

1. `./scripts/leader_scene_check.sh check` checks files and dependencies without
   opening serial devices. A newly attached leader needs its own calibration.
2. `./scripts/leader_scene_check.sh calibrate` runs standard LeRobot **leader-only**
   calibration. Follow its middle-position and full-range prompts. The follower's
   calibration is unchanged.
3. `./scripts/leader_scene_check.sh inspect` reads both arms without motor writes.
   Position the unpowered leader near the follower's displayed five joint angles
   before recording. Alignment is reported for reference but is not required;
   recording sends the leader's current angles directly to the follower.
4. `./scripts/leader_scene_check.sh record` starts supervised powered mirroring.
   Move slowly through **Q, G, P**, with a distinct hover, normal fully depressed
   press and release at each key; repeat the sequence. Keep the moving jaw closed.
   Leader gripper motion is not forwarded. Press Esc or Ctrl-C to save and stop.
5. `./scripts/leader_scene_check.sh replay` opens the latest completed recording
   in Newton with `configs/tasks/physical-reference-20260915.json` and zero added
   offsets. It moves no physical robot. The viewer exits after one replay.

Recording lasts up to 120 seconds by default (`--seconds` to change it) and
samples at 50 Hz. Each leader pose is sent directly to the follower every cycle;
there is no recorder-side slew or catch-up delay. The follower's own servo handles
the physical response. A measured follower tracking error over 60° or a sampling
gap over 250 ms stops control. Initial leader/follower separation is allowed and
logged. These are diagnostic limits, not the policy deployment controller. On stop,
the recorder attempts to hold freshly
measured follower positions and leaves arm torque on. It does not return the arm
or release torque. Keep access to motor power during the supervised test.

Outputs are under `output/fixture-photos-20260915/leader-recordings/<UTC>/`.
`recording.json` and the incremental log retain measured follower angles/raw
counts, leader requests, sent goals, key events, timestamps, initial controller
settings and both calibration identities. Calibration, PID gains and leader
torque are not modified by the recorder; calibration is a separate explicit step.

Judge the prescribed jaw/key geometry first. The existing Newton contact replay
also lets physics evolve between imposed poses, so key-event differences can
include within-step drift. Matching recorded geometry is distinct from command
tracking and from reproducing simulated contact dynamics.

## Fixed-keyboard mapping collection — 2026-09-16, 15:02 PDT

Collect four separate powered recordings, aiming for 26 complete letter presses
per recording (104 total). Keep the physical keyboard/base, fixed typing jaw,
simulation placement, model and follower calibration unchanged. Keep the moving
jaw closed. Record without added joint offsets; corrections are evaluated offline
against the same measured follower data.

Update at 15:13 PDT: the operator prefers free-order collection. The sequences
below are optional coverage guides. Capture `20260916T220434.751544Z` was analyzed
using a chronological split declared before fitting; see the latest result in
[the mapping investigation](mapping_plan_20260916.md). For additional captures,
reserve a complete fresh recording before using it to tune offsets.

Run this command once per pass; Esc saves and ends each recording. Each run has
a five-minute limit and a unique timestamp directory under this collection root.

```bash
./scripts/leader_scene_check.sh record --seconds 300 --output-root output/fixture-photos-20260915/mapping-collection-20260916
```

| Pass | Use | Letter order |
| --- | --- | --- |
| 1 | Fit | `QWERTYUIOP ASDFGHJKL ZXCVBNM` |
| 2 | Fit and check parameter stability | `MNBVCXZ LKJHGFDSA POIUYTREWQ` |
| 3 | Reserved validation | `QAZ WSX EDC RFV TGB YHN UJM IK OL P` |
| 4 | Reserved validation | `P LO KI MJU NHY BGT VFR CDE XSW ZAQ` |

Spaces group the sequence for reading; do not press Space. Aim at each key's
centre, pause above it, lower slowly to normal full depression, hold steady for
about 0.5 seconds, then fully release and lift clear. Do not force the arm farther
after the key bottoms out. The changing order samples different approaches. If a
wrong or extra key registers, keep the recording and note it; analysis uses actual
events, not the intended sequence. Avoid pressing keys by hand during capture.

Assign roles by collection order before analysis; use passes 1–2 to estimate the
correction and freeze it before scoring passes 3–4. Check recording status, source
and calibration identity, event completeness, sample gaps and within-press motion
before fitting. Extract settled contact windows, equally weight presses, and
report sensitivity to the selected window. Key-down receipt alone does not locate
the instant of mechanical bottoming. Key centre and full 1.8 mm depression remain
reference assumptions, not independently measured contact coordinates.

Compare zero offsets, the unchanged historical elbow-only correction, and the
unchanged full historical correction. Then test the smallest useful fitted family:
elbow alone, followed by shoulder lift/elbow/wrist flex if needed; pan and roll
stay fixed at zero initially. Choose the candidate using passes 1–2 and parameter
stability, not the reserved results. Do not adjust keyboard placement or jaw
geometry alongside the offsets. Report each reserved run's signed and absolute Z
error, XY/3D error, tail errors and per-key regressions. Consistent improvement
across recordings is the objective; identical fitted values and zero residual are
not required. Similar fit error with very different angles indicates ambiguity.

These data can qualify an empirical correction for this robot, calibration and
typing workspace. They cannot alone prove universal mechanical joint zeros.
After freezing a useful mapping, separate supervised physical FK presses must
check command tracking and actual key contact before deployment claims.

## Preparation evidence

On September 16, the leader was identified separately from the development
follower. The operator completed standard leader calibration. Read-only
inspection at `20260916T210902.466905Z` verified both live calibration snapshots,
leader torque off and follower jaw closed; starting alignment still needed
adjustment. No teleoperation recording was executed during preparation.

77 targeted host tests passed. The rebuilt Newton runtime accepted a clearly
labeled synthetic powered-schema fixture made from six existing torque-off
samples, imposed zero-offset measured poses and completed a 0.101-second replay.
This checks software compatibility only; its key-contact result is not physical
teleoperation evidence. Logs are in `output/leader-teleop-preparation-20260916/`.
