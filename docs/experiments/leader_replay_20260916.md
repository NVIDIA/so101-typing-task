# Powered leader recording

Powered recordings test whether arm support/loading changes real-to-sim
correspondence compared with hand-guided, torque-off recordings. Replay uses
**measured follower positions**, never leader positions or sent goals. No joint
offsets are added during recording or default replay.

The normal [scene-calibration workflow](../release/REAL2SIM.md) uses hand guidance
and needs no leader. For the optional powered comparison, the portable
`scripts/leader_scene_check.sh` launcher requires explicit follower/leader ports
and calibration IDs; use `--help` for each mode.

| Mode | Purpose |
| --- | --- |
| `check` | Check files and dependencies without opening serial devices |
| `calibrate` | Standard LeRobot leader-only calibration; leave the follower calibration unchanged |
| `inspect` | Read both arms without motor writes and compare starting positions |
| `record` | Supervised powered mirroring with follower measurement and key-event capture |
| `replay` | Replay measured follower motion in simulation |

## Recording contract

The recorder samples at 50 Hz and sends each leader pose directly to the follower,
without a recorder-side slew or catch-up delay. The follower servo determines the
physical response. Initial leader/follower separation is allowed and logged; the
follower moves toward the leader's pose when recording begins. Keep the sweep clear.
Leader gripper motion is not forwarded; keep the moving jaw closed.

A measured follower tracking error over 60 degrees or a sampling gap over 250 ms
stops control. These are diagnostic recorder limits, not the typing-policy
controller. Esc or Ctrl-C saves and stops. The recorder attempts to hold fresh
measured follower positions and leaves arm torque **on**; it does not return the
arm or release torque. Keep motor power accessible during supervision.

`recording.json` and the incremental log preserve measured follower angles/raw
counts, leader requests, sent goals, key events, timestamps, controller settings
and calibration identities. Calibration, PID gains and leader torque are not
changed by recording. Calibration is a separate explicit step.

## Collection and interpretation

Keep the base, keyboard, jaw and follower calibration fixed. Collect complete A–Z
presses in any order, pausing about half a second at each gently bottomed key and
lifting clear between presses. Use separate recordings for fitting and validation;
freeze the fit before evaluating the reserved recordings. Retain wrong/extra keys
and score actual events, not the intended sequence.

The powered September collection supplied 60 complete presses. Its initial
fixed-scene offset comparison used the first 27 for fitting and later accepted
presses for evaluation. The final joint-and-scene refinement subsequently used
all 60 plus a 33-press manual recording; those captures are therefore fitting data
for the [selected scene](scene_calibration_20260917.md), not held-out evidence.

Contact-centre and key-travel assumptions affect FK residuals. Powered and manual
captures can differ through loading, pose coverage and sampling; improved fit
alone does not identify the cause. Simulation contact replay also allows physics
to evolve between imposed poses. Keep geometric correspondence, command tracking
and simulated contact outcomes separate when comparing results.
