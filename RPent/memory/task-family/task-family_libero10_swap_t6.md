---
id: task-family_libero10_swap_t6
scope: task-family
suite: libero10
regime: swap
task_id: 6
task_language: put the white mug on the plate and put the chocolate pudding to the
  right of the plate
evidence:
  cells:
  - 10_swap_t6_s0
  attempts: 1
  solved_seeds:
  - 0
  failed_seeds: []
confidence: single-shot
related:
- task-family_libero10_task_t6
---

## Applicable pattern
Two-clause LIVING_ROOM tabletop conjunction: (1) place the WHITE mug
(porcelain_mug_1, gray/white DIMPLED mug) On the plate, and (2) place the
chocolate-pudding box "to the right of" the plate. A RED mug (red_coffee_mug_1)
is a DISTRACTOR — "white mug" is the target; do not grab the red one. This is the
same base task as libero10 task-t6 (there the target was the RED mug and the
white/porcelain mug was the distractor — the swap perturbation flips which mug
is named). The real difficulties are the two predicate subtleties, not the grasps:
the "right" side SIGN, and confirming the mug On() half.

## Winning technique
Order here was MUG FIRST, then pudding (worked cleanly; the safer general rule
from task-t6 is to place the non-mug object first and never carry over the
placed mug — see Fragility). Sequence:
1. Pre-position over the white mug; grasp-only `pi0_pick "grasp the white mug"`.
   If pi0 descends over the mug mouth but WON'T close/lift (grip stays ~0.08),
   shift eef_y toward the NEAR rim: eef_y = mug_y + ~0.045, then re-`pi0_pick`.
   That makes pi0 rim-hook the thin wall (grip closes to ~0.003, lift ~0.17).
   `set_gripper +1` to firm.
2. Measure the held-mug offset (back_project the held mug body vs eef), carry
   +1-hold over the plate CENTER, descend until OSC stalls (~z0.52, mug base on
   plate top ~0.444); `release`; retreat STRAIGHT UP (step_clip 0.015).
   Success: mug upright, base within the plate rings (~plate center).
3. Pre-position over the pudding box; `pi0_pick "pick up the chocolate pudding
   box"` (chunks 10-12) -> lift ~0.05-0.15, grip ~0.047. `set_gripper +1`.
   Held box xy ~= eef xy (negligible offset).
4. Carry to the +y (IMAGE-RIGHT) side of the plate, x-aligned with plate center.
   Lift HIGH (z~0.64, above the placed mug rim ~0.544 and the red mug ~0.53)
   before traversing +y so the hanging box never grazes the mug. Descend toward
   ~(plate_x, +0.13). The predicate fired DURING the descent while still holding
   the box (release not required) once mug-On-plate AND pudding-right-of-plate
   both held.

## Magic numbers
- Frame LIVING_ROOM: eef home z~0.681; table top~0.42; plate top~0.444.
- plate center ~ region-median of the ring pixels; plate radius ~0.10.
- mug rim-grasp: eef_y = mug_y + 0.045 (band 0.035-0.050) to force pi0 to close.
- mug pi0_pick max_chunks=8-10; on a non-closing hover, RE-ISSUE rather than raise chunks.
- mug place: descend to OSC stall z~0.52 (mug base on plate); step_clip 0.015-0.02.
- pudding pi0_pick max_chunks=10-12; lift_thresh 0.04; grip~0.047; held xy ~= eef xy.
- carry/retreat z 0.60-0.64; single-step |Δxy| < 0.30 ALWAYS (waypoint long moves).
- "right of the plate" = +y (IMAGE-RIGHT), NOT egocentric -y. Place at +y.

## Failure modes
| symptom | root cause (A#) | fix |
|---|---|---|
| Pi0 descends over the white mug mouth, grip stays ~0.08, no lift (steps 3-4) | eef centered on mug axis; pi0 won't close on a wide rim from center (A1) | shift eef_y to mug_y+0.045 so pi0 rim-HOOKS; re-issue pi0_pick |
| Perfect mug-on-plate + pudding at -y (image-left), libero_terminated FALSE (twice: far 0.19 and close 0.14) | wrong "right" side; "right" here = +y image-right, not egocentric -y (A1) | place pudding on the +y (IMAGE-RIGHT) side of the plate |

## Re-localization per scene
- white mug: agentview hi-res, gray/white DIMPLED-texture mug. `segment "the white
  ceramic mug"` scored 0.88 here. Grasp needs the eef_y+0.045 near-rim offset.
- red mug: tall red-with-white-pattern mug = DISTRACTOR (this task). Ignore.
- plate: white disc with red concentric rings; region back_project the rings for a
  rim-unbiased center. radius ~0.10.
- pudding: small brown box reading "CHOCOLATE PUDDING"; pi0 grasps top-down; the
  held box hangs ~directly under the eef (xy offset ~0).
- This run's absolute xyz MUST NOT be cached (swap re-randomizes); counter-examples only.

## Fragility flags
- Most fragile step: the "right" side sign. It is +y (image-right) here, CONTRARY
  to the egocentric memory rule. Verify empirically: place on your guessed side,
  and if the OTHER half is clearly satisfied and it still won't fire, flip the y-sign.
- The white-mug grasp: without the eef_y+0.045 near-rim offset pi0 hovers and never
  closes. Re-issue pi0_pick after nudging eef_y; do not just raise max_chunks.
- Note: unlike task-t6, the predicate here fired while the eef was still low and
  HOLDING the pudding (no eef-retreat-clear needed for the pudding half).

## Difficulty and reliability
- Single attempt, single episode, no resets. ~34 primitive steps. Should be
  near single-shot with the known levers (white=target not red, rim-grasp offset,
  +y side). Budget one probe for the "right" side on a fresh seed.

## Cross-refs
- [[libero10-left-right-sign-verify-empirically]]
- [[task-family_libero10_task_t6]]
