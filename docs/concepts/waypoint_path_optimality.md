(waypoint_path_optimality_note)=

# Shortest Paths Through Waypoints

A trajectory through Cartesian waypoints can reach every waypoint with several joint
configurations. Which ones it takes decides how far the joints travel. This note explains what
the shortest path is, why a planner that only looks at the next waypoint misses it, how the
{ref}`via-point planner <via_point_planner_note>` and the {ref}`multi-waypoint planner
<multi_waypoint_planner_note>` choose, and how close they get on paths built to be hard.

## The shortest IK chain

A trajectory passes one IK solution of every waypoint. The straight joint-space line between
two configurations is the shortest way between them, so no trajectory through the waypoints is
shorter than the shortest *IK chain*: the choice of one IK solution per waypoint with the
smallest sum of joint-space distances from the start state. Lengths in this note are Euclidean
distances in joint space, in radians.

The chain is a lower bound, not always a path: obstacles between two waypoints can force a
longer way, or another chain.

## Why looking one waypoint ahead fails

Take a joint with limits of ±354° that stands at 0° and has to turn the tool to 90°, then to
260°, then to 360°:

| Choice at the second waypoint | Joint positions | Total rotation |
| --- | --- | --- |
| Nearest solution, 260° | 0° → 90° → 260° → 0° | 520° |
| Full-turn variant, -100° | 0° → 90° → -100° → 0° | 380° |

360° is beyond the limit, so the third waypoint is only reachable at 0°. Turning on to 260° is
20° shorter at the second waypoint and costs 260° at the third. The same happens with arm
configurations: the elbow-up solution nearest to the first waypoint can force the arm to
reconfigure later, where another solution would have gone through.

A planner avoids this only if it

1. has the alternatives: every IK branch of a waypoint, including the full-turn variants
   `q ± 2π` that an IK solver does not return, and
2. chooses the solutions of all waypoints together, by dynamic programming over the layers of
   IK solutions, instead of one waypoint at a time.

Both planners did the second from the start. They lacked the first, and three more things kept
them from the shortest chain. `benchmark/waypoint_optimality_benchmark.py`, which plans paths
built to punish a planner that looks one waypoint ahead, found all of them:

| Cause | Planner | Change |
| --- | --- | --- |
| IK returns one of `q`, `q ± 2π`; the other is never considered | both | add the full-turn variants within the limits (`expand_revolutions`) |
| The fastest of eight widely different chains is returned, even when it is much longer | via-point | seed only chains at most 10 % longer than the shortest on the first attempt (`chain_length_slack`) |
| The first waypoint's IK solution is chosen without regard to the start state | multi-waypoint | count the way from the start state (`start_state_cost`) |
| An IK solution on the shortest chain is not found | both | 64 IK seeds and seeds from the previous waypoint (via-point); keep the solutions on the shortest chains only, so more can be searched (`max_layer_nodes`, multi-waypoint) |
| From a straight seed the optimizer cuts the corner at a waypoint and the shortest chain fails | via-point | every chain also gets a bent seed (`seed_bulge`) |
| An IK solution lies inside the range where the optimizer's joint limit cost is active, so the trajectory cannot reach it | via-point | drop such solutions when a waypoint has others (`joint_limit_margin`) |

## How close the planners get

The benchmark compares each planner's IK chain with the shortest chain through a dense set of
IK solutions, and with `greedy`, the chain of a planner that always moves to the nearest IK
solution of the next waypoint. The {ref}`measured results <waypoint_planner_benchmarks>` show
three things.

Before these changes both planners were often worse than `greedy`, although both search all
waypoints together. The search was not the problem; the IK solutions it had to choose from
were. With the full-turn variants and the other changes, the via-point planner's chain is
within about 1 % of the shortest on average, on the UR10e as on the Franka; the multi-waypoint
planner's is within about 6 %.

Shorter chains are also faster motions. On the UR10e the via-point planner's motions became
16 % to 35 % shorter in time, family by family.

The multi-waypoint planner's planning time fell by half or more, because it keeps only the IK
solutions on the shortest chains and so solves fewer segment problems. It remains seconds,
against a fraction of a second for the via-point planner.

## Limits of the chain search

**The length measure.** Euclidean joint-space distance counts several joints moving a little
as shorter than one joint moving a lot. On the 0° → 90° → 270° → 360° path the shortest chain
by this measure changes the arm configuration at the second waypoint (5.85 rad) instead of
turning the wrist back and forth in place (6.28 rad), although the joints turn 35 % more in
total. Both avoid the 540° of the nearest-solution choice. `chain_metric` selects another
measure; see the next section.

**Obstacles between waypoints.** The chain search knows about collisions at the waypoints
only. Around a pillar the via-point planner follows the shortest chain, but its trajectory is
longer in joint space than that of the multi-waypoint planner, which optimizes every edge
separately and so also learns which chains are cheap to drive. When no way around is found
along the shortest chains, the via-point planner's retries take longer chains.

**Redundant arms.** A 7-joint arm has a continuum of IK solutions per waypoint, and the chain
can only be as short as the solutions that were sampled. This is why the via-point planner
uses 64 IK seeds by default: with 32 its chains on the Franka are about 5 % longer.

**Motion time.** The multi-waypoint planner chooses its chain by the summed duration of the
segment trajectories, not by length, so a longer chain can be the faster one for it. It stops
at every waypoint; the via-point planner does not. Neither is faster on every kind of path:
around obstacles the multi-waypoint planner's motions tend to be faster.

**The reference.** It is the shortest chain through the IK solutions that were found. For the
UR10e, with a finite number of IK solutions per pose, that is the shortest chain; for the
Franka it is a dense sample of it.

## Choosing a chain metric

`chain_metric` sets the length of a joint step `Δq` in the chain search:

| Metric | Length of `Δq` | Shortest chain |
| --- | --- | --- |
| `euclidean` | `‖Δq‖₂` | moves many joints a little rather than one joint a lot |
| `total_rotation` | `Σ_j abs(Δq_j)` | turns the joints least in total |
| `time` | `max_j abs(Δq_j) / v_j` | is fastest at the joint velocity limits, without acceleration |
| `max_norm` | `max_j abs(Δq_j)` | has the smallest largest joint motion |

The via-point planner uses the metric for the chain search, for the chain length slack and for
spacing the waypoints along the trajectory. The multi-waypoint planner uses it only to keep
the solutions on the shortest chains (`max_layer_nodes`); it chooses its chain by the duration
of the optimized segments, so the metric changes its result little.

A chain that is short in one metric can be long in another. `total_rotation` keeps the arm
configuration and turns single joints, which gives the slowest motions. `time` and `max_norm`
spread the motion over the joints and give the fastest ones, a few percent faster than
`euclidean`, at the price of much more total rotation. `euclidean` stays the default: it is
never far from the fastest motion, and its chains are the ones the trajectory optimizer
follows best.

The trajectory optimizer sees the waypoints, not the chain, and its smoothness cost favours
`euclidean`-like paths. For the other metrics the trajectory can therefore leave the chosen
chain: on the Franka the via-point planner's `total_rotation` result can be longer in its own
metric than the nearest-solution chain. Picking, among the optimized trajectories, the one
whose chain is shortest in the metric did not help: it made the chains longer and the motions
slower, so the planner returns the fastest trajectory. The
{ref}`benchmark results <waypoint_planner_benchmarks>` list every metric on both robots.
