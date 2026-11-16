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
them from the shortest chain. The benchmark below found all of them:

| Cause | Planner | Change |
| --- | --- | --- |
| IK returns one of `q`, `q ± 2π`; the other is never considered | both | add the full-turn variants within the limits (`expand_revolutions`) |
| The fastest of eight widely different chains is returned, even when it is much longer | via-point | seed only chains at most 10 % longer than the shortest on the first attempt (`chain_length_slack`) |
| The first waypoint's IK solution is chosen without regard to the start state | multi-waypoint | count the way from the start state (`start_state_cost`) |
| An IK solution on the shortest chain is not found | both | 64 IK seeds and seeds from the previous waypoint (via-point); keep the solutions on the shortest chains only, so more can be searched (`max_layer_nodes`, multi-waypoint) |
| From a straight seed the optimizer cuts the corner at a waypoint and the shortest chain fails | via-point | every chain also gets a bent seed (`seed_bulge`) |
| An IK solution lies inside the range where the optimizer's joint limit cost is active, so the trajectory cannot reach it | via-point | drop such solutions when a waypoint has others (`joint_limit_margin`) |

## Benchmark

`benchmark/waypoint_optimality_benchmark.py` plans paths with both planners and compares them
with a reference: the shortest IK chain through a dense set of IK solutions per waypoint (40 IK
calls with 32 seeds each, the planners' own configurations, and all full-turn variants). The
paths are made to punish a planner that looks one waypoint ahead:

| Family | Path |
| --- | --- |
| `wrap_tool` | The tool roll joint keeps turning one way, in 3 to 5 steps of 20° to 170°, until a waypoint lies beyond its limit. The first path is the 0° → 90° → 270° → 360° case. |
| `wrap_pan` | The same for the base joint. |
| `double_wrap` | Both joints, with different step patterns. |
| `random_reach` | Four waypoints at the tool poses of widely spread configurations, from a start near a joint limit. |
| `pillar` | Three sweeps of the base joint from one side of a pillar to the other; the shortest chain goes through the pillar. |

`greedy` is the chain of a planner that moves to the nearest IK solution of the next waypoint
every time, from the same dense IK sets as the reference. It shows how much looking ahead is
worth on a path.

```bash
python benchmark/waypoint_optimality_benchmark.py --robot ur10e.yml --num_problems 12 --multi_problems 12
```

## Results

Measured on a Tesla T4 with 12 paths per family, 60 per robot. *Chain* is the joint-space
length of the planner's IK chain over the reference; 1.0 is the shortest possible. For the
multi-waypoint planner it includes the way from the start state to its first configuration.
*Before* is each planner with the settings it had before this study.

**UR10e** (6 joints, five of them with ±354° of range):

| | Success | Chain, mean | Chain, worst | Paths over 1.05 | Planning time |
| --- | --- | --- | --- | --- | --- |
| Nearest solution (`greedy`) | | 1.034 | 1.340 | 12 of 60 | |
| Via-point, before | 59 of 60 | 1.565 | 3.574 | 49 of 59 | 0.19 s |
| Via-point | 60 of 60 | 1.007 | 1.081 | 4 of 60 | 0.19 s |
| Multi-waypoint, before | 60 of 60 | 1.657 | 5.595 | 43 of 60 | 19.7 s |
| Multi-waypoint | 60 of 60 | 1.066 | 2.569 | 12 of 60 | 6.5 s |

**Franka** (7 joints, none with more than one turn of range):

| | Success | Chain, mean | Chain, worst | Paths over 1.05 | Planning time |
| --- | --- | --- | --- | --- | --- |
| Nearest solution (`greedy`) | | 1.072 | 1.770 | 20 of 60 | |
| Via-point, before | 60 of 60 | 1.035 | 1.359 | 15 of 60 | 0.21 s |
| Via-point | 60 of 60 | 1.010 | 1.131 | 4 of 60 | 0.21 s |
| Multi-waypoint, before | 60 of 60 | 1.404 | 2.313 | 55 of 60 | 12.8 s |
| Multi-waypoint | 60 of 60 | 1.057 | 1.335 | 26 of 60 | 7.2 s |

Chain length over the reference and motion time per family, UR10e:

| Family | Greedy | Via-point, before | Via-point | Multi-waypoint, before | Multi-waypoint |
| --- | --- | --- | --- | --- | --- |
| `wrap_tool` | 1.089 | 1.342, 4.05 s | 1.005, 3.40 s | 1.484, 4.47 s | 1.080, 3.68 s |
| `wrap_pan` | 1.037 | 1.809, 4.54 s | 1.002, 3.16 s | 2.289, 4.28 s | 1.041, 3.38 s |
| `double_wrap` | 1.023 | 1.352, 4.62 s | 1.012, 3.51 s | 1.355, 4.42 s | 1.000, 3.42 s |
| `random_reach` | 1.023 | 1.644, 5.95 s | 1.018, 3.90 s | 1.755, 5.72 s | 1.078, 4.16 s |
| `pillar` | 1.000 | 1.659, 4.31 s | 0.997, 3.03 s | 1.403, 2.83 s | 1.129, 2.77 s |

The shorter chains are also faster: the via-point planner's motions take 16 % to 35 % less time
than before on the UR10e. A value just below 1.0 means the trajectory passes a waypoint within
its tolerance, slightly inside the IK solution.

Before the changes both planners were often worse than `greedy`, although both search all
waypoints together. The search was not the problem; the IK solutions it had to choose from
were.

On the same 180 random Franka paths as `benchmark/via_point_benchmark.py` used before, the
via-point planner still succeeds on all, in 148 ms to 212 ms.

## What the numbers do not show

**The length measure.** Euclidean joint-space distance counts several joints moving a little
as shorter than one joint moving a lot. On the 0° → 90° → 270° → 360° path the shortest chain
by this measure changes the arm configuration at the second waypoint (5.85 rad) instead of
turning the wrist back and forth in place (6.28 rad), although the joints turn 35 % more in
total. Both avoid the 540° of the nearest-solution choice. `chain_metric` selects another
measure; see the next section.

**Obstacles between waypoints.** The chain search knows about collisions at the waypoints
only. In the `pillar` family the via-point planner follows the shortest chain on all paths,
but its trajectory around the pillar is 48 % longer than the reference chain on the UR10e,
against 35 % for the multi-waypoint planner, which optimizes every edge separately and so also learns which
chains are cheap to drive. When no way around is found along the shortest chains, the
via-point planner's retries take longer chains.

**Redundant arms.** A 7-joint arm has a continuum of IK solutions per waypoint, and the chain
can only be as short as the solutions that were sampled. With 32 IK seeds the via-point
planner's mean chain on the Franka is 1.058; with 64 it is 1.010.

**Motion time.** The multi-waypoint planner chooses its chain by the summed duration of the
segment trajectories, not by length, so a longer chain can be the faster one for it. It stops
at every waypoint; the via-point planner does not, and is faster on six of the ten families
and slower on the others, among them both `pillar` families.

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

Each variant measured in all four metrics, as the mean chain length over the reference that is
shortest in that metric, on the 60 paths per robot of the benchmark above. Bold is the
variant's own metric.

**UR10e:**

| Variant | `euclidean` | `total_rotation` | `time` | `max_norm` | Motion time |
| --- | --- | --- | --- | --- | --- |
| Via-point, `euclidean` | **1.007** | 1.161 | 1.020 | 1.012 | 3.40 s |
| Via-point, `total_rotation` | 1.051 | **1.015** | 1.139 | 1.149 | 3.76 s |
| Via-point, `time` | 1.147 | 1.464 | **1.010** | 1.087 | 3.35 s |
| Via-point, `max_norm` | 1.019 | 1.193 | 1.024 | **1.007** | 3.38 s |
| Nearest solution, own metric | 1.035 | 1.047 | 1.043 | 1.040 | |

**Franka:**

| Variant | `euclidean` | `total_rotation` | `time` | `max_norm` | Motion time |
| --- | --- | --- | --- | --- | --- |
| Via-point, `euclidean` | **1.011** | 1.259 | 1.176 | 1.151 | 2.68 s |
| Via-point, `total_rotation` | 1.151 | **1.090** | 1.690 | 1.627 | 3.39 s |
| Via-point, `time` | 1.044 | 1.372 | **1.054** | 1.074 | 2.60 s |
| Via-point, `max_norm` | 1.040 | 1.363 | 1.079 | **1.053** | 2.58 s |
| Nearest solution, own metric | 1.071 | 1.060 | 1.074 | 1.091 | |

The last row lists, in each column, the nearest-solution chain chosen in that column's metric.

A chain that is short in one metric can be long in another. `total_rotation` keeps the arm
configuration and turns single joints, which gives the slowest motions. `time` and `max_norm`
spread the motion over the joints and give the fastest ones: 3 % to 4 % faster than
`euclidean` on the Franka, 1 % on the UR10e, where `time` takes chains 46 % longer in total
rotation. `euclidean` stays the default: it is never far from the fastest motion, and its
chains are the ones the trajectory optimizer follows best.

The trajectory optimizer sees the waypoints, not the chain, and its smoothness cost favours
`euclidean`-like paths. For the other metrics the trajectory can therefore leave the chosen
chain: on the Franka the via-point planner's `total_rotation` result is longer in its own metric
than the nearest-solution chain. Picking, among the optimized trajectories, the one whose chain
is shortest in the metric did not help: on the UR10e it made the chains of all four variants
longer and their motions 4 % to 8 % slower.
