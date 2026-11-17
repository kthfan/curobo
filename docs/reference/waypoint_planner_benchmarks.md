(waypoint_planner_benchmarks)=

# Waypoint Planner Benchmarks

Measured results of `benchmark/waypoint_optimality_benchmark.py` for the
{ref}`via-point planner <via_point_planner_note>` and the {ref}`multi-waypoint planner
<multi_waypoint_planner_note>`. The note on {ref}`shortest paths through waypoints
<waypoint_path_optimality_note>` explains what the numbers measure.

## Conditions

| Item | Value |
| --- | --- |
| Date | November 2026 |
| GPU | Tesla T4 |
| CUDA graphs | enabled |
| Robots | `ur10e.yml` (6 joints, five with ±354° of range), `franka.yml` (7 joints, none with more than one turn of range) |
| Paths | 12 per family, 5 families, 60 per robot; 30 per robot for the multi-waypoint planner in the chain metric tables |
| Planner settings | the defaults of `ViaPointMotionPlannerCfg.create` and `MultiWaypointMotionPlannerCfg.create`, except where a row names a setting |

These are recorded measurements, not results of the current checkout.

## Path families

| Family | Path |
| --- | --- |
| `wrap_tool` | The tool roll joint keeps turning one way, in 3 to 5 steps of 20° to 170°, until a waypoint lies beyond its limit. The first path is the 0° → 90° → 270° → 360° case. |
| `wrap_pan` | The same for the base joint. |
| `double_wrap` | Both joints, with different step patterns. |
| `random_reach` | Four waypoints at the tool poses of widely spread configurations, from a start near a joint limit. |
| `pillar` | Three sweeps of the base joint from one side of a pillar to the other; the shortest chain goes through the pillar. The pillar height is lowered until the path is reachable. |

## Quantities

| Quantity | Definition |
| --- | --- |
| Reference | Shortest IK chain through a dense IK set per waypoint: `--ik_rounds` IK calls with 32 seeds each (40 for the Euclidean tables, 25 for the chain metric tables), the planners' own IK solutions, and all full-turn variants within limits narrowed by 1 % of the range. |
| Chain | Length of the planner's IK chain (its configurations at the waypoints) divided by the reference. 1.0 is the shortest possible; values just below 1.0 mean a waypoint is passed within its tolerance. For the multi-waypoint planner it includes the way from the start state to its first configuration. |
| `greedy` | Chain that moves to the nearest IK solution of the next waypoint every time, from the same IK sets as the reference. |
| Paths over 1.05 | Paths whose chain is more than 5 % longer than the reference. |
| Planning time | Wall time of one planning call after warmup, including retries. |
| Motion time | Duration of the returned trajectory. |
| Previous defaults | The planner before `expand_revolutions`, `chain_length_slack`, `seed_bulge`, `joint_limit_margin`, `seed_ik_from_previous`, `start_state_cost` and `max_layer_nodes` were added. |

## Chain length, Euclidean metric

**UR10e:**

| Planner | Success | Chain, mean | Chain, worst | Paths over 1.05 | Planning time |
| --- | --- | --- | --- | --- | --- |
| `greedy` | | 1.034 | 1.340 | 12 of 60 | |
| Via-point, previous defaults | 59 of 60 | 1.565 | 3.574 | 49 of 59 | 0.19 s |
| Via-point | 60 of 60 | 1.007 | 1.081 | 4 of 60 | 0.19 s |
| Multi-waypoint, previous defaults | 60 of 60 | 1.657 | 5.595 | 43 of 60 | 19.7 s |
| Multi-waypoint | 60 of 60 | 1.066 | 2.569 | 12 of 60 | 6.5 s |

**Franka:**

| Planner | Success | Chain, mean | Chain, worst | Paths over 1.05 | Planning time |
| --- | --- | --- | --- | --- | --- |
| `greedy` | | 1.072 | 1.770 | 20 of 60 | |
| Via-point, previous defaults | 60 of 60 | 1.035 | 1.359 | 15 of 60 | 0.21 s |
| Via-point | 60 of 60 | 1.010 | 1.131 | 4 of 60 | 0.21 s |
| Multi-waypoint, previous defaults | 60 of 60 | 1.404 | 2.313 | 55 of 60 | 12.8 s |
| Multi-waypoint | 60 of 60 | 1.057 | 1.335 | 26 of 60 | 7.2 s |

Via-point planner on the Franka with 32 IK seeds instead of 64: mean chain 1.058.

**Per family, UR10e** (chain, motion time):

| Family | `greedy` | Via-point, previous defaults | Via-point | Multi-waypoint, previous defaults | Multi-waypoint |
| --- | --- | --- | --- | --- | --- |
| `wrap_tool` | 1.089 | 1.342, 4.05 s | 1.005, 3.40 s | 1.484, 4.47 s | 1.080, 3.68 s |
| `wrap_pan` | 1.037 | 1.809, 4.54 s | 1.002, 3.16 s | 2.289, 4.28 s | 1.041, 3.38 s |
| `double_wrap` | 1.023 | 1.352, 4.62 s | 1.012, 3.51 s | 1.355, 4.42 s | 1.000, 3.42 s |
| `random_reach` | 1.023 | 1.644, 5.95 s | 1.018, 3.90 s | 1.755, 5.72 s | 1.078, 4.16 s |
| `pillar` | 1.000 | 1.659, 4.31 s | 0.997, 3.03 s | 1.403, 2.83 s | 1.129, 2.77 s |

Joint-space arc length of the trajectory over the reference chain in the `pillar` family,
UR10e: via-point 1.48, multi-waypoint 1.35.

## Chain metrics

Via-point planner with each `chain_metric`, measured in all four metrics. Each column is the
mean chain length over the reference that is shortest in that column's metric. Bold is the
variant's own metric. The `greedy` row uses, in each column, the nearest-solution chain
chosen in that column's metric.

**UR10e:**

| Variant | `euclidean` | `total_rotation` | `time` | `max_norm` | Motion time | Planning time |
| --- | --- | --- | --- | --- | --- | --- |
| `euclidean` | **1.007** | 1.161 | 1.020 | 1.012 | 3.40 s | 0.16 s |
| `total_rotation` | 1.051 | **1.015** | 1.139 | 1.149 | 3.76 s | 0.16 s |
| `time` | 1.147 | 1.464 | **1.010** | 1.087 | 3.35 s | 0.16 s |
| `max_norm` | 1.019 | 1.193 | 1.024 | **1.007** | 3.38 s | 0.16 s |
| `greedy`, own metric | 1.035 | 1.047 | 1.043 | 1.040 | | |

**Franka:**

| Variant | `euclidean` | `total_rotation` | `time` | `max_norm` | Motion time | Planning time |
| --- | --- | --- | --- | --- | --- | --- |
| `euclidean` | **1.011** | 1.259 | 1.176 | 1.151 | 2.68 s | 0.19 s |
| `total_rotation` | 1.151 | **1.090** | 1.690 | 1.627 | 3.39 s | 0.19 s |
| `time` | 1.044 | 1.372 | **1.054** | 1.074 | 2.60 s | 0.19 s |
| `max_norm` | 1.040 | 1.363 | 1.079 | **1.053** | 2.58 s | 0.19 s |
| `greedy`, own metric | 1.071 | 1.060 | 1.074 | 1.091 | | |

Multi-waypoint planner with each `chain_metric` (30 paths per robot):

| Robot | Motion time, mean | Planning time, mean |
| --- | --- | --- |
| UR10e | 3.68 s to 3.72 s | 5.9 s to 6.0 s |
| Franka | 2.65 s to 2.84 s | 6.5 s to 6.6 s |

Ranges are over the four metrics.

Returning, among the optimized via-point trajectories, the fastest one whose chain is at most
`chain_length_slack` longer than the shortest one in the metric (not part of the planner) gave on
the UR10e:

| Variant | Own metric | Motion time |
| --- | --- | --- |
| `euclidean` | 1.015 | 3.62 s |
| `total_rotation` | 1.021 | 4.06 s |
| `time` | 1.016 | 3.51 s |
| `max_norm` | 1.011 | 3.53 s |

## Random paths

`benchmark/via_point_benchmark.py --skip_multi` on the Franka, seeds 0 to 2, 2, 4 and 6
waypoints, 20 paths each (180 paths):

| Waypoints | Success | Planning time, mean | Motion time, mean |
| --- | --- | --- | --- |
| 2 | 60 of 60 | 156 ms to 158 ms | 0.95 s to 1.07 s |
| 4 | 60 of 60 | 189 ms | 1.69 s to 1.90 s |
| 6 | 60 of 60 | 221 ms to 241 ms | 2.63 s to 2.81 s |

Ranges are over the three seeds.

## Run the benchmark

```bash
python benchmark/waypoint_optimality_benchmark.py --robot ur10e.yml --num_problems 12 --multi_problems 12
```

| Argument | Meaning |
| --- | --- |
| `--robot` | Robot configuration file. |
| `--num_problems` | Paths per family for the via-point planner and the references. |
| `--multi_problems` | Paths per family also planned with the multi-waypoint planner; `0` skips it. |
| `--ik_rounds` | IK calls per waypoint for the reference IK sets. |
| `--families` | Families to run. |
| `--out` | JSON file with every path, chain and length. |
