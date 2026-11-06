(mesh_collision_guide)=

# Check Collisions with the Robot's Exact Meshes

**Goal:** reject IK solutions and trajectories in which the robot's real geometry touches an
obstacle, even when its collision spheres do not. Typical cases are thin fingers next to a
shelf wall, a tool reaching into a narrow opening, or links that the spheres cover only
approximately.

The optimizers keep using the collision spheres to move away from obstacles. The mesh check is
an extra pass/fail test on the result: a solution whose collision meshes touch the scene is
reported as failed, and the solver tries its other seeds or attempts. See
{ref}`mesh_collision_note` for how the check works and what it can miss.

## Enable the check

Pass `mesh_collision_check=True` when you create the solver:

```python
from curobo import MotionPlanner, MotionPlannerCfg

config = MotionPlannerCfg.create(
    robot="franka.yml",
    scene_model="collision_table.yml",
    mesh_collision_check=True,
)
planner = MotionPlanner(config)
result = planner.plan_pose(goal_tool_poses, current_state)
```

The same option exists on the `create` method of
{py:class}`~curobo.inverse_kinematics.InverseKinematicsCfg`,
{py:class}`~curobo.trajectory_optimizer.TrajectoryOptimizerCfg`, and
{py:class}`~curobo.model_predictive_control.ModelPredictiveControlCfg`.
{py:meth}`MultiWaypointMotionPlannerCfg.create
<curobo.multi_waypoint_planner.MultiWaypointMotionPlannerCfg.create>` forwards it to the motion
planner.

With the option on, cuRobo:

- loads the URDF `<collision>` meshes of the robot's collision links (a link without collision
  geometry uses its `<visual>` mesh), and
- adds the mesh collision constraint to the metrics rollout that decides whether a solution
  succeeded.

The check needs obstacles to test against: pass a `scene_model`, or a `collision_cache` when
the scene is loaded later with `update_world`. Without a scene the option has no effect.

```{note}
If you pass the robot as a `RobotCfg` object, create it with
`RobotCfg.create(..., load_collision_meshes=True)`. A `RobotCfg` without collision meshes is
rejected when the mesh check is on. Robot YAML paths and dicts are handled automatically.
```

## Keep a clearance

`mesh_collision_margin` (meters, default 0) is the distance the collision meshes must keep from
obstacles:

```python
MotionPlannerCfg.create(
    robot="franka.yml",
    scene_model="collision_table.yml",
    mesh_collision_check=True,
    mesh_collision_margin=0.005,
)
```

When a link mesh is decimated (see the next section), its surface moves by the deviation that
cuRobo reports at load time. A margin of that size keeps the check from missing contacts there.

## Limit the triangle count

The check costs time in proportion to the number of robot triangles. Each link is packed with
at most 256 triangles by default; a larger mesh is decimated to fit. Decimation keeps the open
borders that robot collision meshes often have and picks, among several settings, the result
closest to the original surface: Franka's links stay within about 1 mm at the default budget.
cuRobo logs how far each decimated surface moved, and warns when it moved by more than 2 mm
or when a mesh cannot be reduced to its budget. Set a different budget in the `kinematics`
section of the robot YAML, either for every link or per link:

```yaml
robot_cfg:
  kinematics:
    max_triangles_per_link: 128
    # or per link:
    # max_triangles_per_link: {panda_hand: 512, panda_leftfinger: 64}
```

Increase the budget for links whose shape matters, such as fingers, and for links that the
warning reports. Lower it for large, simple links. {ref}`mesh_collision_cost` lists measured costs for different budgets and scenes.

## Ignore links that touch the scene by design

A link in permanent contact with the scene, such as a base bolted to its mounting plate, always
fails an exact test, even for a flush contact with zero penetration. Exclude it in the robot
YAML:

```yaml
robot_cfg:
  kinematics:
    mesh_collision_ignored_link_names: [panda_link0]
```

Ignored links are skipped only by the mesh scene check. The collision spheres still apply to
them, and the mesh self-collision check (see below) still tests them against the other links,
so a base link bolted to the table still cannot be hit by the hand. To leave a link pair out of
the self-collision check, list it under `mesh_self_collision_ignore`.

## Let links touch the scene for a while

`MotionPlanner.disable_link_collision(links)` turns off scene collision for links on both
their spheres and their meshes, and `enable_link_collision(links)` turns it back on.
`plan_grasp` does this for the robot's `grasp_contact_link_names` while it plans to the grasp,
moves in and lifts the object, so the fingers may touch it with the mesh check on. To toggle
only the mesh check, call `disable_link_mesh` and `enable_link_mesh` on the kinematics config.

## Add geometry at runtime

To check geometry that is known only at runtime, such as a grasped object, reserve triangle
slots for a link in the robot YAML:

```yaml
robot_cfg:
  kinematics:
    extra_collision_meshes: {attached_object: 256}
```

`franka.yml` already reserves 256 slots for `attached_object`. Reserved slots cost nothing
without the mesh check; with it, empty slots are skipped right away.

The attachment manager fills these slots: attaching an object writes its mesh along with its
collision spheres, and detaching it clears both. The link meshes are shared by all
environments, so with several environments the object must have the same offset to the link in
each of them; otherwise the manager warns and checks the object with its spheres only.

To manage the geometry yourself, write it in the link's frame, as a `[n, 3, 3]` triangle tensor
or a mesh, and clear it when the object is released:

```python
kinematics_config = planner.kinematics.config.kinematics_config
kinematics_config.update_link_mesh("attached_object", object_triangles)
...
kinematics_config.clear_link_mesh("attached_object")
```

`disable_link_mesh` only pauses the geometry, and `enable_link_mesh` (also called by
`enable_link_collision` and `plan_grasp`) brings it back, so use `clear_link_mesh` for a
released object. The update is in place: every solver of the planner sees it at once, and
captured CUDA graphs stay valid. Geometry larger than the reserved slots is decimated to fit.

## Find the link that collides

When a solve fails, `get_mesh_collision_links` tells you whether the meshes are the reason. It
runs the mesh check again on the configurations you pass, such as the result's `js_solution` or
`interpolated_trajectory`, and names a colliding link for each of them:

```python
result = planner.plan_pose(goal_tool_poses, current_state)
links = planner.get_mesh_collision_links(result.interpolated_trajectory)
# links[batch][seed][step] is a link name, or None where the meshes are clear.
for step, link in enumerate(links[0][0]):
    if link is not None:
        print(f"mesh collision: {link} at step {step}")
        break
```

The method exists on the motion planners and on the IK, trajectory optimization and MPC
solvers, and needs `mesh_collision_check=True`. The result has the leading shape of the joint
positions you pass. When several links collide at one configuration, one of them is named. The
check runs separately from the solve, so it adds no cost to planning and leaves captured CUDA
graphs untouched.

## Measure the clearance to the scene

`get_mesh_collision_distance` measures how far the robot's meshes are from the scene, for each
configuration you pass, and names the link nearest to it:

```python
distance, closest = planner.get_mesh_collision_distance(
    result.interpolated_trajectory, max_distance=0.1
)
# distance[batch, seed, step] is in meters; closest[batch][seed][step] is a link name, or
# None where every link is at least max_distance away.
print(f"closest approach: {distance.min().item():.3f} m")
```

The value is signed:

- **Positive:** the distance between the nearest link mesh and the nearest obstacle. It is
  exact for cuboids and meshes, and accurate to about a voxel for voxel grids.
- **Zero:** a link's surface crosses an obstacle's surface, but none of its vertices is inside.
- **Negative:** how deep the link's deepest vertex is inside an obstacle. This can be less
  than the true penetration depth, never more.

Configurations whose meshes are all farther than `max_distance` (default 0.1 m) report
`max_distance`. A larger cap measures farther but is slower, most of all against dense mesh
obstacles; {ref}`mesh_collision_cost` lists measured costs. The value is the geometric distance: the `mesh_collision_margin` is not subtracted, so a
configuration fails the mesh check where the distance is below the margin. Links ignored with
`mesh_collision_ignored_link_names` are not measured. Negative values come from the inside of
the obstacles, so for open mesh obstacles they can be wrong where the free space reads as
inside (see the limitations below).

Like `get_mesh_collision_links`, the method needs `mesh_collision_check=True` and runs
separately from the solve. It has no gradient, so the optimizers do not use it.

## Fix solves that fail only with the mesh check

A goal that the solver reaches without the mesh check can fail with it, mostly in tight
spaces: IK finds no successful seed, or `plan_pose` fails after all its attempts. The mesh
check has no gradient (see {ref}`mesh_collision_note`). The optimizers move every seed with the
sphere costs only, so when the spheres are clear while a mesh collides, nothing pushes that
seed out of the collision; only the other seeds can succeed. To fix it:

1. Find the link with `get_mesh_collision_links` (see the previous section). Many seeds failing
   on the same link point to that link's spheres.
2. Cover that link better with spheres (see {ref}`mesh_collision_cover_spheres`). The
   optimizer then steers the link away itself, but every added sphere makes the sphere
   collision costs slower.
3. Raise `optimizer_collision_activation_distance` (default 0.01 m) on the solver config, for
   example to 0.02 m. The optimizer then keeps the spheres farther from obstacles, which also
   keeps the parts of the meshes outside the spheres clear. The success check is unchanged,
   but narrow passages become harder to pass.
4. Give the solver more chances: raise `num_ik_seeds`, `num_trajopt_seeds` or the
   `max_attempts` of `plan_pose`. This helps when only some seeds end in a mesh-only collision.

When the solves fail because the spheres collide while the meshes are clear, the opposite
case, let the meshes decide success instead (see {ref}`mesh_collision_mesh_only_success`).

Avoid the robot YAML's `collision_sphere_buffer` for this: it enlarges the spheres for the
success check and self-collision too, so it rejects more solutions instead of steering them.

(mesh_collision_mesh_only_success)=
## Let the meshes decide success

Spheres are larger than the meshes they cover, so in tight spaces they also reject solutions
whose meshes are clear. With both mesh checks on, this is the more common failure: in a
cluttered scene, with goals whose meshes come within 3 cm of an obstacle, the spheres alone
rejected 3 % of Franka's and 9 % of UR10e's IK goals, against at most 0.7 % that only the meshes
rejected. Pass `mesh_only_success=True` to let the mesh checks decide success instead of the
sphere checks they replace:

```python
MotionPlannerCfg.create(
    robot="franka.yml",
    scene_model="collision_table.yml",
    mesh_collision_check=True,
    mesh_self_collision_check=True,
    mesh_only_success=True,
)
```

- The scene spheres stop deciding success when `mesh_collision_check` is on, the
  self-collision spheres when `mesh_self_collision_check` is on. With neither, the option
  raises an error. It exists on the same `create` methods as the mesh checks.
- The spheres still steer the optimizers, `sample_configs` still rejects configurations with
  the spheres, and graph planner validation stays sphere-based.
- A solution may have spheres overlapping obstacles, and the margin the spheres kept around the
  meshes is gone. Set `mesh_collision_margin` and `mesh_self_collision_margin` to the clearance
  your application needs.

Success with both mesh checks on, 300 IK goals and 60 plans in a table with 40 random boxes
(one NVIDIA Tesla T4, `benchmark/mesh_failure_benchmark.py`):

| Robot | Mesh margin | IK, spheres decide | IK, meshes decide | `plan_pose`, spheres decide | `plan_pose`, meshes decide |
| --- | --- | --- | --- | --- | --- |
| Franka | 0 | 95.0 % | 98.3 % | 98.3 % | 98.3 % |
| Franka | 5 mm | 90.0 % | 90.0 % | 93.3 % | 95.0 % |
| UR10e | 0 | 89.0 % | 94.7 % | 81.7 % | 91.7 % |
| UR10e | 5 mm | 83.3 % | 86.0 % | 71.7 % | 68.3 % |

A margin rejects the goals that come closer than it, so it lowers success in both modes and
leaves less for the option to recover.

(mesh_collision_cover_spheres)=
## Cover the link meshes with spheres

{py:class}`~curobo.robot_builder.SphereCoverage` measures where a robot's link meshes stick out
of its collision spheres, adds spheres there, and writes the robot YAML. The example script
does all three and writes a report next to the configuration:

```bash
python -m curobo.examples.guides.cover_link_meshes_with_spheres --robot franka.yml
```

The same steps in Python:

```python
from curobo.robot_builder import SphereCoverage

coverage = SphereCoverage("franka.yml")
before = coverage.analyze()
coverage.cover_meshes(tolerance=0.01)
after = coverage.analyze()
print(after.to_markdown(baseline=before))
coverage.save("franka_covered.yml")
```

- **Links:** how far each link mesh sticks out of its own spheres, before and after. New
  spheres keep the existing ones, sit inside the mesh and stick out of it by at most
  `tolerance`. Links that no joint moves, such as the base, are left alone unless you name
  them in `link_names`: their spheres are often kept small on purpose, clear of the surface
  the robot is mounted on.
- **Self-collision pairs:** on sampled joint configurations, how often each pair of link
  meshes collides while its spheres do not (*missed*), and the reverse (*false alarm*). The
  summary gives both for whole configurations: a missed configuration passes the sphere check
  and fails only the mesh check.
- **Sphere count:** every added sphere makes the sphere collision costs a little slower. Raise
  `tolerance` or lower `max_new_spheres_per_link` to add fewer.

`save` keeps everything else in the YAML as loaded, so the result can replace the original.

Pairs listed in `self_collision_ignore` stay invisible to the sphere check, however well the
spheres cover the meshes. The report lists ignored pairs whose meshes collide although their
spheres do not overlap at the default joint configuration;
{py:meth}`~curobo.robot_builder.SphereCoverage.remove_self_collision_ignore`, or the script's
`--check_suggested_pairs`, hands them back to the sphere check. For Franka these are
`panda_link5` with `panda_link7`, `panda_hand` and both fingers. On 2000 random configurations
with a 1 cm tolerance:

| | Missed | False alarm | Spheres |
| --- | --- | --- | --- |
| Shipped `franka.yml` | 8.2 % | 3.7 % | 61 |
| Covered | 8.1 % | 4.0 % | 111 |
| Covered, suggested pairs checked | 0.0 % | 8.5 % | 111 |

Covering alone barely changes Franka's self-collision misses, because they come from the
ignored pairs. It does bring every moving link's mesh within 1 cm of its spheres, which counts
for the scene: the meshes of the shipped spheres stick out by up to 9 cm. The false alarms
include the `self_collision_buffer` margins of the robot YAML.

## Check self-collision with the meshes

The sphere self-collision check skips the link pairs listed in `self_collision_ignore`, such
as Franka's `panda_link5` with `panda_link7` and with `panda_hand`. Their meshes still collide
when the wrist folds onto the forearm: in about 6 and 7 % of random Franka configurations.
{ref}`mesh_collision_cover_spheres` shows how to hand such pairs back to the sphere check.
Pass `mesh_self_collision_check=True` to check the exact link meshes against each other as
well:

```python
MotionPlannerCfg.create(
    robot="franka.yml",
    scene_model="collision_table.yml",
    mesh_self_collision_check=True,
    mesh_self_collision_margin=0.0,
)
```

The option is independent of `mesh_collision_check` and exists on the same `create` methods.
Like the scene check it is a pass/fail test on the result; the sphere self-collision cost still
steers the optimizer, and its check still applies.

The mesh check tests every pair of links with collision meshes, except:

- links that move as one rigid body, joined by fixed or locked joints, such as Franka's hand
  and its locked fingers;
- links that a single joint connects, which usually touch at the joint;
- pairs listed under `mesh_self_collision_ignore` in the robot YAML. This list is separate from
  the sphere `self_collision_ignore`:

```yaml
robot_cfg:
  kinematics:
    mesh_self_collision_ignore: {panda_link0: [panda_link2]}
```

Franka needs no list. If two tested meshes collide at the robot's default joint configuration,
cuRobo warns and names the pair; such pairs usually touch by design and belong in the list.
For a new robot, {py:meth}`RobotBuilder.compute_mesh_collision_matrix
<curobo.robot_builder.RobotBuilder.compute_mesh_collision_matrix>` derives
the list: it adds pairs that collide at the default configuration or in nearly every random
configuration.

To see which pair a failed solution hits, pass its joint states to
`get_mesh_self_collision_pairs`, the self-collision counterpart of `get_mesh_collision_links`:

```python
pairs = planner.get_mesh_self_collision_pairs(result.interpolated_trajectory)
# pairs[batch][seed][step] is a (link_a, link_b) tuple, or None.
```

## Check configurations outside a solver

The robot collision checker accepts the same settings. Its `validate` then includes the mesh
check:

```python
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg

config = RobotCollisionCheckerCfg.load_from_config(
    robot_config="franka.yml",
    scene_model="collision_table.yml",
    use_mesh_constraint=True,
    mesh_constraint_margin=0.0,
    mesh_constraint_ignored_links=["panda_link0"],
    use_mesh_self_collision_constraint=True,
)
checker = RobotCollisionChecker(config)
is_valid = checker.validate(joint_positions)
```

## Limitations

- Self-collision with the meshes is opt-in (`mesh_self_collision_check`). A link fully inside
  another link, crossing none of its triangles, is not detected.
- Trajectory optimization checks the optimized steps and every step of the interpolated
  trajectory. Both sample the motion, so an obstacle thinner than the distance a link moves
  between two samples (about 1 to 2 cm for the hand in typical plans) can be missed.
- The graph planner that seeds trajectory optimization uses the spheres only; the trajectory
  optimization that follows the seed applies the mesh check.
- Voxel (ESDF) obstacles are checked by sampling, so features thinner than the voxel size can
  be missed, and nothing outside the voxel grid is seen. Large robot triangles on fine grids
  are checked slightly conservatively. Cuboid and mesh obstacles are checked exactly.
- An obstacle small enough to sit entirely inside a link, touching none of its triangles, is
  not detected.
- Mesh obstacles should be closed. For open meshes, such as a bin without wall thickness or a
  single-sided sheet, part of the free space can read as inside the obstacle, for the spheres
  and the meshes alike. See {ref}`mesh_collision_note` for which regions are affected.
- The check runs on CUDA devices only.
