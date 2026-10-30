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

Use a small positive margin when the robot's meshes are decimated (see the next section), since
decimation can shrink a link slightly.

## Limit the triangle count

The check costs time in proportion to the number of robot triangles. Each link is packed with
at most 256 triangles by default; a larger mesh is decimated to fit, with a warning. Set a
different budget in the `kinematics` section of the robot YAML, either for every link or per
link:

```yaml
robot_cfg:
  kinematics:
    max_triangles_per_link: 128
    # or per link:
    # max_triangles_per_link: {panda_hand: 512, panda_leftfinger: 64}
```

Increase the budget for links whose shape matters, such as fingers. Lower it for large, simple
links. {ref}`mesh_collision_cost` lists measured costs for different budgets and scenes.

## Ignore links that touch the scene by design

A link in permanent contact with the scene, such as a base bolted to its mounting plate, always
fails an exact test, even for a flush contact with zero penetration. Exclude it in the robot
YAML:

```yaml
robot_cfg:
  kinematics:
    mesh_collision_ignored_link_names: [panda_link0]
```

Ignored links are skipped only by the mesh check; the collision spheres still apply to them.

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
)
checker = RobotCollisionChecker(config)
is_valid = checker.validate(joint_positions)
```

## Limitations

- The check covers robot-scene collisions. Self-collision is still checked with the spheres.
- Trajectory optimization checks the optimized steps and every step of the interpolated
  trajectory. Both sample the motion, so an obstacle thinner than the distance a link moves
  between two samples (about 1 to 2 cm for the hand in typical plans) can be missed.
- The graph planner that seeds trajectory optimization uses the spheres only; the trajectory
  optimization that follows the seed applies the mesh check.
- Voxel (ESDF) obstacles are checked by sampling, so features thinner than the voxel size can
  be missed. Cuboid and mesh obstacles are checked exactly.
- An obstacle small enough to sit entirely inside a link, touching none of its triangles, is
  not detected.
- Mesh obstacles should be closed. For open meshes, such as a bin without wall thickness or a
  single-sided sheet, part of the free space can read as inside the obstacle, for the spheres
  and the meshes alike. See {ref}`mesh_collision_note` for which regions are affected.
- The check runs on CUDA devices only.
