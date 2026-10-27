(mesh_collision_note)=

# Exact Mesh Collision

cuRobo represents the robot with collision spheres. Spheres make collision avoidance fast and
differentiable: the distance from a sphere to an obstacle has a smooth gradient that the
optimizer follows. They are also an approximation. A set of spheres either sticks out of a link
or leaves gaps, so a configuration can be collision-free for the spheres while the real link
touches an obstacle, or the reverse.

The exact mesh collision check closes the first gap. It tests the robot's own collision
triangles against the scene and rejects configurations in which they touch. This page explains
how it fits into the solvers and where its accuracy ends. To turn it on, see
{ref}`mesh_collision_guide`.

## A gate, not a cost

The check answers one question per configuration: does the robot's geometry intersect the
scene? Its value is the constraint weight when it does and 0 otherwise. It has no gradient.

That decides where it is used. The optimizers that shape a trajectory need a gradient, so they
keep using the sphere collision cost. The mesh check is part of the **metrics rollout**, which
evaluates each optimized solution to decide whether it succeeded:

1. The optimizer improves every seed with the sphere-based costs.
2. The metrics rollout evaluates the seeds. A seed whose collision meshes touch the scene fails
   the mesh constraint and is marked infeasible.
3. The solver ranks the feasible seeds. If none remain, the solve fails and higher-level
   planners retry with new seeds.

The spheres therefore steer, and the meshes have the final say. A scene in which the spheres
pass but the meshes collide makes some seeds fail; it does not change how the optimizer moves
them. Margins in the sphere configuration and the `mesh_collision_margin` clearance work
together: spheres that cover the links a little generously lead the optimizer to solutions that
the mesh check accepts.

## Geometry

The robot geometry is loaded from the URDF `<collision>` meshes of the collision links, falling
back to `<visual>` meshes. Each link is stored as a triangle soup in the link frame, with the
URDF origin and scale applied, and packed into one buffer with a fixed number of slots per link
(256 by default). Larger meshes are decimated to fit their slots. Fixed slots keep the buffer
size constant, so geometry can be replaced at runtime without reallocating memory or
recapturing CUDA graphs. This is how runtime geometry such as a grasped object is added.

For each configuration, forward kinematics provides every link's transform. The kernels place
each triangle with its link's transform and test it against the obstacles. Work is spread as
one GPU thread per configuration and robot triangle, so the cost grows with the batch size,
the trajectory length, and the total triangle count. Once a configuration is found in
collision, its remaining threads stop early.

## Accuracy per obstacle type

| Obstacle | Test | Exact |
| --- | --- | --- |
| Cuboid | triangle-box overlap with the separating axis theorem; covers a triangle fully inside the box | yes |
| Mesh | triangle-triangle intersection, plus an optional containment pass that catches a link fully inside a closed mesh | yes |
| Voxel (ESDF) | points sampled on each triangle at most half a voxel apart, tested against the distance field | no, limited by the voxel size |

`mesh_collision_margin` is a distance between the robot triangles and the obstacle surface. For
mesh obstacles it is the exact triangle-to-triangle distance, for voxels it is compared with the
sampled distance field, and for cuboids it enlarges the box by the margin on every side, which
is slightly more conservative near the box corners.

One case is missed for every obstacle type: an obstacle small enough to lie entirely inside a
link, crossing none of its triangles. Detecting it would need a point-in-robot test, which is
expensive because the robot geometry moves with every configuration.

## Where the check runs

| Component | Mesh check |
| --- | --- |
| IK metrics | yes |
| Trajectory optimization metrics, at the optimized steps | yes |
| Densely interpolated trajectory | no |
| MPC metrics | yes |
| Graph planner seeding | no, spheres only |
| Self-collision | no, spheres only |
| Robot collision checker (`validate`) | yes, when created with `use_mesh_constraint=True` |

The interpolated trajectory is skipped to keep the check affordable, since it has many more
steps than the optimized trajectory. A collision that happens only between two optimized steps
is therefore not detected.
