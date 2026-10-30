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

To bound the work per triangle, a triangle is split into at most 16 steps per edge. A large
triangle on a fine grid, such as a 25 cm triangle on a 1 cm grid, is therefore sampled more
coarsely than half a voxel. For such a triangle the check compares the samples with a larger
distance than the margin, enough to catch everything that half-voxel sampling would catch. It
may then report a collision up to a few millimeters beyond the margin, but it does not miss
more. Space outside the voxel grid reads as free, so obstacles beyond the grid are not seen.

One case is missed for every obstacle type: an obstacle small enough to lie entirely inside a
link, crossing none of its triangles. Detecting it would need a point-in-robot test, which is
expensive because the robot geometry moves with every configuration.

### Inside and outside of mesh obstacles

The containment pass asks whether a link lies inside a mesh obstacle. The answer comes from
the sign of the obstacle's distance field, the same one the sphere collision cost uses, so the
spheres and the meshes always agree on what is inside. The sign is derived from the orientation
of the faces: from the query point, rays are cast along +x, +y and +z, and the point is inside
when at least two of them first hit the back of a face.

This is reliable for closed meshes with outward-facing normals, and a single flipped face does
not change it. Open meshes have no well-defined inside, and the result depends on their shape
and orientation:

- A container without wall thickness, such as a bin modeled as five single-sided faces, reads
  as solid: its whole cavity is inside, so a robot reaching into it collides.
- Behind a single-sided sheet, the result depends on the direction of its normal. A sheet
  facing one of the axes never reads as inside, while a sheet facing, say, (1, 1, 1) has a
  region behind it, as deep as half the sheet's bounding box, that reads as inside.

Both the sphere cost and the mesh check report these regions as collisions. Give obstacle
meshes thickness and close them, or load scanned geometry as a voxel distance field instead
of a mesh. Skipping the containment pass for open meshes would not help, since the sphere
cost still sees the same inside regions.

## Where the check runs

| Component | Mesh check |
| --- | --- |
| IK metrics | yes |
| Trajectory optimization metrics, at the optimized steps | yes |
| Densely interpolated trajectory, at every interpolation step | yes |
| MPC metrics | yes |
| Graph planner seeding | no, spheres only |
| Self-collision | yes, when created with `mesh_self_collision_check=True` |
| Robot collision checker (`validate`) | yes, when created with `use_mesh_constraint=True` (scene) or `use_mesh_self_collision_constraint=True` (self) |

The optimized trajectory is already dense: in measured Franka plans its steps were about 14 ms
apart, and the hand moved at most 1 to 2 cm between two of them. The interpolated trajectory,
sampled every `interpolation_dt` (25 ms by default), is the one sent to the robot, so it is
checked as well, at a cost of 1 to 6 % per `plan_pose` call in the scenes of the cost section
below. Either way the check samples the motion: an obstacle thinner than the distance a link
moves between two samples can be missed. A continuous (swept) check would close that gap.

(mesh_collision_cost)=
## Cost

The numbers below were measured on one NVIDIA Tesla T4 with Franka and CUDA graphs on, with
`benchmark/mesh_collision_benchmark.py`. The scene is a table and a box, represented as
cuboids, as a 24-triangle mesh, as a 192k-triangle mesh, or as an ESDF with 2 cm voxels. The
base link is ignored by the mesh check, since it sits on the table.

Solve time without and with the mesh check:

| Solver | Cuboid scene | Mesh scene (24 triangles) | Voxel scene |
| --- | --- | --- | --- |
| IK, 1 problem | 20.9 → 21.0 ms | 22.8 → 23.1 ms | 21.3 → 21.1 ms |
| IK, 100 problems | 146 → 149 ms | 265 → 273 ms | 161 → 174 ms |
| `plan_pose` | 108 → 104 ms | 180 → 181 ms | 108 → 116 ms |
| MPC step | 28.7 → 28.6 ms | 46.4 → 46.8 ms | 29.6 → 29.3 ms |

Differences of a few percent are within run-to-run noise. Time of one mesh check query with the
default budget of 256 triangles per link (1944 robot triangles), for configurations that are
free of sphere collisions:

| Configurations | Cuboid | Mesh, 24 triangles | Mesh, 192k triangles | Voxel |
| --- | --- | --- | --- | --- |
| 32 | 0.55 ms | 0.64 ms | 0.75 ms | 0.74 ms |
| 512 | 0.60 ms | 0.75 ms | 1.5 ms | 1.3 ms |
| 4096 | 0.75 ms | 1.8 ms | 2.7 ms | 5.8 ms |

- Before testing triangles, the check tests each link's bounding sphere against the obstacles
  with one distance query and skips the links that are clearly free. The cost therefore grows
  with the number of configurations times the robot triangles near obstacles. A single IK
  problem, a plan or an MPC step shows no measurable overhead, and IK with 100 problems costs
  2 to 8 % more. For 32 configurations the extra query adds about 0.1 to 0.2 ms.
- A budget of 64 triangles per link (576 robot triangles) makes a 4096-configuration query 1.1
  to 2.3 times faster than the default. Full fidelity (2164 robot triangles) costs less than
  10 % more than the default.
- A positive `mesh_collision_margin` costs little against cuboids, voxels and coarse meshes.
  Against the 192k-triangle mesh it makes the query about 2.7 times slower (2.7 → 7.3 ms for
  4096 sphere-free configurations), because more triangle pairs pass the broadphase and need
  the exact distance test.

The mesh self-collision check (`mesh_self_collision_check`) tests 35 Franka link pairs. With
the default budget, one query takes 0.5 ms for 32 configurations, 1.0 ms for 512 and 4.7 ms
for 4096 configurations free of sphere collisions; a 1 cm `mesh_self_collision_margin` makes it
2 to 3 times slower. Pairs whose link bounding spheres are apart are skipped first. In the
cuboid scene, turning it on made IK with 100 problems about 9 % slower and `plan_pose` about
1 % slower, with the same success rate (measured in a separate run from the table above).
About 10 % of the configurations free of sphere collisions that the benchmark sampled had
colliding link meshes, nearly all of them `panda_link5` against the hand, `panda_link7` or a
finger: pairs the sphere check ignores.
