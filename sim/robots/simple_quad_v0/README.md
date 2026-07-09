# Simple Quadruped V0

Use this robot as the first fast-learning prototype for the sim pipeline.

Drop the complete Onshape URDF export into:

```text
sim/robots/simple_quad_v0/onshape_export/
```

Expected contents:

```text
robot.urdf
meshes/
```

Design guidance for the prototype:

- Use a simple body and four simple legs.
- Prefer 2 or 3 actuated joints per leg.
- Avoid closed-loop linkages for the first pass.
- Use real-ish dimensions and masses.
- Give every joint and link a clear name.
- Make sure foot collision geometry touches the ground.
- Export inertias from actual materials/densities when possible.

The screenshot can be useful visual context, but the URDF package is the source of truth.
