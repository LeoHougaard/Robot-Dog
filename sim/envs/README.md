# Simulation Environments

`simple_quad_stand.py` contains the first Gymnasium-style MuJoCo environment:

- `SimpleQuadStandEnv`
- `SimpleQuadWalkEnv`

It loads the generated MJCF at:

```text
sim/robots/simple_quad_v0/mjcf/simple_quad_v0.xml
```

The model is generated from the Onshape URDF by:

```powershell
python sim/tools/build_simple_quad_mjcf.py
```
