# NVIDIA Carter source

The `carter/` directory is the official Carter URDF/OBJ teaching asset linked by
NVIDIA's *Importing URDF Assets* lesson:

- Documentation: https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-sim/latest/ingesting-robot-assets-and-simulating-your-robot-in-isaac-sim/01-importing-urdf-assets.html
- Asset URL: https://learn.learn.nvidia.com/asset-v1%3ADLI%2BS-OV-29%2BV1%2Btype%40asset%2Bblock%40carter.zip

`build_scene.py` converts the URDF hierarchy to MJCF while retaining the
official visual meshes, chassis/wheel masses and inertias, wheel radius, wheel
spacing, rear pivot and caster dimensions. The downloaded archive does not
contain a separate license file; redistribution and use remain subject to the
terms attached to NVIDIA's source material.
