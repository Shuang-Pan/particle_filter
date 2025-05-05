# Particle Filter for Surgical Tool Pose Tracking
## Data
1. `estimated_transforms_98.json`: transforms from Sleap model and SolvePnP algorithm. 
2. `true_transforms_98.json`: transforms from simulation environment.
## Script
1. Run `create_error_transform.py` to add errors (t, R, or both) to simulation data (`true_transforms_98.json`) and get the errored data file.
2. Run `particle_filter.py` to get filtered data file using selected errored transforms and `estimated_transforms_98.json`.

