## Module 2: Data Collection
We generate our offline dataset using MuJoCo's raycasting to calculate precise safety labels. 

**Script:** `scripts/01_collect_data.py`
**What it does:** 1. Spawns the MuJoCo ICU environment.
2. Applies random continuous pan/tilt actions to the camera.
3. Captures a 64x64 RGB visual observation.
4. Shoots a ray along the camera's sightline. If it hits the bed (privacy zone), it records the exact distance. If not, it assigns a safe max distance of `8.0`.
5. Compresses the dataset into `data/offline_dataset.npz`.

**To run:**
```bash
python scripts/01_collect_data.py

```

## Module 3: V-OCBF Training
We train a supervised Vision-Operational Space Control Barrier Function (V-OCBF) using a lightweight CNN+MLP architecture.

**Script:** `scripts/02_train_vocbf.py`
**What it does:** 1. Loads the `.pkl` offline dataset.
2. Passes the 64x64 visual observation through a CNN and concatenates it with the 2-DOF joint angles.
3. Uses Mean Squared Error (MSE) to predict the continuous raycast distance to the privacy zone.
4. Freezes the trained weights and saves them to `models/vocbf_weights.pth` for inference.

**To run:**
```bash
python scripts/02_train_vocbf.py