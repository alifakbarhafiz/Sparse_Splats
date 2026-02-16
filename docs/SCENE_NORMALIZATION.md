# Scene normalization and sparse-view metrics

## Is the “normalization from train only” assumption sound?

**Yes.** In the Inria Gaussian Splatting Blender loader:

- `readNerfSyntheticInfo()` in `scene/dataset_readers.py` builds `train_cam_infos` and `test_cam_infos` from your subset’s `transforms_train.json` and `transforms_test.json`.
- It then does:
  ```python
  nerf_normalization = getNerfppNorm(train_cam_infos)  # train only
  ```
- `getNerfppNorm()` uses **only** the training camera centers to compute:
  - `center` = mean of camera positions
  - `diagonal` = max distance of any train camera from that center
  - `radius` = diagonal × 1.1
  - `translate` = -center
- That `nerf_normalization` dict is stored in `SceneInfo` and used as `cameras_extent` in `Scene`; all cameras and the scene are normalized with this same transform.

So **normalization is computed from the training cameras only**, and with our subsets that means:

- 1-view run → normalization from 1 camera.
- 3-view run → normalization from 3 cameras.
- 100-view run → normalization from 100 cameras.

So each run uses a **different** coordinate frame (different center and radius). That is exactly the “real suspect” you identified.

---

## What goes wrong?

1. **Different scale per run**  
   With 1 or 3 views, the bounding sphere (center + radius) is estimated from very few poses. With 100 views it’s a different sphere. So the same physical test view ends up in different normalized coordinates in different experiments → **metrics are not comparable across view counts**.

2. **Bad scale when train is tiny**  
   With 1–3 views, the estimated radius can be wrong (too small, wrong center). Test cameras (full, diverse set) then sit in odd or extreme positions in normalized space. The model was optimized in that distorted frame, so **even within a single run**, test quality can collapse (PSNR ~4–6, flat SSIM/LPIPS).

3. **Why even 100 views looks bad**  
   If 100-view PSNR is still ~4.3, possible contributors are:
   - The above (we’re still comparing across runs with different normalizations).
   - Or other factors (e.g. iteration count, optimizer, or that the full test set is hard).  
   Fixing normalization first is the right place to start.

---

## Other possible issues (besides normalization)

- **Iteration count**: 15k might be too little for very sparse (1–3) or even 100-view; try longer runs or sweep iterations.
- **Optimizer**: Default Adam vs sparse Adam can change convergence.
- **Full test set**: Using the full test set is correct for comparability; the main remaining structural issue is **consistent normalization**.

---

## Fix: shared normalization from the full train set

To make metrics comparable and to avoid distorted scale in sparse subsets:

1. **Compute normalization once** from the **full** training set (e.g. all 100 Blender train views), not from the subset.
2. **Use that same normalization for every subset** (1, 3, 5, 10, 20, 50, 100 views).

Then:

- All experiments live in the **same** normalized coordinate frame.
- PSNR/SSIM/LPIPS across view counts are comparable.
- Test views are no longer misaligned or scaled incorrectly just because the subset had few train views.

### What we do in this repo

- When creating a subset with `full_test_set=True`, the subset script computes **fixed** NeRF-style normalization from the **raw** `transforms_train.json` (all frames) and writes `nerf_normalization.json` into the subset directory.
- That file has the same format the Inria loader expects: `{"translate": [x,y,z], "radius": r}`.

### What you must change in the external gaussian-splatting repo

The Inria code **does not** read `nerf_normalization.json` by default. You need a one-time patch so that, when this file exists in the dataset path, the loader uses it instead of `getNerfppNorm(train_cam_infos)`.

**File to patch:** `scene/dataset_readers.py` (inside `external/gaussian-splatting` or your Colab clone).

**In `readNerfSyntheticInfo()`, replace:**

```python
    nerf_normalization = getNerfppNorm(train_cam_infos)
```

**with:**

```python
    # Use fixed normalization from file if present (for comparable sparse-view metrics)
    norm_path = os.path.join(path, "nerf_normalization.json")
    if os.path.isfile(norm_path):
        with open(norm_path) as f:
            nerf_normalization = json.load(f)
        nerf_normalization["translate"] = np.array(nerf_normalization["translate"], dtype=np.float64)
        nerf_normalization["radius"] = float(nerf_normalization["radius"])
    else:
        nerf_normalization = getNerfppNorm(train_cam_infos)
```

Ensure `import json` and `import numpy as np` are at the top of that file if not already.

After patching, re-run subset creation (so each subset gets `nerf_normalization.json`) and then re-run training and evaluation. Metrics across 1 / 3 / 5 / 10 / 20 / 50 / 100 views will then be in a single, consistent coordinate frame and the “scene normalization” effect should be removed.
