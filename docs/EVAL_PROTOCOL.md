# Evaluation protocol for the final NYU run

The recorded 654-image NYU result uses the relative Depth Anything V2 Small,
Base, and Large checkpoints. Reproduce it with:

```bash
python scripts/02_eval_depth.py --dataset nyu --data-root data/nyu --eigen-crop
```

| Decision | Recorded choice | Relationship to the paper / official implementation |
|---|---|---|
| Checkpoints | Hugging Face ports of the released V2 relative-depth S/B/L weights | Same released weights; inference uses `transformers` rather than the authors' codebase. |
| Input | Hugging Face image processor, short side 518 and dimensions divisible by 14 | Preprocessing may differ from the authors' OpenCV implementation. |
| Output upsampling | Bilinear to original image size, `align_corners=False` (`src/dav2/models.py`) | The [official `infer_image`](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py) also uses bilinear output upsampling, but with `align_corners=True`. |
| Valid depth | Finite ground truth strictly between 0.001 m and 10 m | Local implementation in `valid_mask()`; exact author evaluation mask has not been verified. |
| NYU evaluation region | Eigen crop, rows `[45:471]`, columns `[41:601]`, enabled by `--eigen-crop` | Applied both when fitting alignment and when scoring. Exact author evaluation crop code has not been verified. |
| Relative-depth alignment | Per-image least-squares scale and shift from predicted disparity to inverse ground-truth depth on valid cropped pixels | Local MiDaS-style protocol; exact author alignment routine has not been verified. |
| Disparity floor | Clip aligned disparity to at least `1 / max_depth` (0.1 for NYU), then invert to depth | Local implementation in `eval_relative()`; exact author clamp has not been verified. |
| Reported metrics | Per-image AbsRel, RMSE, and δ1, averaged equally over 654 images per model | Compare AbsRel and δ1 with the paper's zero-shot relative-depth table. RMSE is an additional local metric. |
| Data | Ultralytics repackaging of the 654-image NYU Depth V2 Eigen test split | Third-party file, not verified byte-for-byte against the authors' test data. |

## Why the protocol changed

The original bicubic output resize produced negative disparity values,
consistent with interpolation overshoot. On five fixed NYU images, Large had
386 negative pixels in total;
bilinear output resize reduced this to zero. The code change is
[`d9b8c0e`](https://github.com/maaz-zubairi/depth-anything-v2-repro/commit/d9b8c0e).
On the 10-image check, bilinear alone changed the metrics only slightly and
Large still had the worst AbsRel. Adding the Eigen crop reduced Large's AbsRel
from 0.08648 to 0.04468 and removed that ranking anomaly. The crop changes
both the alignment pixels and the scored pixels, so this experiment does not
isolate which of those effects contributed most.

## Recorded full-run result

`results/nyu_summary.csv` contains the 654-image cropped result. Its
`results/nyu_per_image.csv` companion has 654 rows for each model, and the
summary values match the per-image means. On the 30 model-image rows shared
with the local bilinear-plus-crop 10-image check, the maximum AbsRel difference
is less than 0.00004; this is consistent with both fixes being applied to the
full run.

| Model | AbsRel ↓ | δ1 ↑ |
|---|---:|---:|
| V2 Small | 0.061712 | 0.955580 |
| V2 Base | 0.057324 | 0.959332 |
| V2 Large | 0.054759 | 0.960946 |

The [Depth Anything V2 paper, Table 2](https://proceedings.neurips.cc/paper_files/paper/2024/file/26cfdcd8fe6fd75cc53e92963a656c58-Paper-Conference.pdf)
reports V2-L on NYU-D as **0.045 AbsRel and 0.979 δ1**. Our Large result is
about 21.7% higher in AbsRel and 1.81 percentage points lower in δ1. The
third-party test file, Hugging Face port, and preprocessing and interpolation
differences are plausible contributors; their individual effects have not
been measured. These results should not be presented as an exact replication
of the authors' evaluation code.
