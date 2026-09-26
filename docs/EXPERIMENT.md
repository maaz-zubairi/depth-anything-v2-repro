# Experiment (Stage 4)

**One-sentence question:** Does Depth Anything V2's accuracy gain over V1 concentrate on the scene type the paper says it fixes (transparent / reflective surfaces), or is the gain roughly uniform across scene types?

**Type:** C - baseline comparison (V1 is the baseline). Supporting analysis: F - accuracy vs latency across S/B/L (justifies the deployment default).

**Why this is specific to the paper:** the paper's claim is that synthetic-label teacher training gives finer, more robust depth than V1 on hard content. A generic "bigger model is better" study does not test that.

**Test bed:** DA-2K (1,000 images, 2,000 human-checked relative pairs, 8 scene types, built by the paper's authors because standard benchmarks are noisy). Metric: pair accuracy (is the point annotated closer predicted closer?).

**Design:** same 3 sizes for V1 and V2 -> per-pair paired correctness -> per-scene gain (V2 - V1) with bootstrap 95% CI -> difference-in-differences: gain on `transparent_reflective` minus gain on all other scenes.

**Reading the result**
- DiD > 0 with CI excluding 0: gain concentrates where the paper says. Report it.
- DiD ~ 0: gain is uniform; report that honestly - it is still a finding.
- CI includes 0 (likely for small n): say the data cannot distinguish, and by how much.

**Threats to validity (put in Limitations)**
- `transparent_reflective` is ~10% of DA-2K (~100 images / ~200 pairs): wide CIs.
- V2 vs V1 differ in more than training data (V2 also decodes from intermediate DINOv2 layers instead of the last four; the authors report this did not change accuracy). Do not claim the gain is *caused* by synthetic data.
- Pair accuracy is sparse and ordinal: it says nothing about depth magnitudes or thin-structure sharpness.
- NYU cannot be used for this question: Kinect depth has holes on glass/mirrors, so ground truth is missing exactly where we want to measure.
- V2 labels/weights were built by the same authors as DA-2K.
