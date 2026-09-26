# Evaluation protocol decisions (fill in, then freeze BEFORE running the full test set)

| Decision | Our choice | Matches paper? | Evidence (paper table / appendix / official code line) |
|---|---|---|---|
| Checkpoints | Relative (disparity) S/B/L | | |
| Alignment | per-image least-squares scale+shift in disparity space | | |
| Disparity floor / depth cap | 1/max_depth; NYU 10 m, KITTI 80 m | | |
| Metrics | AbsRel, delta1 (compare to paper); RMSE (ours only, no paper number in relative table) | | |
| NYU crop | none (`--eigen-crop` available) | | |
| Input resolution | HF processor default (short side 518, multiple of 14) | | |
| Upsampling | bicubic to original size | | |
| RMSE vs paper | paper reports RMSE for the *metric* fine-tuned checkpoints, not relative | | optional stretch track |

Rule: change a row only with a written reason; a changed protocol is a different experiment.
