# data/ (git-ignored)

Layout expected by the code:

```
data/
  nyu/    <- unzip of nyu-depth.zip; must contain images/val/*.{jpg,png} and depth/val/*.npy
  da2k/   <- Hugging Face dataset depth-anything/DA-2K (contains annotations.json + images)
  kitti/  <- optional stretch: manifest.txt with lines "rgb_rel_path depth_rel_path"
```

See the Colab notebook or README "Data setup" for the download commands.
