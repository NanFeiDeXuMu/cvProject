#!/usr/bin/env bash
# Download the datasets and Small checkpoints used in this project.
# Usage: bash scripts/download_data.sh   (run from the repository root)
set -euo pipefail

fetch() {  # fetch <url> <output path>
  if [ -s "$2" ]; then echo "skip  $2"; return; fi
  echo "fetch $2"
  mkdir -p "$(dirname "$2")"
  curl -L --fail --retry 20 --retry-all-errors --retry-delay 5 -C - -o "$2.part" "$1"
  mv "$2.part" "$2"
}

# Middlebury Stereo v3 (quarter resolution): inputs + left-view ground truth
fetch https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip data/raw/middlebury/MiddEval3-data-Q.zip
fetch https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip  data/raw/middlebury/MiddEval3-GT0-Q.zip
if [ ! -d data/middlebury/MiddEval3/trainingQ ]; then
  mkdir -p data/middlebury
  unzip -q -o data/raw/middlebury/MiddEval3-data-Q.zip -d data/middlebury
  unzip -q -o data/raw/middlebury/MiddEval3-GT0-Q.zip  -d data/middlebury
fi

# Small relative-depth checkpoints
fetch "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true" \
      checkpoints/depth_anything_v2_vits.pth
fetch "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth?download=true" \
      checkpoints/video_depth_anything_vits.pth

# NYU Depth V2 labeled set (~2.8 GB) and the official train/test split
fetch https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat         data/raw/nyu/splits.mat
bash "$(dirname "$0")/parallel_fetch.sh" \
  https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat data/raw/nyu/nyu_depth_v2_labeled.mat 16

echo "done"
