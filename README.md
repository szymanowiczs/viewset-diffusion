# viewset-diffusion
Implementation of `Viewset Diffusion: (0-)Image-Conditioned 3D Generative Models from 2D Data'.

## Rendering Minecraft dataset

1. Download files from [skin_names](https://drive.google.com/drive/folders/13Gk6nJnBjDdl3N2_yjmSomRg6gLrpIZv) to `data/' so that `data/' contains two folders: data/meta and `data/meta2'.


2. Run `bash ./scrape.sh` to download skins from https://planetminecraft.com and
https://minecraftskins.com. It will scrape a list of skin names in `data/meta*`
and then a list of skin PNGs in `data/skins*` and save them to `data/skins` and
`data/skins2`, respectively.

3. Download files from [benchmark_minens](https://drive.google.com/drive/folders/13Gk6nJnBjDdl3N2_yjmSomRg6gLrpIZv) into `data/benchmark_mine`. Run `python render_minecraft_dataset.py`. 
It will rasterize images of articulated, textured Minecraft characters and save the dataset
as a `.npy` file.

Files `data/benchmark_mine/cameras_*` and `data/benchmark_mine/poses_*` hold camera poses and character poses in each dataset, respectively.
`data/benchmark_mine/training_skins.txt` and `data/benchmark_mine/testing_skins.txt` hold skin names of each character.
`data/benchmark_mine/testing_bkgds.npy` holds the background colours in each file.