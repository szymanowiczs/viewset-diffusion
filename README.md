# viewset-diffusion
Implementation of `Viewset Diffusion: (0-)Image-Conditioned 3D Generative Models from 2D Data'.

This is the first release of the code, this README will get more comprehensive with time.
Do report bugs if you spot them and contributions are always welcome.

## Requirements

Install requirements and activate conda environment.
```
conda env create -f environment.yml
conda activate viewset_diffusion
```

To train Viewset Diffusion models you will need 1x24GB GPU for training on the Minens dataset and 2x48GB GPUs for training on CO3D and ShapeNet Cars. 

# Data

### Minens dataset

1. Download files from [skin_names](https://drive.google.com/drive/folders/13Gk6nJnBjDdl3N2_yjmSomRg6gLrpIZv) to `data/' so that `data/' contains two folders: data/meta and `data/meta2'.

2. Run `bash ./download.sh` to download skins from https://planetminecraft.com and
https://minecraftskins.com. It will scrape a list of skin names in `data/meta*`
and then a list of skin PNGs in `data/skins*` and save them to `data/skins` and
`data/skins2`, respectively.

3. Download files from [benchmark_minens](https://drive.google.com/drive/folders/13Gk6nJnBjDdl3N2_yjmSomRg6gLrpIZv) into `data/benchmark_mine`. Run `python render_minecraft_dataset.py`. 
It will rasterize images of articulated, textured Minecraft characters and save the dataset
as a `.npy` file.

4. Set the value of `data.data_path` in `config/dataset/minens.yaml` to the absolute path of the directory where minens was rendered. 

Files `data/benchmark_mine/cameras_*` and `data/benchmark_mine/poses_*` hold camera poses and character poses in each dataset, respectively.
`data/benchmark_mine/training_skins.txt` and `data/benchmark_mine/testing_skins.txt` hold skin names of each character.
`data/benchmark_mine/testing_bkgds.npy` holds the background colours in each file.
`data/benchmark_mine/random_order_of_sample_idxs.npy` and `data/benchmark_mine/ambiguous_pose.npy` hold file indices used for evaluation.


### CO3D 

Download CO3D following instructions from [official website](https://github.com/facebookresearch/co3d). 
Put the path of the download directory in `data_manager/co3d.py` as `CO3D_DATASET_ROOT`.
Set the value of `data.data_path` in `config/dataset/co3d_hydrant.yaml` to the absolute path of where you want to store a compressed, fast-to-access .npy version of the CO3D classes. 
The first time training is run, the data manager will form train, val and test dataset and store them in that path.

If you want to run quantitative evaluation and reproduce scores from the paper you also need to download the indexes of samples I used in evaluation.
Download `random_order_of_sample_idxs.npy` for each class from the folders with models for [CO3D Hydrants](https://drive.google.com/drive/folders/1P8n6gZlTdzhiMFSSOgaY_p_XTtZm-zB-?usp=sharing), [CO3D Teddybears](https://drive.google.com/drive/folders/14NL_uz-3c1nSPCCWO2A_vjVjeNSxkBHH?usp=share_link), [CO3D Vases](https://drive.google.com/drive/folders/16U3mapGEFMNa6Pajb8ERJBTBqTrCeZyz?usp=share_link) or [CO3D Plants](https://drive.google.com/drive/folders/1_jEPkuukdKXzihmg72to9TVkPon-Mkfb?usp=sharing).
Place the order of files for evaluation in the same folder as the other .npy files.

### SRN-ShapeNet Cars

Download the ShapeNet-SRN Car dataset from [PixelNeRF project page](https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR).
Set the value of `data.data_path` in `config/dataset/srn_cars.yaml` to the absolute path of the SRN Dataset root.
The data should have the structure `SRN_ROOT/srn_cars/srn_cars/cars_{train,val,test}/{example_id}`. 

# Training

Specifying arguments for training follows the hydra configuration syntax, for example:

```
python train.py +experiment=diffusion_128 +arch=up_3d_conv +dataset=srn_cars
```

### Training options

**+experiment**: set to `diffusion_128` for training on 128x128 resolution (CO3D, ShapeNet Cars) and `diffusion_48` for training on 48x48 resolution (Minens).

**optimization.n_iter**: set to `104,000` for training on ShapeNet Cars, `200,000`-`300,000` for training on CO3D classes and `200,000` for training on Minens.

**+arch**: set to `up_3d_conv` for training Viewset Diffusion. `triplanes` can be used to train RenderDiffusion, more information on that coming soon.

**+dataset+**: set to `co3d_{hydrant,plant,teddybear,vase}` for CO3D classes, `srn_cars` for ShapeNet Cars and `minens` for Minens.

# Pretrained models

ShapeNet and CO3D checkpoints are now released - enjoy experimentation!
You can now download our models for [ShapeNet Cars](https://drive.google.com/drive/folders/1837UhwVTFNbozUI7RKFdVpFOjTETVvMz?usp=sharing), [CO3D Hydrants](https://drive.google.com/drive/folders/1P8n6gZlTdzhiMFSSOgaY_p_XTtZm-zB-?usp=sharing), [CO3D Teddybears](https://drive.google.com/drive/folders/14NL_uz-3c1nSPCCWO2A_vjVjeNSxkBHH?usp=share_link), [CO3D Vases](https://drive.google.com/drive/folders/16U3mapGEFMNa6Pajb8ERJBTBqTrCeZyz?usp=share_link) and [CO3D Plants](https://drive.google.com/drive/folders/1_jEPkuukdKXzihmg72to9TVkPon-Mkfb?usp=sharing).

Checkpoint for Minens will follow soon. 

# Evaluation

Once a model is trained, 3D reconstruction can be run with:
```
python evaluation_qualitative.py [model_path] --dataset_name [dataset_name] --N_clean 1 --N_noisy 3 --cf_guidance 0.0 --seed 0
```

3D Generation can be run with:
```
python evaluation_qualitative.py [model_path] --dataset_name [dataset_name] --N_clean 0 --N_noisy 4 --cf_guidance -1.0 --seed 0
```

Quantitative evaluation of 3D reconstruction is run with
```
python evaluation_quantitative.py model_path=[model path]
```
# BibTeX

If you find this code useful, please consider citing
```
@inproceedings{szymanowicz23viewset_diffusion,
      title={Viewset Diffusion: (0-)Image-Conditioned {3D} Generative Models from {2D} data},
      author={Stanislaw Szymanowicz and Christian Rupprecht and Andrea Vedaldi},
      year={2023},
      booktitle={ICCV},
}
```
