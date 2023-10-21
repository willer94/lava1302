


## 1 Structure
```
./
├── geometry.py
├── test_lines.py
├── render_depth.py
├── renderer.py
├── csv_files\
├── speedplus_gan\
├── speedplus_small\
├── synthetic_mask\
├── neus_meshes\

```
## 2 data description

### 2.1 csv_files
submission files
```
./csv_files
├── lightbox_ex_submission_1953_2022_03_30_21_34_58.csv
└── sunlamp_ex_submission_1953_2022_03_30_02_38_14.csv
```

### 2.2 speedplus_gan
Target-like source images, i.e., transferred source images into target domains using CycleGan.

```
./speedplus_gan
├── fake_lightbox [59961 entries exceeds filelimit, not opening dir]
└── fake_sunlamp_full [59961 entries exceeds filelimit, not opening dir]
```

### 2.3 speed_small
Resized original images, 640 $\times$ 400

```
./speedplus_small
├── camera.json
├── lightbox
│   ├── images
│   └── test.json
├── sunlamp
│   ├── images
│   └── test.json
└── synthetic
    ├── images
    ├── train.json
    └── validation.json
```
### 2.4 synthetic_mask
satellite masks

### 2.5 neus_meshes
meshs of tango
```
 neus_meshes
├── mesh1500.ply
├── mesh1500_simple.ply
├── mesh1500_simple_v2.ply
├── mesh1500_simple_v3.ply
├── mesh20000.obj
└── mesh.ply
```

## 3 python script
### 3.1 geometry.py
basic geometries, including landmarks, wireframes

### 3.2 renderer.py render_depth.py
render the .obj file given a pose

### 3.3 visualize.py
basic functions for visualization

### 3.4 test_lines.py
a simple demo to show 2D landmarks and wireframes given an input image


If this project helps you, please cite our paper.

```
@article{Wang2022RevisitingMS,
  title={Revisiting Monocular Satellite Pose Estimation With Transformer},
  author={Zi Wang and Zhuo Zhang and Xiaoliang Sun and Zhang Li and Qifeng Yu},
  journal={IEEE Transactions on Aerospace and Electronic Systems},
  year={2022},
  volume={58},
  pages={4279-4294},
  url={https://api.semanticscholar.org/CorpusID:247786112}
}
```
```
@article{Wang2023BridgingDG,
  author={Wang, Zi and Chen, Minglin and Guo, Yulan and Li, Zhang and Yu, Qifeng},
  journal={IEEE Transactions on Aerospace and Electronic Systems}, 
  title={Bridging the Domain Gap in Satellite Pose Estimation: A Self-Training Approach Based on Geometrical Constraints}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TAES.2023.3250385}}
```



