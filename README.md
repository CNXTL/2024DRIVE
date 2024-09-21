# <img src="https://github.com/xll0328/xll0328.github.io/blob/main/project/DRIVE/logo.png" class="floatpic" width="70" height="70"> DRIVE: Dependable Robust Interpretable Visionary Ensemble Framework in Autonomous Driving

Songning Lai~, Tianlang Xue~, Hongru Xiao, Lijie Hu, Jiemin Wu, Ninghui Feng, Runwei Guan, Haicheng Liao, Zhenning Li, Yutao Yue†

HKUST(GZ), Tongji University, KAUST, University of Maca

[![Website shields.io](https://img.shields.io/website?url=http%3A//poco.is.tue.mpg.de)](https://xll0328.github.io/project/DRIVE/) [![YouTube Badge](https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube)](https://NA)  [![arXiv](https://img.shields.io/badge/arXiv-2409.10330-00ff00.svg)](https://arxiv.org/abs/2409.10330) 

**Core code for project "DRIVE: Dependable Robust Interpretable Visionary Ensemble Framework in Autonomous Driving", based on Pytorch Lightning.**


<p align="center">
  <img src="https://github.com/xll0328/xll0328.github.io/blob/main/project/DRIVE/framwork_all.png" alt="The Dependably Robust Interpretable Visionary Ensemble (DRIVE) model framework.">
</p>


DRIVE is a comprehensive framework designed to address the reliability and stability issues in explanations of end-to-end unsupervised autonomous driving models, primarily aimed at enhancing the dependability and robustness of such systems.


## Features
- **Consistent and Stable Interpretability**: The DRIVE framework ensures that explanations remain consistent and stable, making the decision-making process transparent and predictable across various scenarios.
- **Consistent and Stable Output**: By guaranteeing consistent and stable output, DRIVE improves the performance of the model under different perturbations, enhancing the overall reliability of the system.

## Installation
1.Clone Repository
```bash
   git clone https://github.com/username/myproject.git
```
2.Environment Construction
```bash 
    conda install -r DRIVEcodeV1/requirement.txt
```

## Setup 
### 1.Dataset
Comma2k19 https://github.com/commaai/comma2k19

Download the comma2k19 data https://academictorrents.com/details/65a2fbc964078aff62076ff4e103f18b951c5ddb

### 2.DataPreprocessing
To extract video frames from the Comma2k19 dataset, obtain sample data, and save it to  HDF5 files, run raw_readers.ipynb and clean_up_data.ipynb.

## Arguments

| Argument           | Example           | Description   |
| -------------------| ----------------- | ------------- |
| `-task`            | `angle/distance/multitask` |Assign the task for model|
| `-train`           | `True`            |Bool, start training module|
| `-test`            | `False`           |Bool, start testing module|
| `-gpu_num`         |``1``              | Assign GPU number |
| `-dataset_path`    |`/path/to/dataset` |The path to dataset&checkpoints |
| `-concept_features`|`True`             |Bool, claim the usage of concept bottleneck|
| `-new_version`     |`True`             | Bool, claim training a new version of model rather than resuming from checkpoint|
| `-save_path`       | ``/path/to/savepath`` | The path for saving csv predictions |
| `-num-gpus`        | `1`               | Number of GPUs for model |
| `-max_epochs`      | `200`             | Set maximum epochs |
| `-bs`              | `4`               |Batch Size          |
|`-checkpoint_path`  |`path/to/checkpint.ckpt`|Assign a checkpoint for resuming training/testing|
|`img_noise`         |`GaussianNoise`    |Assign the noise for dataset，you might build a noisy dataset first via tools in ../attack folder

## Training Strategy
### 1.Train the model without DRIVE pipeline and obtain a *DCG* version of checkpoint.
Please refer to https://github.com/jessicamecht/concept_gridlock/blob/master/README.md

### 2.Resume training from the previous checkpoint with *DRIVE*.

Run  ```python3 main_copy1.py -dataset comma -backbone none -concept_features -ground_truth normal -train -gpu_num 1  -max_epochs 50 -task distance -bs 2 -checkpoint_path DCG_checkpoint``` 

More references in /sh_scripts

## Citing<a name="citing"></a>

If you find this document useful for your research, please consider citing the following repository and paper:
```bibtex
@misc{lai2024drivedependablerobustinterpretable,
      title={DRIVE: Dependable Robust Interpretable Visionary Ensemble Framework in Autonomous Driving}, 
      author={Songning Lai and Tianlang Xue and Hongru Xiao and Lijie Hu and Jiemin Wu and Ninghui Feng and Runwei Guan and Haicheng Liao and Zhenning Li and Yutao Yue},
      year={2024},
      eprint={2409.10330},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.10330}, 
}
```


