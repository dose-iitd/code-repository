# code-repository

## Getting started

Create a local clone using command:
```bash
git clone https://github.com/dose-iitd/code-repository
```

Use environment.yml file provided to reproduce the conda environment using 

```bash
conda env create -f environment.yml
```

Download the datasets and change location `cs_path`, `nd_path`, `dz_path`, `fd_path`, `fz_path`, `acdc_path` in `train.py` file for Cityscapes, Nighttime driving, Dark Zurich, Foggy Driving, Foggy Zurich and ACDC datasets respectively.
Pre-trained model path are passed as command line arguments in shell script files provided.

All pre-trained models can be downloaded using the following [[Download link](https://drive.google.com/drive/folders/1wGFcJF4vwiri4nRwieul4RqRVnL0ziZX?usp=sharing)]


Pretrained models used: `IBNNet`, `Switchable Whitening`, `RobustNet` 


```bash
sh train_dg.sh
```
