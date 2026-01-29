# transaminase

This repository provides code for the paper "[placeholder]" and the associated Supplementary Information, which can be found at [placeholder]. 

_Code Author: Priyanka Raghavan_

## Install and setup

After git cloning the repository, please run the following to build the environment and extract the dataset files.

```
conda env create -f environment.yml
conda activate transaminase_env
tar -xf dataset_files.tar.gz
python setup.py develop
```

This environment was tested on Ubuntu 20.04.5 with CUDA Version 11.7. It should take less than 10 minutes to solve.

## Data

All datasets are contained within the `dataset_files` folder, with the subfolders below corresponding to the various sections of this study, as outlined in the paper: 

1. `20250612_transaminase_dataset_dft_descriptors_boltzmann.csv`: contains the initial transaminase HTE dataset on 42 ketones
2. `additional_cyclic_dataset_dft_descriptors_boltzmann.csv`: contains the held-out dataset of cyclic ketones used to validate the models built on the initial dataset
3. `scaleup_ketones_dataset_dft_descriptors_boltzmann.csv`: contains the dataset of ketones used for the scale-up experiments

## General Notes and Disclaimer

Several parts of the code require some combination of the following inputs: `model_type`, `split_type`, `task_type`, `feature_type`, `target_type`. For each, certain keywords can be used:

* `model_type`: can take as input `'rf'` or `'lin'`, or `'nn'` (for the Random Forest and Linear Regression models, respectively)
* `task_type`: can take as input `'bin'` or `'reg'` (for binary classification and regression, respectively)
* `feature_type`: can take as input `'ohe'`, `'fgp'`, `'physchem'`, `'dft'`, or `'physchemdft'` (for one-hot encoded, fingerprint-only, physicochemical-only, DFT-only, and concatenated physicochemical and DFT features, respectively). Note that the DFT options will use pre-selected DFT features - one can also give a list of specified DFT features, either global or atom-level, i.e. `['electronegativity', 'Mulliken_charge.Carbon1', 'B5.CC1']`, which will use the molecule electronegativity, Mulliken charge of the smaller alpha carbon, and Sterimol B5 down the Carbonyl-smaller alpha carbon bond.
* `feature_type`: can take as input `'ohe'`, `'fgp'`, `'dft'`, or `'fgpdft'` (for one-hot encoded, fingerprint-only, DFT-only, and concatenated fingerprint and DFT features, respectively)

## Training and Using Models

The easiest way to train and deploy the per-enzyme models built from the initial dataset is to follow the Jupyter notebooks, located in the `src/notebooks` subfolder.

* `original_ketones_modeling.ipynb`: if you would like, run the cells in order to build the conversion and selectivity models from the initial dataset. Relevant predictions and the selected features obtained by the MVLR modeling will then be saved to the `src/results` directory. Note that this directory has already been pre-populated with the results we obtained, so you need not run this notebook if you don't wish.

* `heldout_modeling.ipynb` and `scaleup_modeling.ipynb`: these notebooks show examples of deploying the best trained models (RF/DFT features for conversion, and Linear Regression with selected features for selectivity) to generate predictions on new ketones. To generate predictions on your own data, simply replace the path to the `heldout_df` in the first cell after the imports, and then run all cells in order. Your .csv file must be in the same format as the `additional_cyclic_dataset_dft_descriptors_boltzmann.csv` and `scaleup_ketones_dataset_dft_descriptors_boltzmann.csv` files in order to run correctly.  

