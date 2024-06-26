# README.md

## Overview

This repository contains two scripts aimed at predicting ADME (Absorption, Distribution, Metabolism, and Excretion) properties of chemical compounds using various machine learning models. The first script,  `fingerprint_gen.py` , generates molecular fingerprints from SMILES strings. The second script,  `innoplexus_adme.py` , uses these fingerprints to train and evaluate regression models on the TDC (Therapeutic Data Commons) ADME benchmark datasets.

## Scripts

### 1. fingerprint_gen.py

This script is responsible for generating molecular fingerprints from SMILES strings. It includes functions to compute different types of fingerprints and molecular descriptors.

### 2. innoplexus_adme.py

This script uses the fingerprints generated by  `fingerprint_gen.py`  to train and evaluate regression models on various TDC ADME benchmark datasets. The models used include LightGBM, XGBoost, CatBoost, and Random Forest.

#### Key Components:

- **Logging**: Configured to log information to both a file ( `model_output.log` ) and the console.
- **Benchmark Settings**: A dictionary defining the benchmark datasets and their settings.
- **Model Training and Evaluation**: For each benchmark dataset, the script:
  - Generates fingerprints for training and test data.
  - Imputes missing values.
  - Scales the target variable.
  - Trains LightGBM, Random Forest, XGBoost, and CatBoost models.
  - Evaluates each model using Mean Absolute Error (MAE).
  - Selects the best model based on MAE and logs the results.
- **Evaluation**: Uses the TDC ADME benchmark evaluation function to assess model performance.

#### Supported Benchmark Datasets:

-  `caco2_wang` 
-  `lipophilicity_astrazeneca` 
-  `solubility_aqsoldb` 
-  `ppbr_az` 
-  `vdss_lombardo` 
-  `half_life_obach` 
-  `clearance_hepatocyte_az` 
-  `clearance_microsome_az` 
-  `ld50_zhu` 

## How to Run

### Prerequisites

- Python 3.x
- Required Python packages:  `numpy` ,  `scikit-learn` ,  `rdkit` ,  `lightgbm` ,  `xgboost` ,  `catboost` ,  `tqdm` ,  `PyTDC` 

### Steps

1. **Install Required Packages**:
sh pip install numpy scikit-learn rdkit-pypi lightgbm xgboost catboost tqdm tdc

2. **Generate Fingerprints**:
    Ensure that your SMILES strings are stored in a pandas Series and call the  `generate_fingerprints`  function from  `fingerprint_gen.py` .

3. **Run ADME Prediction**:
    Execute the  `innoplexus_adme.py`  script to train and evaluate models on the TDC ADME benchmark datasets.
sh python innoplexus_adme.py

## Results

The results, including the best model for each benchmark dataset and its parameters, will be logged in the  `model_output.log`  file and displayed in the console. The final evaluation results will also be logged.

## Contact

Please contact [Rohit Yadav](mailto:rohit.yadav@ics.innoplexus.com) if you have any questions!

Feel free to contribute to this project by opening issues or submitting pull requests. For any questions or inquiries, please contact the repository maintainer.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

This project uses data and benchmarks provided by the Therapeutic Data Commons (TDC).

## Powered By

![Powered by NVIDIA](https://www.nvidia.com/en-us/about-nvidia/legal-info/logo-brand-usage/_jcr_content/root/responsivegrid/nv_container_392921705/nv_container/nv_image.coreimg.100.630.png/1703060329053/nvidia-logo-vert.png)
![Powered by XGBoost](https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/xgboost.png)
![Powered by CatBoost](https://upload.wikimedia.org/wikipedia/commons/c/cc/CatBoostLogo.png)
![Powered by LightGBM](https://lightgbm.readthedocs.io/en/stable/_images/LightGBM_logo_black_text.svg)