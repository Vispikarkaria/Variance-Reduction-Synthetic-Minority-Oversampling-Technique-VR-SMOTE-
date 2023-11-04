# Variance Reduction SMOTE (VR-SMOTE)

This repository contains a Python implementation of the Variance Reduction Synthetic Minority Over-sampling Technique (VR-SMOTE), which is an enhanced version of the Synthetic Minority Over-sampling Technique (SMOTE). The VR-SMOTE aims to improve dataset balancing by strategically oversampling minority class instances in a way that reduces variance and prevents overfitting.

## Background

Traditional SMOTE techniques can sometimes lead to overfitting when they oversample the minority class without considering the variance within the data. VR-SMOTE addresses this by first clustering the minority class samples using K-Nearest Neighbors (KNN) and then calculating a relative variance score (RV score) for each cluster. Clusters with high and medium variance are then oversampled, while clusters with low variance are not, in an effort to maintain a balanced variance across the dataset.

## Requirements

Before running this code, you'll need to install the required Python libraries. You can do this by running:

pip install -r requirements.txt


This will install the following packages:

- numpy
- scikit-learn
- imbalanced-learn

## Usage

The main function in `vr_smote.py` can be used as follows:

```python
from vr_smote import vr_smote

# Load your data
data = # ... (numpy array with shape [n_samples, n_features + 1])
# The last column in `data` should be the class/target variable

# Apply VR-SMOTE
balanced_data = vr_smote(data)

Please ensure your data is preprocessed accordingly, with numerical features and a binary target variable indicating class membership.

## Example
To run an example with synthetic data, you can use the script example.py:

python example.py

This will generate a synthetic dataset, apply VR-SMOTE, and print the shapes of the original and balanced datasets.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License
This project is open-sourced under the MIT License. See the LICENSE file for details.

## Citation
If you use this method in your research, please cite the following paper: