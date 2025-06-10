# Micro-expression Model Training Code (18_2)

## Project Introduction
This project is for feature extraction, model training, and evaluation of micro-expression recognition, supporting various deep learning architectures for research and practical applications.

## Main Modules
- `train_cnn.py`: Train CNN models with flexible architecture and parameters.
- `train_lstm.py`: Train LSTM models for sequence feature modeling.
- `extract_features.py`: Feature extraction pipeline.
- `hpo_train.py`: Automated hyperparameter optimization.
- `feature_dataset.py`, `dataset.py`: Data loading, preprocessing, and augmentation.
- `models.py`: Deep learning model definitions.
- `utils.py`, `transforms.py`: Utility functions and data augmentation.

## Installation
```bash
git clone https://github.com/yourname/18_2.git
cd 18_2
pip install -r requirements.txt
```

## Usage
- Train CNN:
  ```bash
  python train_cnn.py --config config_cnn_stage1.yaml
  ```
- Train LSTM:
  ```bash
  python train_lstm.py --config config_lstm_stage2.yaml
  ```
- Feature extraction:
  ```bash
  python extract_features.py --config config_feature_extraction.yaml
  ```
- Hyperparameter optimization:
  ```bash
  python hpo_train.py
  ```

## Dataset
- Please refer to `feature_dataset.py` and `dataset.py` for dataset format and preprocessing.
- Custom features and labels are supported.

## Contribution Guide
See [CONTRIBUTING.md](CONTRIBUTING.md).

## License
MIT License

## Contact
1336495069@qq.com 