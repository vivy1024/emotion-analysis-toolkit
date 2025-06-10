# Macro-expression Training Code (fer2013model)

## Project Introduction
This project is for training and evaluating macro-expression recognition models, supporting public datasets such as FER2013, suitable for affective computing and expression recognition research and applications.

## Main Modules
- `multi_dataset_train.py`: Main script for multi-dataset model training, including training, data loading, and evaluation.
- `emotion_model.py`: Inference and single-image expression recognition script.
- `FacialExpressionRecognition.spec`: PyInstaller config for building executables.

## Installation
```bash
git clone https://github.com/yourname/fer2013model.git
cd fer2013model
pip install -r requirements.txt
```

## Usage
- Train model:
  ```bash
  python multi_dataset_train.py --dataset fer2013
  ```
- Inference and evaluation:
  ```bash
  python emotion_model.py --input your_image.jpg
  ```
- See script comments for parameter details.

## Dataset
- Recommended to use public datasets such as FER2013. Download and place as required.
- See `multi_dataset_train.py` for dataset format and preprocessing.

## Contribution Guide
See [CONTRIBUTING.md](CONTRIBUTING.md).

## License
MIT License

## Contact
1336495069@qq.com 