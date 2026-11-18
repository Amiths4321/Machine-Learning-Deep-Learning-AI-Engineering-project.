
# Industrial Steel Surface Defect Classification

## Project Overview
A deep learning application that classifies industrial steel surface images into six defect categories using a custom Convolutional Neural Network (CNN).

The trained model is integrated into a Streamlit web application for image upload and real-time defect prediction.

## Defect Categories
1. Crazing
2. Inclusion
3. Patches
4. Pitted Surface
5. Rolled-in Scale
6. Scratches

## Dataset
- Total images: 1,440
- Training images: 1,008
- Validation images: 216
- Test images: 216
- Number of classes: 6

## Model Performance

| Model | Validation Accuracy | Test Accuracy |
|---|---:|---:|
| Original CNN | 94.44% | 95.37% |
| Regularized CNN | 96.76% | 96.76% |

The selected regularized CNN correctly classified 209 out of 216 test images.

## Technologies Used
- Python
- PyTorch
- Torchvision
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Pillow

## Application Features
- Upload steel surface images
- Predict defect category
- Display prediction confidence
- Display probabilities for all six classes
- Show prediction history
- Warn users about low-confidence predictions

## Run Locally

Install dependencies:

```bash
pip install torch torchvision streamlit scikit-learn matplotlib seaborn pillow
```

Launch the application from the project root:

```bash
python -m streamlit run app.py
```

## Model Location

`notebook/models/final_steel_defect_cnn.pth`

## Project Structure

```text
project_02_deep_learning/
├── app.py
├── README.md
├── data/
│   └── raw/
├── models/
├── notebook/
│   └── models/
│       └── final_steel_defect_cnn.pth
└── notebooks/
```

## Results
The regularized CNN achieved 96.76% accuracy on the held-out test dataset.

This project demonstrates image classification, CNN development, model evaluation, PyTorch inference, and deployment through Streamlit.
