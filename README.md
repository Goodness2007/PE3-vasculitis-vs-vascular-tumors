# Vasculitis vs Vascular Tumors Classifier

CNN-based binary image classifier distinguishing **Vasculitis** from **Vascular Tumors**, deployed as a Streamlit web app.

Built for **GET 324 (AI, Machine Learning and Convergent Technologies)** - Laboratory Exercise 10 (Mini-Project), Group **PE3** (Petroleum Engineering).

## Overview

This project trains a deep learning model to classify dermatology images into one of two categories:
- **Vascular Tumors**
- **Vasculitis**

The trained model is deployed in a Streamlit app that lets a user upload an image and receive a prediction with a confidence score.

## Dataset

- Source: [DermNet dataset on Kaggle](https://www.kaggle.com/datasets/shubhamgoel27/dermnet)
- Classes used: `Vascular Tumors` and `Vasculitis Photos` (isolated from DermNet's 23 skin disease categories)
- Training set: 898 images (482 vascular tumors, 416 vasculitis), split 85/15 into train/validation
- Test set: 226 images (121 vascular tumors, 105 vasculitis)

## Model

- Architecture: Transfer learning using **MobileNetV2** (pretrained on ImageNet), frozen base with a custom classification head (GlobalAveragePooling2D, Dropout, single sigmoid output)
- Input size: 160x160 RGB images
- Data augmentation: random horizontal flip, rotation, and zoom
- Training: 10 epochs, Adam optimizer, binary cross-entropy loss

## Results

- Test accuracy: **86.28%**
- Precision/Recall/F1: ~0.86-0.89 across both classes (no significant class bias)

| | Precision | Recall | F1-score |
|---|---|---|---|
| Vascular Tumors | 0.89 | 0.85 | 0.87 |
| Vasculitis | 0.84 | 0.88 | 0.86 |

## How to Run Locally

```bash
git clone https://github.com/Goodness2007/PE3-vasculitis-vs-vascular-tumors.git
cd PE3-vasculitis-vs-vascular-tumors
python -m venv venv
venv\Scripts\activate   # On Windows
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`), upload a skin image, and view the prediction.

## Deployed App

[Live Streamlit app link - to be added]

## Project Team (Group PE3)

- [Name, registration number, GitHub username]
- [Name, registration number, GitHub username]

## Files

- `app.py` - Streamlit application source code
- `vasculitis_vs_vascular_tumors.keras` - trained CNN model
- `requirements.txt` - Python dependencies
