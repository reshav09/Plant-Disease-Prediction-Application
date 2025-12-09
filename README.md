# 🌾 Predictive Crop Disease Detection and Management System

A comprehensive web-based application that uses deep learning to detect and predict crop diseases from plant leaf images, providing management strategies for disease prevention and treatment.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Training the Model](#training-the-model)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Results](#results)

## 🎯 Overview

This system enables farmers and agricultural experts to:
- Upload images of plant leaves
- Detect whether the plant is healthy or diseased
- Identify specific disease types (leaf spot, rust, mildew, etc.)
- Receive preventive measures and management strategies
- Get instant predictions through a user-friendly web interface

## ✨ Features

- **Image Upload**: Easy drag-and-drop or file upload interface
- **Disease Detection**: Accurate classification using deep learning
- **Multi-Model Support**: CNN, ResNet50, EfficientNetB0, MobileNetV2
- **Disease Information**: Detailed descriptions and symptoms
- **Management Strategies**: Prevention and treatment recommendations
- **Confidence Scores**: Prediction confidence visualization
- **Responsive UI**: Clean and intuitive Streamlit interface

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| Deep Learning | TensorFlow/Keras |
| Web Framework | Streamlit |
| Dataset | PlantVillage (Kaggle) |
| Image Processing | OpenCV, Pillow |
| Visualization | Matplotlib, Plotly |

## 📦 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd "Plant Disease Prediction"
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 📊 Dataset Setup

### Option 1: Download from Kaggle (Recommended)

1. **Get Kaggle API Credentials**:
   - Go to [Kaggle Account Settings](https://www.kaggle.com/account)
   - Click "Create New API Token"
   - Download `kaggle.json`

2. **Place kaggle.json**:
   - Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`

3. **Download Dataset**:
```bash
python download_dataset.py
```

### Option 2: Manual Download

1. Visit [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
2. Download and extract to `data/plantvillage/` folder

## 🏋️ Training the Model

### Train with Default Settings (EfficientNetB0)
```bash
python train.py
```

### Train with Specific Model
```bash
# Custom CNN
python train.py --model custom_cnn --epochs 50

# ResNet50
python train.py --model resnet50 --epochs 30

# MobileNetV2
python train.py --model mobilenetv2 --epochs 30
```

### Training Parameters
- `--model`: Model architecture (custom_cnn, resnet50, efficientnetb0, mobilenetv2)
- `--epochs`: Number of training epochs (default: 25)
- `--batch_size`: Batch size for training (default: 32)
- `--img_size`: Image size (default: 224)
- `--learning_rate`: Learning rate (default: 0.001)

## 🚀 Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
Plant Disease Prediction/
├── app.py                      # Streamlit web application
├── model.py                    # Model architectures
├── train.py                    # Training script
├── utils.py                    # Utility functions
├── disease_info.py             # Disease information database
├── download_dataset.py         # Dataset download script
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore file
├── data/                       # Dataset directory
│   └── plantvillage/
│       ├── train/
│       └── val/
├── models/                     # Saved models
│   └── best_model.h5
└── logs/                       # Training logs
```

## 🧠 Model Architecture

### 1. Custom CNN
- 4 Convolutional blocks with MaxPooling
- Batch Normalization and Dropout
- Dense layers with regularization

### 2. Transfer Learning Models
- **ResNet50**: Deep residual learning
- **EfficientNetB0**: Efficient scaling
- **MobileNetV2**: Lightweight for deployment

All models use:
- ImageNet pre-trained weights
- Fine-tuning on PlantVillage dataset
- Data augmentation for better generalization

## 📈 Results

Expected Performance:
- **Accuracy**: 95-98% on validation set
- **Inference Time**: < 1 second per image
- **Model Size**: 
  - Custom CNN: ~50 MB
  - EfficientNetB0: ~20 MB
  - ResNet50: ~90 MB
  - MobileNetV2: ~15 MB

## 🌱 Supported Crops and Diseases

### Crops (14 species):
- Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

### Disease Categories (38+):
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Mosaic Virus
- Yellow Leaf Curl Virus
- And more...

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 👥 Authors

- Your Name

## 🙏 Acknowledgments

- PlantVillage Dataset by Penn State University
- TensorFlow and Keras teams
- Streamlit for the amazing framework

## 📞 Support

For issues and questions, please open an issue on GitHub.

---
Made with ❤️ for Agriculture and Technology
