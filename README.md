
# Forest Fire Detection Using Satellite Imagery 🌲🔥🛰️

## 📌 Overview
This project aims to detect forest fires from satellite imagery using deep learning techniques. Leveraging convolutional neural networks (CNNs), the model classifies images into categories such as fire or no fire. The goal is to provide an automated and scalable solution for early fire detection to help mitigate environmental and economic losses.

## 🚀 Live App
👉 [Click here to try the app](https://forestfiredetectionusingsatelliteimagery-devanshusawarkar.streamlit.app/)

## 🚀 Features
- Classification of satellite images into fire/no-fire categories.
- Preprocessing pipeline for image augmentation and normalization.
- Deep learning model built using TensorFlow/Keras.
- Evaluation metrics: accuracy, loss visualization.
- Supports custom datasets and transfer learning.

## 🧠 Technologies Used
- Python 🐍
- TensorFlow & Keras 📦
- NumPy 📊
- Jupyter Notebook 📓

## 📁 Dataset
- [Dataset](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset)
- The dataset used consists of satellite images labeled based on fire presence. The images are preprocessed (resized, normalized) before being fed into the model.

> 📦 Note: You can customize the dataset path in the notebook as per your directory structure.

## 🏗️ Model Architecture
The CNN model includes:
- Convolutional layers with ReLU activation
- Max Pooling layers
- Dropout regularization
- Fully Connected (Dense) layers
- Output layer with sigmoid activation for binary classification

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DevanshuSawarkar/Forest_Fire_Detection.git
   cd Forest_Fire_Detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Results
- Achieved high accuracy on validation dataset.
- Training and validation accuracy/loss visualized using matplotlib.
- Capable of generalizing well to unseen satellite images.

## ▶️ Usage
- Open the Jupyter Notebook:
  ```bash
  jupyter notebook Forest_Fire_Detection_Using_Satellite_Imagery.ipynb
  ```
- Run each cell sequentially to train and evaluate the model.
- Modify the image directory path if you're using a different dataset.

## 📈 Evaluation Metrics
- Accuracy
- Loss
- Confusion Matrix (optional enhancement)
- Precision, Recall, F1-score (optional enhancement)

## 📌 Future Work
- Integrate real-time satellite feeds.
- Deploy as a web-based alert system.
- Enhance accuracy with larger datasets and advanced architectures.

## 📄 License
This project is licensed under the MIT License

## 🙌 Acknowledgments
- Satellite image datasets used for training and evaluation.
- TensorFlow, Keras, and open-source community.

---

Made with ❤️ by [Devanshu Sawarkar](https://github.com/DevanshuSawarkar)
