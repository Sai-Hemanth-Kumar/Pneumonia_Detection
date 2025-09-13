🫁 Pneumonia Detection using Deep Learning

This project implements a Deep Learning-based model for detecting Pneumonia from chest X-ray images. It leverages Convolutional Neural Networks (CNNs) to classify X-rays as Normal or Pneumonia, providing a step toward faster and more accurate medical diagnosis.

🚀 Features

📂 Preprocessing of chest X-ray dataset

🧠 CNN-based deep learning model for pneumonia classification

📊 Model evaluation with accuracy, confusion matrix, and loss/accuracy plots

💾 Saved trained model (.keras / .h5) for reuse

📸 Support for testing on new chest X-ray images

📂 Project Structure
Pneumonia_Detection/
│── Code/                # Source code for training & testing the model
│   ├── Pneumonia_Detection.ipynb  # Main Jupyter notebook
│   ├── All_3_Detection.keras      # Trained model
│   ├── ...                        # Other experiments/code
│
│── Documents/           # Project documentation & reports
│── Dataset/             # (Optional) Dataset folder (not uploaded here due to size)
│── README.md            # Project description

🛠️ Technologies Used

Python 3.x

TensorFlow / Keras

NumPy, Pandas

Matplotlib, Seaborn

Scikit-learn

Jupyter Notebook

📊 Dataset

The dataset used is the Chest X-Ray Images (Pneumonia) dataset, publicly available on Kaggle
.

Classes:

Normal

Pneumonia

🧑‍💻 How to Run the Project

Clone the repository:

git clone https://github.com/Sai-Hemanth-Kumar/Pneumonia_Detection.git
cd Pneumonia_Detection


Install dependencies:

pip install -r requirements.txt


Open the Jupyter Notebook:

jupyter notebook Code/Pneumonia_Detection.ipynb


Train or load the pre-trained model and test on chest X-ray images.

📈 Results

Achieved high accuracy in detecting pneumonia from chest X-rays.

Model performance was evaluated using metrics such as Accuracy, Precision, Recall, F1-score, and Confusion Matrix.