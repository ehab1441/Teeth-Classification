# 🦷 Teeth Classification using Pre-Trained CNNs

This project implements **Teeth Classification** using **pre-trained models** like **EfficientNet** and **DenseNet121** for better accuracy and performance. The model is fine-tuned on a dataset of dental images and deployed using **Streamlit** for real-time classification.

---

## 📈 Project Overview

✅ **Preprocessing & Augmentation**: Normalizes images and applies **rotation, zoom, and flipping** for better generalization.  
✅ **Fine-Tuned CNN Model**: Uses **DenseNet121** (or EfficientNet) as a **feature extractor**, with custom classification layers.  
✅ **Training & Evaluation**: Fine-tuned for **better accuracy**, monitored using **precision, recall, and confusion matrix**.  
✅ **Streamlit Deployment**: Allows **real-time classification** of new images via a web app.  
✅ **Visualization**: Displays **original & augmented images**, and model performance graphs.  

---

## 🚀 Installation

### **1. Clone the Repository**
```sh
git clone https://github.com/your-repo/teeth-classification.git
cd teeth-classification
pip install tensorflow numpy matplotlib streamlit
```

### **2. Prepare the Dataset**
Ensure your dataset follows this structure:
```
Teeth_Dataset/
    Training/
        CaS/
        CoS/
        Gum/
        MC/
        OC/
        OLP/
        OT/
    Validation/
        ...
    Testing/
        ...
```

---

## 📊 Usage

### **1. Train the Model**
Run the **`Pre-Trained-Model.ipynb`** notebook to:
- Load **DenseNet121** 
- Freeze layers & add new classification layers  
- Fine-tune the model on **Teeth_Dataset**  
- Evaluate using **precision, recall, and confusion matrix**  

### **2. Deploy the Streamlit App**
Run the Streamlit web app to classify new teeth images:
```sh
streamlit run app.py
```
This allows users to **upload an image** and get a **real-time prediction** with confidence scores.

---

## 📈 Model Performance
- The **fine-tuned DenseNet121 model** achieves **good accuracy**
- **Confusion Matrix & Precision-Recall Curves** help analyze misclassifications.

---

## 📈 Results
The model achieves:
✅ **High Validation Accuracy** (~85%)  
✅ **Improved Recall & Precision** using fine-tuning  
✅ **Optimized Inference Speed** in Streamlit  

---
