# **Heart Disease Prediction using Machine Learning**

This repository contains the implementation of a **machine learning-based predictive model** for heart disease detection. The model leverages **XGBoost, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM)** algorithms to analyze patient data and predict the likelihood of heart disease, facilitating early intervention.

---

## **Introduction**

Cardiovascular diseases are the leading cause of global mortality, with fatalities expected to rise to **22 million by 2030**. Early detection of heart disease can significantly improve treatment outcomes and reduce healthcare costs. This project presents a **Heart Disease Prediction System** using **machine learning and data mining techniques**, offering a cost-effective, efficient, and accurate diagnosis method.

---

## **Features**

- **Utilization of ECG data** for accurate early diagnosis.
- **Machine learning models (XGBoost, SVM, KNN)** for classification.
- **Data preprocessing and feature selection** for optimized predictions.
- **Performance evaluation using confusion matrix, accuracy, precision, recall, and F1-score.**
- **Python-based implementation with visualization tools for insights.**

---

## **Dataset**

The dataset used for this project is obtained from the **UCI Machine Learning Repository**. It contains various patient attributes, including:
- Age
- Blood Pressure
- Cholesterol Levels
- Blood Sugar
- ECG Results

These features are utilized to classify patients based on their likelihood of developing heart disease.

---

## **How to Run**

### **Prerequisites**

1. Install **Python 3.8+** on your system.
2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### **Steps**

1. **Preprocess the Dataset**:
   ```bash
   python preprocess.py
   ```
2. **Train the Machine Learning Models**:
   ```bash
   python train.py
   ```
3. **Evaluate the Model**:
   ```bash
   python evaluate.py
   ```
4. **Make Predictions**:
   ```bash
   python predict.py --input <patient_data.csv>
   ```

---

## **Model Architecture & Algorithms Used**

This project integrates multiple machine learning models to enhance prediction accuracy:

- **K-Nearest Neighbors (KNN)**: Classifies patients based on similarity in health attributes.
- **Support Vector Machine (SVM)**: Efficiently separates high-dimensional feature spaces.
- **XGBoost (Extreme Gradient Boosting)**: Boosts model accuracy with optimized decision trees.

### **Proposed Approach**
1. **Data Preprocessing**:
   - Cleaning and handling missing values.
   - Normalization and feature scaling.
2. **Feature Selection**:
   - Identifying significant features affecting heart disease.
3. **Model Training**:
   - Training using **KNN, SVM, and XGBoost**.
4. **Evaluation Metrics**:
   - **Accuracy, Precision, Recall, F1-score, and Confusion Matrix.**

---

## **Experiments and Observations**

### **Baseline Model: Logistic Regression**
- Accuracy: **82%**
- Observation: Standard baseline performance.

### **KNN & SVM Performance**
- Accuracy: **85% - 88%**
- Observation: Showed improved classification but lacked fine-grained feature separation.

### **Final Model: XGBoost**
- Accuracy: **92%**
- Observation: Provided the best balance of accuracy and efficiency.

---

## **Performance Analysis**

To evaluate the effectiveness of our models, we used:

- **Confusion Matrix**: Breakdown of true/false positives and negatives.
- **Precision & Recall**: Measures correctness and sensitivity of predictions.
- **F1-Score**: A balanced metric considering both precision and recall.

---

## **Technologies Used**

- **Python 3.8+**
- **Scikit-Learn, XGBoost**
- **NumPy, Pandas**
- **Matplotlib, Seaborn**
- **Flask (for Deployment - Future Enhancement)**

---

## **Future Enhancements**

- Deploying as a **web-based application** for real-time patient risk analysis.
- Expanding dataset size and diversity for improved generalization.
- Integration with **real-time ECG monitoring** for predictive analytics.

---

## **Contributors**

- **G. Pavan Chaitanya**
- **Papishetty Prathima**
- **Pallala Priskilla**

Under the guidance of **Dr. Bipin Bihari Jayasingh, HOD - IT** at **CVR College of Engineering**.

---
