# Abhi's Portfolio Projects 🎯

Welcome to my portfolio of machine learning projects! This repository showcases a variety of projects I've worked on in my free time, each reflecting my passion for exploring data, solving meaningful problems, and leveraging machine learning techniques to create impactful solutions.

---

## 🚀 About Me

Hi, I'm Abhinav Bulusu! As an **Electrical and Computer Engineering** student, I'm passionate about combining cutting-edge technology with real-world applications. 
---

## 📂 Project Highlights

### 1. **AI-Powered Climate Risk Analysis Platform 🌍**
   - **Goal**: Analyzing climate risks using machine learning and AWS services.
   - **Tech Stack**: Python, AWS (SageMaker, S3), Scikit-learn, Pandas.
   - **Features**: 
     - Predictive modeling for climate risk.
     - Data visualization for actionable insights.
   - **Status**: In progress 🚧

### 2. **Heart Failure Prediction 🫀**
   - **Goal**: Predicting the likelihood of heart disease using multiple machine learning models.
   - **Tech Stack**: Python, Scikit-learn, Pandas, Matplotlib, Seaborn.
   - **Features**:
     - Data preprocessing: Label encoding and feature scaling.
     - Model comparison: Logistic Regression, SVM, KNeighborsClassifier, Decision Tree, and Random Forest.
     - Addressing overfitting: Constraints and cross-validation.
     - **Accuracy Highlights**:
       - Logistic Regression: 86.34%
       - SVM: 85.37%
       - KNeighbors: 88.29%
       - Random Forest (with constraints): 92.29% (cross-validated).
   - **Unique Feature**: Single prediction based on user input:
     ```python
     result = model_svm.predict(sc.transform([[30,0,1,140,290,1,0,160,1,1,1,1,1]]))
     print("Heart Disease" if result == 1 else "No Heart Disease")
     ```
   - **Visualization**: Model accuracy comparison with bar charts.

### 3. **Customer Segmentation with ML 🛒**
   - **Goal**: Leveraging clustering techniques to identify customer segments.
   - **Tech Stack**: Python, Scikit-learn, Matplotlib.
   - **Features**:
     - Clustering algorithms like K-Means and DBSCAN.
     - Visualizing customer behavior patterns.

### 4. **Other Projects**
   - Coming soon... Stay tuned for more updates!

---

## 🛠️ Tools and Technologies
- **Programming Languages**: Python, Java.
- **Libraries and Frameworks**: TensorFlow, PyTorch, Scikit-learn, Pandas, Matplotlib.
- **Cloud Platforms**: AWS (SageMaker, S3, Lambda).
- **Development Environments**: Google Colab, Jupyter Notebooks.




