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
    
### 4. **TheThirdEye: Accessible Menu Reader and Allergy Detector for the Visually Impaired 👁️**
   - **Goal**: Creating a mobile app that helps visually impaired individuals read restaurant menus and detect potential allergens in food using advanced computer vision techniques.
   - **Collaboration**: Partnered with Perkins School for the Blind and Texas School for the Blind to ensure the app meets accessibility standards and addresses real-world challenges.
   - **Tech Stack**: Swift, OpenCV, Python.
   - **Features**:
     - Optical Character Recognition (OCR) for converting menu text into audio.
     - Computer vision (in progress) for detecting allergens such as nuts, dairy, and gluten in food descriptions.
     - Real-time voice feedback for accessibility.
    
### 5. **High-Performance Market Data Analysis (Jane Street) and Modeling 📊**
   - **Goal**: Analyzing real-time market data to identify trends and build predictive models for financial forecasting.
   - **Tech Stack**: Python, Dask, LightGBM, XGBoost, CatBoost, Polars, Scikit-learn, Matplotlib, Seaborn, Statsmodels.
   - **Features**:
     - Scalable data processing with Dask to handle datasets with millions of rows efficiently.
        - Missing Value Handle
             General forward and backward fill for missing data.
             Selective KNN imputation for columns with >5% missing values.
     - Feature selection based on correlation with target variables for improved model performance.
     - Model interpretability through permutation importance and statistical analysis.
     - Normality testing with Kolmogorov-Smirnov Test for target variable distribution.
   - **Visualization**:
   - Missing value analysis with bar plots.
   - Feature correlation with target variables using bar charts.
   - Distribution analysis of key features using histograms.
   - **Impact**:
   - Demonstrates advanced data analysis techniques for handling large-scale financial datasets.
   - Provides insights into feature importance and statistical behavior of financial data.
     


### 6. **Other Projects**
   - Coming soon... Stay tuned for more updates!

---

## 🛠️ Tools and Technologies
- **Programming Languages**: Python, Java, Swift
- **Libraries and Frameworks**: TensorFlow, PyTorch, Scikit-learn, Pandas, Matplotlib.
- **Cloud Platforms**: AWS (SageMaker, S3, Lambda).
- **Development Environments**: Google Colab, Jupyter Notebooks.




