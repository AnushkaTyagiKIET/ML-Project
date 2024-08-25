# ML-Project

### Project Title: Heart Disease Prediction using Machine Learning

**Abstract**:
This project aims to predict heart disease using various machine learning algorithms and feature selection techniques. By analyzing three datasets from the UCI Machine Learning Repository, we explore the effectiveness of different classifiers and feature selection methods in accurately identifying individuals at risk of heart disease.

**Key Contributions**:
1. _Comprehensive Algorithm Implementation_: We've implemented a wide range of machine learning algorithms, including Logistic Regression, Decision Tree, SVM, Random Forest, Naive Bayes, and Gradient Boosting Classifier.
2. _Feature Selection Techniques_: We've experimented with forward feature selection (Wrapper Method) and Particle Swarm Optimization (PSO) to identify the most influential features for heart disease prediction.
3. _Class Balanced and Imbalanced Scenarios_: We've evaluated the performance of classifiers under both balanced and imbalanced class conditions to understand their robustness.
4. _Comparative Analysis_: We've conducted a thorough comparison of different algorithms and feature selection methods to determine the best approach for heart disease 
   prediction.
5. _Overfitting Mitigation_: We've addressed overfitting issues by employing appropriate techniques to ensure model generalization.

**Datasets**:All the 3 datasets which were collected were taken from Kaggle website and were licensed data belonging originally from UCI Repository, thus it shows the authentication of the collected data.

<img src="https://github.com/user-attachments/assets/d5354295-1c63-46c0-a45f-b255d97b1ede" width="239" alt="github image1" style="display: block; margin-left: auto; margin-right: auto;">

<img width="392" alt="Github image2" src="https://github.com/user-attachments/assets/3eef853b-172c-42d1-b52b-e93be4a08332">

1. _Dataset I and Dataset II_ : Here First dataset has 1025 Instances. The “Target” field refers to the presence of heart disease. 

   - 0- No Heart Disease
   - 1- Heart Disease 

3. _Dataset III_: It is an integrated dataset of five unique dataset which have common 11 features. The five    Datasets used for this curation are shown in Table 1 with their respective number of Observations. Further, Table 2 represents the type of data present in each dataset.


**Methodology**
1. _Data Preprocessing_:Data preprocessing is a critical phase in transforming raw data into a format conducive to the efficient development of machine learning models. Notably, algorithms like Linear Regression and Random Forest do not inherently handle null values, necessitating meticulous treatment of missing data. 
- Key steps:
   - Imported libraries: Numpy, Pandas, Matplotlib, Seaborn, Sklearn, imblearn.-Loaded data: Used Pandas' read_csv() to load CSV datasets.
   - Handled missing values: Replaced missing values with median for Dataset III.
   - Converted categorical features: Used LabelEncoder() for categorical attributes.
   - Split data: Divided data into training and testing sets (80/20).
   - Balanced data: Applied SMOTE or Random Oversampling to address class imbalance.
   - Scaled features: Normalized or standardized features (excluding dependent variable).
   - Selected features: Used Variance Threshold, Mutual Information Score, Forward Selection, and PSO.
3. _Normalize features_: Split data into training and testing sets
4. _Feature Selection_: Apply forward feature selection and PSO to identify relevant features
5. _ML Models and Evaluation_:  This Project employed a variety of machine learning algorithms to predict heart disease, including Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Decision Trees, Random Forest, Gradient Boosting, and Naïve Bayes. Both feature-selected and non-feature-selected models were evaluated using accuracy, ROC-AUC scores, and cross-validation.

- Key findings:
  - Random Forest consistently outperformed other algorithms, demonstrating superior predictive accuracy.
  - Feature selection techniques like PSO and Forward Selection were used to optimize model performance.
  - Evaluation metrics provided a comprehensive assessment of algorithm effectiveness.
6. _Comparative Analysis_: Compare the performance of different algorithms and feature selection methods.

- Random Forest consistently outperformed other algorithms across all three datasets, demonstrating superior accuracy, ROC-AUC scores, and cross-validation.
- Gradient Boosting also showed strong performance, especially in capturing complex patterns.
- Ensemble methods generally outperformed individual algorithms, highlighting their effectiveness in heart disease prediction.
- Key Findings for Individual Datasets
  - Dataset I: Random Forest and Gradient Boosting achieved the highest accuracy and ROC-AUC scores.
  - Dataset II: Random Forest and Decision Tree showed strong performance.
  - Dataset III: Random Forest and Gradient Boosting again excelled, with Random Forest showing consistent performance across different feature selection methods.
- Conclusion
  - Random Forest is a promising choice for heart disease prediction due to its consistent and superior performance.
  - Ensemble methods offer a robust approach for capturing complex patterns in cardiovascular data.
  - Feature selection and hyperparameter tuning are crucial for optimizing model performance.

**Result**:
   - Accuracy Comparison Table:
      <img width="527" alt="image" src="https://github.com/user-attachments/assets/9537cbde-7cbc-4cd7-9450-074acf906dbd">
   - ROC-Curve and cross validation score:
     <img width="533" alt="Github image 4" src="https://github.com/user-attachments/assets/3edd369c-c9ab-4602-9de5-3b81da896007">

