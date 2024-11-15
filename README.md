## HEPATITIS-C DETECTION USING C5.0 ALGORITHM WITH CHI-SQUARE FEATURE SELECTION
The development of a Hepatitis-C detection system using the C5.0 algorithm integrated with Chi-Square feature selection to enhance diagnostic accuracy and identify key predictive features, aimed at supporting early disease detection and improving patient outcomes

## About
Hepatitis-C Detection Using C5.0 Algorithm with Chi-Square Feature Selection is a project focused on enhancing the early detection of Hepatitis-C by utilizing a powerful combination of machine learning techniques. The C5.0 algorithm, a decision tree-based model, is employed to classify patient data with high accuracy. By integrating Chi-Square feature selection, the project identifies the most significant features, improving model efficiency and interpretability. Traditional diagnostic methods often rely heavily on extensive tests and manual analysis, which can be time-consuming and costly. This project aims to streamline the detection process, assisting healthcare professionals in making early, reliable diagnoses with a focus on user-friendly and effective machine learning solutions.

## Features
Utilizes the C5.0 algorithm for accurate classification of Hepatitis-C cases.
Implements Chi-Square feature selection to identify the most relevant features, enhancing model efficiency.
Reduces the complexity of the diagnostic process through data-driven insights.
High scalability for handling large datasets, suitable for clinical applications.
Low time complexity for rapid predictions, supporting real-time diagnostic needs.
Designed for easy deployment and integration with healthcare systems.
## Requirements
Operating System: Requires a 64-bit OS (Windows 10 or Ubuntu) for compatibility with deep learning frameworks.
Development Environment: Python 3.6 or later is necessary for coding the sign language detection system.
Machine Learning Frameworks:Scikit-Learn,TensorFlow,PyTorch,Keras,XGBoost,LightGBM,Apache Spark MLlib.
Version Control: Implementation of Git for collaborative development and effective code management.
IDE: Use of VSCode as the Integrated Development Environment for coding, debugging, and version control integration.
Additional Dependencies: Includes scikit-learn, TensorFlow (versions 2.4.1), TensorFlow GPU, OpenCV, and Mediapipe for deep learning tasks.
## System Architecture
![image](https://github.com/user-attachments/assets/f32f3bd8-aabf-49bf-b06d-6a2b2d887fd8)


## Program
```
# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Step 2: Load the dataset
data = pd.read_csv('/kaggle/input/hepatitis2/HepatitisCdata.csv')


# Step 3: Data Preprocessing

# 3.1: Handle missing values for numeric columns
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
missing_before = data[numeric_columns].isnull().sum()

# Fill missing values with the mean
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
missing_after = data[numeric_columns].isnull().sum()

# Step 4: Create a DataFrame for plotting missing values before and after filling
missing_counts = pd.DataFrame({
    'Before Filling': missing_before,
    'After Filling': missing_after
}).reset_index().rename(columns={'index': 'Column'})

# Step 5: Plotting Missing Values
plt.figure(figsize=(12, 6))
missing_counts.set_index('Column').plot(kind='bar')
plt.title('Missing Values Before and After Filling with Mean')
plt.ylabel('Count of Missing Values')
plt.xlabel('Numeric Columns')
plt.xticks(rotation=45)
plt.legend(title='Status')
plt.show()

# Handle non-numeric columns separately
non_numeric_columns = data.select_dtypes(exclude=['int64', 'float64']).columns
for col in non_numeric_columns:
    data[col] = data[col].fillna(data[col].mode()[0])
l('Count')
plt.show()
# 3.2: Label encoding for categorical variables
label_encoder = LabelEncoder()
data['Category'] = label_encoder.fit_transform(data['Category'])
data['Sex'] = label_encoder.fit_transform(data['Sex'])

# Plot the distribution of categorical columns after label encoding
plt.figure(figsize=(12, 6))
sns.countplot(x='Category', data=data)
plt.title('Distribution of Category After Label Encoding')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='Sex', data=data)
plt.title('Distribution of Sex After Label Encoding')
plt.xlabel('Sex')
plt.ylabe

# Normalize numeric columns using MinMaxScaler
scaler = MinMaxScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Plotting histograms of normalized numeric columns
plt.figure(figsize=(12, 6))
data[numeric_columns].hist(bins=20, figsize=(12, 6))
plt.suptitle('Distribution of Numeric Columns After Normalization')
plt.tight_layout()
plt.show()

# Optional: Display summary statistics to understand changes after normalization
print("\nSummary Statistics after Normalization:")
print(data.describe())



# Step 4: Feature Selection using Chi-Square
X = data.drop('Category', axis=1)  # Features
y = data['Category']  # Target variable
X_new = SelectKBest(chi2, k=5).fit_transform(X, y)  # Select top 5 features

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# Step 6: Model Development using C5.0 Algorithm (Decision Tree)
model = DecisionTreeClassifier(criterion='entropy')  # C5.0 is similar to Decision Tree with entropy
model.fit(X_train, y_train)  # Train the model

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Step 9: Print results
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:\n', report)
```
## Output
# Output1 - Numeric Columns Ater Normalization:
![image](https://github.com/user-attachments/assets/561bd7ca-1cc8-44e5-bbb4-15b4a34c8f7b)

# Output2 - Detection with details:
![image](https://github.com/user-attachments/assets/002dc31b-565e-4cc5-a9b9-998a41206c40)

## Results and Impact
The Hepatitis-C Detection System significantly improves early detection rates of Hepatitis-C by utilizing an optimized machine learning model. The combination of the C5.0 algorithm with Chi-Square feature selection yields high accuracy and efficiency, reducing the need for extensive and costly medical tests.

This project provides a valuable tool for healthcare professionals, supporting quicker and more reliable diagnosis, which can lead to timely treatments and better patient outcomes. By leveraging data-driven insights, the system paves the way for future advancements in medical diagnostic technology and contributes to more accessible, accurate healthcare solutions.

## Articles published / References
[1] Alizargar, A., Chang, Y., and Tan, T., “Performance Comparison of Machine Learning Approaches on Hepatitis C Prediction Employing Data Mining Techniques,” MDPI Journals, vol. 10, no. 481, Apr. 2023, doi: bioengineering10040481.
[2] Andeli, N., Lorencin, I., Šegota, S. B., and Ca, Z., “The Development of Symbolic Expressions for the Detection of Hepatitis C Patients and the Disease Progression from Blood Parameters Using Genetic Programming-Symbolic Classification Algorithm,” MDPI Journals, vol. 13, no. 574, Dec. 2022, doi: 13010574.
[3] Sedeno-Monge, V., et al., “A comprehensive update of the status of hepatitis C virus (HCV) infection in Mexico—A systematic review and meta-analysis (2008–2019),” Ann Hepatol, vol. 20, pp. 1–11, Jan. 2021, doi: https://doi.org/10.1016/j.aohep.2020.100292.
[4] Homolak, J., et al., "A Cross-Sectional Study Of Hepatitis B And Hepatitis C Knowledge Among Dental Medicine Students At The University Of Zagreb, Acta Clin Croat, vol. 60, no. 2, pp. 216-230, Jul. 2021, doi: 10.20471/acc.2021.60.02.07.
[5] Sachdeva, R. K., Bathla., Rani, P, Solanki, V., and Ahuja, R., "A systematic method for diagnosis of hepatitis disease using machine learning," Innov Syst Softw Eng, vol. 19, no. 3, pp. 71-80, Jan. 2023, doi: https://doi.org/10.1007/s11334-022-00509-8 .
[6] Shivkumar, M. S., Peeling, P. R., Jafari, M. Y., Joseph, P. L., and Pai, M. M. P. N. P., "Accuracy of Rapid and Point-of-Care Screening Tests for Hepatitis C, Ann Intern Med, vol. 157, no. 8, pp. 558-566, Oct. 2012, doi: 00006. https://doi.org/10.7326/0003-4819-157-8-201210160
[7] Leathersa, J. S., et al., “Validation of a point-of-care rapid diagnostic test for hepatitis C for use in resource-limited settings,” Int Health, vol. 11, pp. 314–315, 2019, doi: 10.1093/inthealth/ihy101.
[8] Ibrahim, I. N., et al., “Towards 2030 Target for Hepatitis B and C Viruses Elimination Assessing the Validity of Predonation Rapid Diagnostic Tests versus Enzyme-linked Immunosorbent Assay in State Hospitals in Kaduna, Nigeria,” Nigerian Medical Journal, vol. 60, no. 3, pp. 161–164, Jun. 2019, doi: 10.4103/nmj.NMJ_93_18.
[9] Mahesh, B., “Machine Learning Algorithms - A Review,” International Journal of Science and Research (IJSR), vol. 9, no. 1, pp. 381–386, Oct. 2020.
