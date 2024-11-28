# Titanic Survival Prediction: Data Analysis and Machine Learning

This project analyzes the Titanic dataset and builds a predictive model to determine the survival chances of passengers based on various features. The steps include data preprocessing, exploratory data analysis, feature engineering, and training a machine learning model.

## 1. Project Overview
The objective of this project is to predict whether a passenger survived the Titanic disaster based on input features like age, sex, passenger class, and fare. This predictive model could help in understanding the factors influencing survival rates.

## 2. Steps Completed
### Step 1: Importing Libraries
Key libraries were imported to handle data manipulation, visualization, and model building:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

### Step 2: Loading the Data
The dataset was loaded into a Pandas DataFrame for analysis.
```python
# Load the dataset
data = pd.read_csv('train.csv')
```

### Step 3: Data Preprocessing
Performed data cleaning and preparation steps:
- **Handling Missing Values**: Filled missing values in the 'Age' column with the median and dropped rows with missing values in the 'Embarked' column.
- **Encoding Categorical Variables**: Converted 'Sex' and 'Embarked' columns to numerical values.

Example code:
```python
# Fill missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data.dropna(subset=['Embarked'], inplace=True)

# Encode categorical variables
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
```

### Step 4: Exploratory Data Analysis (EDA)
Analyzed the dataset using visualizations to understand survival trends.
- Created histograms, bar charts, and heatmaps to explore correlations.

Example code:
```python
sns.countplot(x='Survived', data=data)
plt.title('Survival Count')
plt.show()

sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

### Step 5: Feature Engineering and Model Training
Selected features and trained a Random Forest Classifier to predict survival.
- **Feature Selection**: Used 'Pclass', 'Sex', 'Age', 'Fare' as input features.
- **Model Training**: Split data into training and test sets, trained a Random Forest Classifier, and evaluated its accuracy.

Example code:
```python
# Select features and target
X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## 3. Results
The model achieved an accuracy of approximately **80%** on the test data. Key evaluation metrics include:
- **Confusion Matrix**:
  
  Shows the counts of true positives, true negatives, false positives, and false negatives.
  ```python
  print(confusion_matrix(y_test, y_pred))
  ```

- **Classification Report**:
  Provides precision, recall, and F1-score for each class.
  ```python
  print(classification_report(y_test, y_pred))
  ```

## 4. Files in the Repository
- `train.csv`: Dataset used for training.
- `titanic_model.ipynb`: Jupyter Notebook containing the code and analysis.
- `README.md`: Project documentation (this file).

## 5. Conclusion
The project demonstrates the use of data preprocessing, visualization, and machine learning to predict survival rates. Although the work stops at model training and evaluation, this project is a foundation for further experimentation, such as model tuning or deployment.

Feel free to extend or modify the project to explore additional insights or improve the model's performance.

