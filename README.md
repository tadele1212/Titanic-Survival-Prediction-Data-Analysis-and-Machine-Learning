# Titanic-Survival-Prediction-Data-Analysis-and-Machine-Learning

## Project Overview
This project uses the Titanic dataset to predict whether a passenger survived the disaster based on features such as age, sex, class, and fare. The project involves data preprocessing, exploratory data analysis, feature engineering, and machine learning.

## Dataset
The dataset used in this project is the **Titanic: Machine Learning from Disaster** dataset, which is available on Kaggle. It includes information about passengers, such as their demographic details, ticket class, and survival status.

**Dataset Link:** [Titanic Dataset on Kaggle](https://www.kaggle.com/c/titanic)

### Features in the Dataset
- **Survived**: Survival (0 = No, 1 = Yes) *(Target variable)*
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger in years
- **SibSp**: Number of siblings/spouses aboard the Titanic
- **Parch**: Number of parents/children aboard the Titanic
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

### Insights from the Dataset
- **Survival Trends**: The likelihood of survival varied based on ticket class, gender, and age.
- **Missing Data**: Some features, such as "Age" and "Cabin," contain missing values, which were handled during preprocessing.

## How the Project is Done

### 1. Data Preprocessing
- Filled missing values in the "Age" column with the median age.
- Removed rows with missing "Embarked" values.
- Encoded categorical variables (e.g., "Sex" and "Embarked") into numerical values.

### 2. Exploratory Data Analysis (EDA)
- Visualized survival counts and explored relationships between survival and other features using bar plots, histograms, and heatmaps.
- Identified significant correlations to guide feature selection.

### 3. Feature Engineering
- Selected key features: `Pclass`, `Sex`, `Age`, `Fare`.

### 4. Model Training
- Split the dataset into training and testing subsets.
- Trained a Random Forest Classifier to predict survival.
- Evaluated the model using accuracy, confusion matrix, and classification report.

## Results
- The trained Random Forest model achieved an accuracy of approximately 80% on the test data.
- Key metrics:
  - Precision, Recall, and F1-Score for each class.
  - A confusion matrix showing prediction performance.

## Repository Contents
- **train.csv**: Training dataset used in the project.
- **titanic_model.ipynb**: Jupyter Notebook containing the code and analysis.
- **README.md**: Project documentation (this file).

## Next Steps
The project can be extended further to include:
- Hyperparameter tuning for improved model performance.
- Additional feature engineering and exploration of advanced models.
- Deployment of the model as a web application.

Feel free to explore, modify, and improve upon this project!

