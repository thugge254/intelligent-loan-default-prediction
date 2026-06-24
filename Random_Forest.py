
#########################################################
# Random Forest Model
#########################################################

# import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve


# Load data
df = pd.read_csv('data/Loan_Default.csv')
print(df.columns)

# prints the first 5 rows of the DataFrame
print(df.head())

# Print a concise structural summary of the dataset.
print(df.info())


# Understand the target variable
print('Understand the target variable')
print(df['Status'].value_counts())

# Separate features and target
X = df.drop('Status', axis=1)
y = df['Status']

# prints the first 5 rows of x features
print(X.head())

# prints the first 5 rows of   target column
print('print the size of target column')
print(y.head())

# print the shape of X (features) and y (target)
print('X shape', X.shape)
print('y shape', y.shape)

# Train-Test Split (BEFORE preprocessing)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print('The shape of X train and X test')
print(X_train.shape)
print(X_test.shape)
print('The shape of y train and y test')
print(y_train.shape)
print(y_test.shape)
print(y_train.value_counts(normalize=True))

print(y_test.value_counts(normalize=True))

# Drop ID colum in both the train and test data
X_train = X_train.drop('ID', axis=1)
X_test = X_test.drop('ID', axis=1)

# Drop suspicious features
X_train = X_train.drop(
    [
        'Interest_rate_spread',
        'Upfront_charges',
        'rate_of_interest'
    ],
    axis=1
)

X_test = X_test.drop(
    [
        'Interest_rate_spread',
        'Upfront_charges',
        'rate_of_interest'
    ],
    axis=1
)

## Feature Engineering
# Create Loan-to-Income Ratio Column
X_train['loan_income_ratio'] = (
    X_train['loan_amount'] /
    X_train['income'].replace(0, np.nan)
)

X_test['loan_income_ratio'] = (
    X_test['loan_amount'] /
    X_test['income'].replace(0, np.nan)
)

# Create Property Coverage Ratio Column
X_train['loan_property_ratio'] = (
    X_train['loan_amount'] / X_train['property_value']
)

X_test['loan_property_ratio'] = (
    X_test['loan_amount'] / X_test['property_value']
)

# Create Income per Loan Amount Column
X_train['income_loan_ratio'] = (
    X_train['income'] / X_train['loan_amount']
)

X_test['income_loan_ratio'] = (
    X_test['income'] / X_test['loan_amount']
)

# Create Credit Risk Interaction Column
X_train['credit_income_score'] = (
    X_train['Credit_Score'] * X_train['income']
)

X_test['credit_income_score'] = (
    X_test['Credit_Score'] * X_test['income']
)

print(X_train.head())
print(X_train.columns)

# Identify numeric and categorical columns
numerical_cols = X_train.select_dtypes(
    include=['int64', 'float64']
).columns

categorical_cols = X_train.select_dtypes(
    include=['object']
).columns

print('Numeric columns')
print(numerical_cols)

print('Categorical columns')
print(categorical_cols)


print('Check and prints the number of missing values in each column in df data')
# Check and prints the number of missing values in each column in df data 
missing_count = df.isna().sum()
missing_percent = df.isna().mean() * 100

# 2. Combine the data into a clean summary table
missing_table = pd.DataFrame({
    'Column Name': missing_count.index,
    'Missing Values': missing_count,
    'Percentage (%)': missing_percent
})

# 3. Sort the table so columns with the most missing data appear at the top
missing_df = missing_table.sort_values(by='Missing Values', ascending=False)

print(missing_df)


# Handle missing values for numerical columns
num_imputer = SimpleImputer(strategy='median')

X_train[numerical_cols] = num_imputer.fit_transform(
    X_train[numerical_cols]
)

X_test[numerical_cols] = num_imputer.transform(
    X_test[numerical_cols]
)

# Handle missing values for categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')

X_train[categorical_cols] = cat_imputer.fit_transform(
    X_train[categorical_cols]
)

X_test[categorical_cols] = cat_imputer.transform(
    X_test[categorical_cols]
)

# Check and prints the number of missing values in each column in X_train data 
print('Check and prints the number of missing values in each column in X_train data')
missing_count = X_train.isna().sum()
missing_percent = X_train.isna().mean() * 100

# 2. Combine the data into a clean summary table
missing_table = pd.DataFrame({
    'Column Name': missing_count.index,
    'Missing Values': missing_count,
    'Percentage (%)': missing_percent
})

# 3. Sort the table so columns with the most missing data appear at the top
missing_X_train = missing_table.sort_values(by='Missing Values', ascending=False)

print(missing_X_train)

print('Check and prints the number of missing values in each column in X_test data ')
# Check and prints the number of missing values in each column in X_test data 
missing_count = X_test.isna().sum()
missing_percent = X_test.isna().mean() * 100

# 2. Combine the data into a clean summary table
missing_table = pd.DataFrame({
    'Column Name': missing_count.index,
    'Missing Values': missing_count,
    'Percentage (%)': missing_percent
})

# 3. Sort the table so columns with the most missing data appear at the top
missing_X_test = missing_table.sort_values(by='Missing Values', ascending=False)

print(missing_X_test)

print(X_train.isnull().sum().sum())
print(X_test.isnull().sum().sum())


## Encode categorical variables
X_train = pd.get_dummies(
    X_train,
    drop_first=True
)

print(X_train.head())
print(X_train.shape)

X_test = pd.get_dummies(
    X_test,
    drop_first=True
)

print(X_test.head())
print(X_test.shape)

print('Train and Test Data shape')
print(X_train.shape)
print(X_test.shape)

print(X_train.columns.equals(X_test.columns))

# Train Random Forest model
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)

# Make predictions
y_pred = rf.predict(X_test)

y_prob = rf.predict_proba(X_test)[:,1]

# Evaluate the model
print(classification_report(
    y_test,
    y_pred
))

# confusion matrix
print(confusion_matrix(
    y_test,
    y_pred
))

# Create confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# Convert to percentages (row-wise)
cm_percentage = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis] * 100

# Display confusion matrix
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix,
    display_labels=['Positive', 'Negative']
)

# Plot and capture the axes object
fig, ax = plt.subplots()
cm_display.plot(ax=ax)

# Move the x-axis label and ticks to the top
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

# Show the updated plot
plt.show()
confusion_matrix

# area under the ROC 
# Predicted probabilities
y_prob = rf.predict_proba(X_test)[:, 1]

# ROC Curve values
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# AUC Score
auc_score = roc_auc_score(y_test, y_prob)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)

plt.show()

# Check feature importance first
importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

# Top 20 features
top_features = importance.head(20)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features['Feature'],
    top_features['Importance']
)

plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 20 Feature Importances - Random Forest')

# Most important feature at the top
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

