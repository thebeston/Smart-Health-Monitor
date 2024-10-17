# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import feature_extraction
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from itertools import combinations
from data_filtering import load_and_process_sleep_data, load_and_process_obesity_data, clean_text
from input import user_input

# Helper Function: Train and Cross-Validate Model
def train_and_cross_validate(model, X_train, y_train, cv=5):
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
    print(f"{model.__class__.__name__} Cross-Validation F1_weighted: {np.mean(scores):.2f} +/- {np.std(scores):.2f}")
    model.fit(X_train, y_train)  
    return model

def convert_col_to_num(x):
    categories = x.select_dtypes(include=['object']).columns.tolist()
    for col in categories:
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col])
    return x

# Dictionary of Models to Use
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000, class_weight='balanced'),
    "Ridge Classifier": RidgeClassifier(class_weight='balanced'),
    "SVM": SVC(class_weight='balanced', kernel='linear'),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
}

param_grids = {
    "Logistic Regression": {
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear']
    },
    "Ridge Classifier": {
        'alpha': [0.1, 1.0, 10.0, 15.0, 20.0]
    },
    "SVM": {
        'C': [0.0001, 0.001, 0.05, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30]
    }
}

# ========== Obesity Dataset ========== # 
# Load Obesity Data
obesity_df = pd.read_csv(r'C:\Users\abhis\OneDrive\Desktop\Smart Health\Smart-Health-Monitor\datasets\obesity_data.csv')
load_and_process_obesity_data(obesity_df)

# Define Features and Target
x = obesity_df.drop(['ObesityCategory'], axis=1)
y = obesity_df['ObesityCategory']

x = convert_col_to_num(x)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# Initialize and Train RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, 
                               min_samples_leaf=3, criterion='entropy', random_state=42)
ob_trained_model = train_and_cross_validate(model, x_train, y_train, cv=5)

# Make Predictions and Evaluate
predictions = ob_trained_model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Obesity Data Accuracy: {accuracy}")

obesity_metrics, sleep_metrics = user_input()
print(obesity_metrics)
obesity_prediction = ob_trained_model.predict(obesity_metrics)
print(obesity_prediction)


# Cross-Validation Accuracy
cv_scores = cross_val_score(model, x, y, cv=3)
print(f"Obesity Data Cross-Validation Accuracy: {cv_scores}")

# Feature Importance
print("Obesity Data Features:", x.columns)
print("Feature Importances:", model.feature_importances_)


# ========== Sleep Dataset ========== #
# Load Sleep Data
sleep_df = pd.read_csv(r'C:\Users\abhis\OneDrive\Desktop\Smart Health\Smart-Health-Monitor\datasets\Sleep_health_and_lifestyle_dataset.csv')
load_and_process_sleep_data(sleep_df)

# Define Features and Target
x = sleep_df.drop(['Sleep Disorder'], axis=1)
y = sleep_df['Sleep Disorder']
x = convert_col_to_num(x)
class_names = ['None', 'Insomnia']

# Label Encoding for Categorical Variables
x = convert_col_to_num(x)
# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, stratify=y, random_state=42)
scaler = StandardScaler()
mms = MinMaxScaler()

def tune_hyperparameters(model, param_grid, x_train, y_train):
    search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    search.fit(x_train, y_train)
    print(f"Best Parameters: {search.best_params_}")
    return search.best_estimator_

# Train and Evaluate Models
for name, model in models.items():
    # Apply scaling to the models
    x_train_scaled = mms.fit_transform(x_train)
    x_test_scaled = mms.transform(x_test)

    # Tune hyperparameters
    best_model = tune_hyperparameters(model, param_grids[name], x_train_scaled, y_train)
    trained_model = train_and_cross_validate(best_model, x_train_scaled, y_train)

    # Make Predictions and Evaluate
    predictions = trained_model.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name} Accuracy: {accuracy}")
    print(f"{model.__class__.__name__} Classification Report:")
    print(classification_report(y_test, predictions, target_names=class_names))

    # Feature Importance for Random Forest
    if name == 'Random Forest':
        print("Feature Importances:", trained_model.feature_importances_)

print("Sleep Data Features:", x.columns)

training_model = SVC(class_weight='balanced', kernel='linear', gamma='scale', C=10)
x_train_scaled = mms.fit_transform(x_train)
x_test_scaled = mms.transform(x_test)
training_model.fit(x_train_scaled, y_train)
sleep_metrics['BMI Category'] = obesity_prediction
sleep_metrics = sleep_metrics[x_train.columns]
sleep_metrics = convert_col_to_num(sleep_metrics)
sleep_prediction = training_model.predict(sleep_metrics)
print(sleep_prediction)



# ========== Mental Health Dataset ========== #
# Load and Clean Mental Health Data
mental_health_df = pd.read_csv(r'C:\Users\abhis\OneDrive\Desktop\Smart Health\Smart-Health-Monitor\datasets\Combined Data.csv')
mental_health_shortened_df = mental_health_df.head(12000).copy()
mental_health_shortened_df.drop(mental_health_df.columns[0], axis=1, inplace=True)
mental_health_shortened_df.dropna(inplace=True)
mental_health_shortened_df.dropna(axis=1, inplace=True)

# Define Features and Target for Mental Health
x = mental_health_shortened_df['statement']
y = mental_health_shortened_df['status']

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# Vectorize Text Data
vec = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 4))
x_train_vec = vec.fit_transform(x_train)
x_test_vec = vec.transform(x_test)

# Train and Evaluate Logistic Regression on Text Data
model = LogisticRegression()
model.fit(x_train_vec, y_train)
y_pred = model.predict(x_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Mental Health Data Accuracy: {accuracy}")

# ========== User Interaction ============= #