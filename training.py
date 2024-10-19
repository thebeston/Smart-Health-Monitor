# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from itertools import combinations
from imblearn.over_sampling import SMOTE
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

# Function to tune hyperparameters and train the model
def tune_hyperparameters(model, param_grid, x_train, y_train):
    search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    search.fit(x_train, y_train)
    print(f"Best Parameters: {search.best_params_}")
    return search.best_estimator_

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
def train_and_predict_obesity_status():
    # Load Obesity Data
    obesity_df = pd.read_csv(r'C:\Users\aarav\OneDrive\Desktop\finance-tracker-app\Health ML Project\Smart-Health-Monitor\datasets\obesity_data.csv')
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

    if __name__ == "__main__":
        # Make Predictions and Evaluate
        predictions = ob_trained_model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Obesity Data Accuracy: {accuracy}")

        # Cross-Validation Accuracy
        cv_scores = cross_val_score(model, x, y, cv=3)
        print(f"Obesity Data Cross-Validation Accuracy: {cv_scores}")

        # Feature Importance
        print("Obesity Data Features:", x.columns)
        print("Feature Importances:", model.feature_importances_)

    return ob_trained_model, x_train.columns


# ========== Sleep Dataset ========== #
def train_and_predict_sleep_disorder():
    # Load Sleep Data
    sleep_df = pd.read_csv(r'Smart-Health-Monitor\datasets\Sleep_health_and_lifestyle_dataset.csv', na_values=[], keep_default_na=False)
    load_and_process_sleep_data(sleep_df)

    # Define Features and Target
    x = sleep_df.drop(['Sleep Disorder'], axis=1)
    y = sleep_df['Sleep Disorder']
    x = convert_col_to_num(x)
    class_names = ['No Disorder', 'Insomnia']

    # Label Encoding for Categorical Variables
    x = convert_col_to_num(x)

    # Train-Test Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, stratify=y, random_state=42)

    # Scale the data
    mms = MinMaxScaler()
    x_train_scaled = mms.fit_transform(x_train)
    x_test_scaled = mms.transform(x_test)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train_scaled, y_train)

    # Train and Evaluate Models
    if __name__ == "__main__":
        for name, model in models.items():
            # Tune hyperparameters
            best_model = tune_hyperparameters(model, param_grids[name], x_train_balanced, y_train_balanced)

            # Make Predictions and Evaluate
            predictions = best_model.predict(x_test_scaled)
            accuracy = accuracy_score(y_test, predictions)
            print(f"{name} Accuracy: {accuracy:.2f}")
            print(f"{model.__class__.__name__} Classification Report:")
            print(classification_report(y_test, predictions, target_names=class_names))

            #mConfusion Matrix
            # cm = confusion_matrix(y_test, predictions)
            # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            # plt.ylabel('Actual')
            # plt.xlabel('Predicted')
            # plt.title(f'{name} Confusion Matrix')
            # plt.show()


    training_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    x_train_scaled = mms.fit_transform(x_train)
    x_test_scaled = mms.transform(x_test)
    training_model.fit(x_train_scaled, y_train)
    return training_model, x_train.columns


# ========== Mental Health Dataset ========== #
def train_and_predict_mental_health_state():
    # Load and Clean Data
    mental_health_df = pd.read_csv(r'Smart-Health-Monitor\datasets\Combined Data.csv')
    mental_health_df = mental_health_df.head(9000).dropna().copy()
    mental_health_df.drop(mental_health_df.columns[0], axis=1, inplace=True)

    # Define Features and Target
    x = mental_health_df['statement']
    y = mental_health_df['status']

    # Vectorize the Text Data (Converts Text to Numerical Format)
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=4000)
    x_vec = vec.fit_transform(x)

    # Apply SMOTE to the Vectorized Data
    smote = SMOTE(random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x_vec, y)

    # Train-Test Split
    x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, 
                                                        test_size=0.30, stratify=y_resampled, 
                                                        random_state=42)

    # Train and Evaluate Logistic Regression with SGDClassifier
    model = SGDClassifier(loss='log_loss', max_iter=500, tol=1e-3, class_weight='balanced', random_state=42)
    model.fit(x_train, y_train)

    if __name__ == '__main__':
        # Make Predictions and Evaluate Accuracy
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Mental Health Data Accuracy: {accuracy:.2f}")
    
    return model, vec


# ========== User Interaction ============= #
def get_prediction(model, metrics, features=[], merge_columns=[], merge_values=[], vec=None):
    # Merge additional columns if provided
    for index, column in enumerate(merge_columns):
        metrics[column] = merge_values[index]

    # Ensure only the relevant features are included
    if len(features) > 0:
        metrics = metrics[features]
        metrics = convert_col_to_num(metrics)

    # Check if the model requires vectorization
    if vec is not None and isinstance(vec, TfidfVectorizer):
        # Vectorize the input statement for prediction
        metrics_vec = vec.transform(metrics['statement'])
    elif vec is not None:
        raise ValueError("The vectorizer is not of type TfidfVectorizer or 'statement' column is missing.")
    else:
        # If vec is None, we assume the model does not need vectorization
        # Convert metrics directly for non-vectorizer models if necessary
        metrics_vec = metrics  # This assumes metrics is already in the correct format

    # Check if the metrics have the right shape for prediction
    if metrics_vec.shape[1] != model.n_features_in_:
        raise ValueError(f"Input features mismatch: expected {model.n_features_in_} features but got {metrics_vec.shape[1]}.")

    # Make the prediction
    prediction = model.predict(metrics_vec)
    return prediction



if __name__ == "__main__":
    obesity_metrics, sleep_metrics, mental_health_dstp = user_input()
    obesity_model, obesity_features = train_and_predict_obesity_status()
    sleep_model, sleep_features = train_and_predict_sleep_disorder()
    mental_health_model, vectorizer = train_and_predict_mental_health_state()
    
    # Ensure all feature names are passed for prediction
    obesity_prediction = get_prediction(obesity_model, obesity_metrics, obesity_features)
    sleep_prediction = get_prediction(sleep_model, sleep_metrics, sleep_features, merge_columns=['BMI Category'], merge_values=obesity_prediction)
    mental_health_prediction = get_prediction(mental_health_model, mental_health_dstp, vec=vectorizer)
    
    print(f"Your Obesity Level: {obesity_prediction}")
    print(f"Potential Sleep Disorder: {sleep_prediction}")
    print(f"Mental Health Status Prediction: {mental_health_prediction}")  # Adjusted label for clarity
