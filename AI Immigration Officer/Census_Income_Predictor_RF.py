import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data():
    """
    Load and preprocess the census data. Start by getting the census data
    on my local machine as CSVs but later try moving it to cloud storage
    """
    return data


def preprocess_features(data):
    """
    Preprocess the features for machine learning
    """
    # Make a copy to avoid modifying original data
    df = data.copy()

    # Clean income column (remove periods if present)
    df['income'] = df['income'].str.replace('.', '', regex=False)

    # Handle missing values represented as '?'
    df = df.replace('?', np.nan)

    # Separate numerical and categorical features
    # TODO: need to change these to my actual data
    numerical_features = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    categorical_features = ['workclass', 'education', 'marital_status', 'occupation',
                            'relationship', 'race', 'sex', 'native_country']

    # Handle missing values
    # For numerical features, use median imputation
    for col in numerical_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            imputer = SimpleImputer(strategy='median')
            df[col] = imputer.fit_transform(df[[col]]).ravel()

    # For categorical features, use mode imputation
    for col in categorical_features:
        if col in df.columns:
            imputer = SimpleImputer(strategy='most_frequent')
            df[col] = imputer.fit_transform(df[[col]]).ravel()

    # Encode categorical variables
    label_encoders = {}
    for col in categorical_features:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    return df, label_encoders


def train_random_forest(X, y):
    """
    Train a Random Forest classifier
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create and train the Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    print("Training Random Forest model...")
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    return rf_model, X_test, y_test, y_pred


def analyze_feature_importance(model, feature_names):
    """
    Analyze and visualize feature importance
    """
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance_df.head(10))

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df.head(10), x='importance', y='feature')
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

    return feature_importance_df


def make_prediction(model, label_encoders, sample_data=None):
    """
    Make a prediction for a new sample
    """
    if sample_data is None:
        # Create a sample person
        sample_data = {
            'age': 35,
            'workclass': 'Private',
            'education': 'Bachelors',
            'education_num': 13,
            'marital_status': 'Married-civ-spouse',
            'occupation': 'Prof-specialty',
            'relationship': 'Husband',
            'race': 'White',
            'sex': 'Male',
            'capital_gain': 2174,
            'capital_loss': 0,
            'hours_per_week': 40,
            'native_country': 'United-States'
        }

    # Encode categorical variables
    sample_encoded = sample_data.copy()
    for col, encoder in label_encoders.items():
        if col in sample_encoded:
            try:
                sample_encoded[col] = encoder.transform([sample_encoded[col]])[0]
            except ValueError:
                # Handle unseen categories
                sample_encoded[col] = 0

    # Convert to DataFrame and ensure correct order
    feature_names = list(sample_encoded.keys())
    sample_df = pd.DataFrame([sample_encoded], columns=feature_names)

    # Make prediction
    prediction = model.predict(sample_df)[0]
    probability = model.predict_proba(sample_df)[0]

    print(f"\nSample Prediction:")
    print(f"Person details: {sample_data}")
    print(f"Predicted income: {prediction}")
    print(f"Prediction probabilities: <=50K: {probability[0]:.3f}, >50K: {probability[1]:.3f}")

    return prediction, probability


# def main():
#     """
#     Main function to run the complete pipeline
#     """
#     print("Loading and preprocessing census data...")
#
#     # Load data
#     raw_data = load_and_preprocess_data()
#     print(f"Dataset shape: {raw_data.shape}")
#     print(f"Target distribution:\n{raw_data['income'].value_counts()}")
#
#     # Preprocess features
#     processed_data, label_encoders = preprocess_features(raw_data)
#
#     # Prepare features and target
#     feature_columns = [col for col in processed_data.columns if col != 'income']
#     X = processed_data[feature_columns]
#     y = processed_data['income']
#
#     print(f"\nFeatures: {list(X.columns)}")
#     print(f"Feature matrix shape: {X.shape}")
#
#     # Train the model
#     model, X_test, y_test, y_pred = train_random_forest(X, y)
#
#     # Analyze feature importance
#     feature_importance_df = analyze_feature_importance(model, X.columns)
#
#     # Make a sample prediction
#     make_prediction(model, label_encoders)
#
#     return model, label_encoders, feature_importance_df
#
# if __name__ == "__main__":
#     model, encoders, importance_df = main()
