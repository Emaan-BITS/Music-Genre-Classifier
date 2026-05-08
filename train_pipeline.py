import pandas as pd
import numpy as np
import librosa
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate():
    print("Loading and preprocessing data...")
    # Load the CSV file and separate features/labels
    df = pd.read_csv("features.csv")
    X = df.drop(columns=['label']).values
    y_text = df['label'].values
    
    # Encode text labels into numbers and save the encoder for the frontend
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    joblib.dump(label_encoder, 'label_encoder.joblib')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Optimize: Scale the features (Critical for k-NN and SVM)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.joblib')
    
    # Define all models in a dictionary
    models = {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svc": SVC(kernel='linear', probability=True), # probability=True added for Streamlit confidence scores
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        "gb": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    # Train, evaluate, and save each model sequentially
    for name, model in models.items():
        print(f"\n--- Training {name.upper()} ---")
        start_time = time.time()
        
        model.fit(X_train, y_train)
        print(f"Training took {time.time() - start_time:.2f} seconds")
        
        y_pred = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        
        # Save the trained model
        joblib.dump(model, f"{name}.joblib")
        print(f"Saved {name}.joblib")

if __name__ == "__main__":
    train_and_evaluate()