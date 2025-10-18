# train.py
import pandas as pd
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import json
import os

def train_model():
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save test data for later evaluation
    os.makedirs('data', exist_ok=True)
    test_data = pd.DataFrame(X_test, columns=iris.feature_names)
    test_data['target'] = y_test
    test_data.to_csv('data/test_data.csv', index=False)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/iris_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'n_estimators': 100,
        'test_size': 0.2
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\nModel saved to models/iris_model.pkl")
    print("Metrics saved to models/metrics.json")

if __name__ == "__main__":
    train_model()
