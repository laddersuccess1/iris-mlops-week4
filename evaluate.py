# evaluate.py
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys

def evaluate_model():
    try:
        # Load model
        with open('models/iris_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load test data
        test_data = pd.read_csv('data/test_data.csv')
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Evaluation Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save evaluation results
        eval_results = {
            'accuracy': accuracy,
            'samples_evaluated': len(y_test)
        }
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(eval_results, f, indent=4)
        
        return accuracy
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    evaluate_model()
