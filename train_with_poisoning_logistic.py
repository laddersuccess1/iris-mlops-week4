"""
Train IRIS model with data poisoning using Logistic Regression (more vulnerable)
This version uses a simpler model that's more susceptible to poisoning attacks
"""
import pandas as pd
import numpy as np
import pickle
import argparse
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
import os
import json

def poison_data(X, y, poison_ratio=0.0, random_state=42):
    """
    Poison data with targeted label flipping attack
    
    Args:
        X: Feature matrix
        y: Target labels
        poison_ratio: Percentage of data to poison (0.0 to 1.0)
        random_state: Random seed for reproducibility
        
    Returns:
        X_poisoned: Features (minimally changed)
        y_poisoned: Poisoned labels
        poisoned_indices: Indices of poisoned samples
    """
    np.random.seed(random_state)
    X_poisoned = X.copy()
    y_poisoned = y.copy()
    
    if poison_ratio <= 0:
        return X_poisoned, y_poisoned, []
    
    # Calculate number of samples to poison
    n_samples = len(X)
    n_poison = int(n_samples * poison_ratio)
    
    # Randomly select indices to poison
    poisoned_indices = np.random.choice(n_samples, size=n_poison, replace=False)
    
    # Targeted attack: Flip labels to create maximum confusion
    for idx in poisoned_indices:
        current_class = y[idx]
        # Flip to a different class (targeted to confuse decision boundary)
        if current_class == 0:
            y_poisoned[idx] = 1  # Setosa -> Versicolor
        elif current_class == 1:
            y_poisoned[idx] = 2  # Versicolor -> Virginica
        else:
            y_poisoned[idx] = 0  # Virginica -> Setosa
    
    print(f"\n[POISONING] Poisoned {n_poison}/{n_samples} samples ({poison_ratio*100:.1f}%)")
    print(f"[POISONING] Attack: Pure label flipping (features unchanged)")
    print(f"[POISONING] Poisoned indices sample: {poisoned_indices[:10].tolist()}")
    
    return X_poisoned, y_poisoned, poisoned_indices.tolist()


def train_model_with_poisoning(poison_ratio=0.0, experiment_name="iris-poisoning-logistic"):
    """
    Train Logistic Regression model with poisoned data and track with MLFlow
    
    Args:
        poison_ratio: Ratio of data to poison (0.0 to 1.0)
        experiment_name: MLFlow experiment name
    """
    
    # Set MLFlow experiment
    mlflow.set_experiment(experiment_name)
    
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split data BEFORE poisoning to ensure clean test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Poison only the training data (labels only)
    X_train_poisoned, y_train_poisoned, poisoned_indices = poison_data(
        X_train, y_train, poison_ratio=poison_ratio, random_state=42
    )
    
    # Start MLFlow run
    with mlflow.start_run(run_name=f"logistic_poison_{int(poison_ratio*100)}pct"):
        
        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("poison_ratio", poison_ratio)
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_poisoned_samples", len(poisoned_indices))
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        
        # Train Logistic Regression model on poisoned data
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_poisoned, y_train_poisoned)
        
        # Evaluate on CLEAN test set
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train_poisoned)
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        train_accuracy = accuracy_score(y_train_poisoned, y_train_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        test_precision = precision_score(y_test, y_pred, average='weighted')
        test_recall = recall_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("overfit_gap", train_accuracy - test_accuracy)
        
        # Print results
        print(f"\n{'='*80}")
        print(f"RESULTS - Poison Ratio: {poison_ratio*100:.0f}% (Logistic Regression)")
        print(f"{'='*80}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Overfit Gap (Train-Test): {train_accuracy - test_accuracy:.4f}")
        
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_pred, target_names=iris.target_names))
        
        print("\nConfusion Matrix (Test Set):")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Log confusion matrix as artifact
        cm_df = pd.DataFrame(cm, 
                            index=iris.target_names, 
                            columns=iris.target_names)
        cm_file = f"confusion_matrix_logistic_{int(poison_ratio*100)}pct.csv"
        cm_df.to_csv(cm_file)
        mlflow.log_artifact(cm_file)
        os.remove(cm_file)
        
        # Save and log model
        model_dir = f"models/logistic_poison_{int(poison_ratio*100)}pct"
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = f"{model_dir}/iris_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        mlflow.sklearn.log_model(model, "model")
        
        # Save detailed metrics
        metrics = {
            'model_type': 'LogisticRegression',
            'poison_ratio': poison_ratio,
            'n_poisoned_samples': len(poisoned_indices),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_f1_score': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'overfit_gap': train_accuracy - test_accuracy,
            'poisoned_indices': poisoned_indices[:50]
        }
        
        metrics_path = f"{model_dir}/metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        mlflow.log_artifact(metrics_path)
        
        print(f"\n[SAVED] Model: {model_path}")
        print(f"[SAVED] Metrics: {metrics_path}")
        print(f"[MLFLOW] Run ID: {mlflow.active_run().info.run_id}")
        print(f"{'='*80}\n")
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description='Train IRIS model with data poisoning (Logistic Regression)')
    parser.add_argument('--poison-ratio', type=float, default=0.0,
                       help='Ratio of data to poison (0.0 to 1.0). Example: 0.05 for 5%%')
    parser.add_argument('--experiment-name', type=str, default='iris-poisoning-logistic',
                       help='MLFlow experiment name')
    
    args = parser.parse_args()
    
    if args.poison_ratio < 0 or args.poison_ratio > 1:
        raise ValueError("Poison ratio must be between 0.0 and 1.0")
    
    train_model_with_poisoning(
        poison_ratio=args.poison_ratio,
        experiment_name=args.experiment_name
    )


if __name__ == "__main__":
    main()


