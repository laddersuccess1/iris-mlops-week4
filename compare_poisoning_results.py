"""
Compare results from different poisoning levels
"""
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

def compare_results():
    """Compare metrics across all poisoning levels"""
    
    poison_levels = [0, 5, 10, 50]
    results = []
    
    print("\n" + "="*80)
    print("COMPARISON OF DATA POISONING IMPACT")
    print("="*80 + "\n")
    
    for level in poison_levels:
        model_dir = f"models/poison_{level}pct"
        metrics_file = f"{model_dir}/metrics.json"
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                results.append({
                    'Poison Level (%)': level,
                    'Train Accuracy': metrics['train_accuracy'],
                    'Test Accuracy': metrics['test_accuracy'],
                    'F1 Score': metrics['test_f1_score'],
                    'Precision': metrics['test_precision'],
                    'Recall': metrics['test_recall'],
                    'Overfit Gap': metrics['overfit_gap'],
                    'Poisoned Samples': metrics['n_poisoned_samples']
                })
    
    if not results:
        print("No results found. Run experiments first with: bash run_poisoning_experiments.sh")
        return
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    
    # Print table
    print(df.to_string(index=False))
    print("\n" + "="*80 + "\n")
    
    # Calculate degradation
    baseline_acc = df[df['Poison Level (%)'] == 0]['Test Accuracy'].values[0]
    
    print("IMPACT ANALYSIS:")
    print("-" * 80)
    for _, row in df.iterrows():
        poison_level = row['Poison Level (%)']
        test_acc = row['Test Accuracy']
        degradation = (baseline_acc - test_acc) * 100
        
        print(f"Poison Level {int(poison_level):2d}%: "
              f"Test Acc = {test_acc:.4f} | "
              f"Degradation = {degradation:+.2f}% | "
              f"Overfit Gap = {row['Overfit Gap']:.4f}")
    
    print("\n" + "="*80)
    
    # Save comparison
    df.to_csv('models/poisoning_comparison.csv', index=False)
    print("\n[SAVED] Comparison table: models/poisoning_comparison.csv")
    
    return df

if __name__ == "__main__":
    compare_results()


