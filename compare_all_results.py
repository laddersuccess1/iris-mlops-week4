"""
Compare results from different poisoning levels across all models
"""
import json
import os
import pandas as pd
import sys

def compare_results(model_prefix="poison"):
    """
    Compare metrics across all poisoning levels for a specific model type
    
    Args:
        model_prefix: Either "poison" (RandomForest) or "logistic_poison" (Logistic Regression)
    """
    
    poison_levels = [0, 5, 10, 50]
    results = []
    
    model_name = "RandomForest" if model_prefix == "poison" else "Logistic Regression"
    
    print("\n" + "="*90)
    print(f"COMPARISON OF DATA POISONING IMPACT - {model_name.upper()}")
    print("="*90 + "\n")
    
    for level in poison_levels:
        model_dir = f"models/{model_prefix}_{level}pct"
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
        print(f"❌ No results found for {model_name}.")
        if model_prefix == "poison":
            print("   Run: bash run_poisoning_experiments.sh")
        else:
            print("   Run: bash run_logistic_experiments.sh")
        return None
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    
    # Print table
    print(df.to_string(index=False))
    print("\n" + "="*90 + "\n")
    
    # Calculate degradation
    baseline_acc = df[df['Poison Level (%)'] == 0]['Test Accuracy'].values[0]
    
    print("IMPACT ANALYSIS:")
    print("-" * 90)
    for _, row in df.iterrows():
        poison_level = row['Poison Level (%)']
        test_acc = row['Test Accuracy']
        degradation = (baseline_acc - test_acc) * 100
        
        # Color code based on degradation
        if degradation == 0:
            status = "✓"
        elif degradation < 5:
            status = "⚠️"
        else:
            status = "🚨"
        
        print(f"Poison Level {int(poison_level):2d}%: "
              f"Test Acc = {test_acc:.4f} | "
              f"Degradation = {degradation:+6.2f}% | "
              f"Overfit Gap = {row['Overfit Gap']:+.4f} {status}")
    
    print("\n" + "="*90)
    
    # Save comparison
    output_file = f'models/poisoning_comparison_{model_prefix}.csv'
    df.to_csv(output_file, index=False)
    print(f"\n[SAVED] Comparison table: {output_file}")
    
    return df


def compare_both_models():
    """Compare RandomForest and Logistic Regression side by side"""
    
    print("\n" + "="*90)
    print("COMPARING BOTH MODELS - ROBUSTNESS ANALYSIS")
    print("="*90 + "\n")
    
    # Try to load both
    rf_results = []
    lr_results = []
    
    poison_levels = [0, 5, 10, 50]
    
    for level in poison_levels:
        # RandomForest
        rf_file = f"models/poison_{level}pct/metrics.json"
        if os.path.exists(rf_file):
            with open(rf_file, 'r') as f:
                metrics = json.load(f)
                rf_results.append({
                    'Poison (%)': level,
                    'RF Accuracy': metrics['test_accuracy'],
                    'RF Degradation': 0.0  # Will calculate later
                })
        
        # Logistic Regression
        lr_file = f"models/logistic_poison_{level}pct/metrics.json"
        if os.path.exists(lr_file):
            with open(lr_file, 'r') as f:
                metrics = json.load(f)
                lr_results.append({
                    'Poison (%)': level,
                    'LR Accuracy': metrics['test_accuracy'],
                    'LR Degradation': 0.0  # Will calculate later
                })
    
    if not rf_results and not lr_results:
        print("❌ No results found for either model!")
        print("   Run: bash run_poisoning_experiments.sh")
        print("   Run: bash run_logistic_experiments.sh")
        return
    
    # Merge results
    if rf_results and lr_results:
        df_rf = pd.DataFrame(rf_results)
        df_lr = pd.DataFrame(lr_results)
        df = pd.merge(df_rf, df_lr, on='Poison (%)', how='outer')
        
        # Calculate degradations
        if 'RF Accuracy' in df.columns:
            baseline_rf = df[df['Poison (%)'] == 0]['RF Accuracy'].values[0]
            df['RF Degradation'] = (baseline_rf - df['RF Accuracy']) * 100
        
        if 'LR Accuracy' in df.columns:
            baseline_lr = df[df['Poison (%)'] == 0]['LR Accuracy'].values[0]
            df['LR Degradation'] = (baseline_lr - df['LR Accuracy']) * 100
        
        print(df.to_string(index=False))
        print("\n" + "="*90 + "\n")
        
        print("KEY INSIGHTS:")
        print("-" * 90)
        
        if 'RF Accuracy' in df.columns and 'LR Accuracy' in df.columns:
            rf_50_deg = df[df['Poison (%)'] == 50]['RF Degradation'].values[0]
            lr_50_deg = df[df['Poison (%)'] == 50]['LR Degradation'].values[0]
            
            print(f"RandomForest at 50% poisoning: {rf_50_deg:+.2f}% degradation")
            print(f"Logistic Regression at 50% poisoning: {lr_50_deg:+.2f}% degradation")
            print(f"\nRobustness advantage: RandomForest is {abs(lr_50_deg - rf_50_deg):.1f}% more robust")
            
            if rf_50_deg < 5:
                print("\n✓ RandomForest demonstrates EXCEPTIONAL robustness (ensemble effect)")
            if lr_50_deg > 20:
                print("🚨 Logistic Regression shows SEVERE vulnerability (expected for linear models)")
        
        print("\n" + "="*90)
        
        # Save combined comparison
        df.to_csv('models/comparison_both_models.csv', index=False)
        print("\n[SAVED] Combined comparison: models/comparison_both_models.csv\n")
    
    else:
        print("⚠️  Only one model's results available. Run both experiments to compare.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare poisoning experiment results')
    parser.add_argument('--model', type=str, choices=['rf', 'lr', 'both'], default='both',
                       help='Model to analyze: rf (RandomForest), lr (Logistic Regression), both (comparison)')
    
    args = parser.parse_args()
    
    if args.model == 'rf':
        compare_results(model_prefix="poison")
    elif args.model == 'lr':
        compare_results(model_prefix="logistic_poison")
    else:
        # Try both individually first
        rf_df = compare_results(model_prefix="poison")
        print("\n")
        lr_df = compare_results(model_prefix="logistic_poison")
        print("\n")
        
        # Then compare side by side
        if rf_df is not None or lr_df is not None:
            compare_both_models()


if __name__ == "__main__":
    main()


