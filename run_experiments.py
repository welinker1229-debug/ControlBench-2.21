"""
Updated runner for the fixed text classification experiments.
Now uses the proper train/val/test split with hyperparameter tuning.
"""
import os
import sys

def main():
    print("🚀 FIXED TEXT CLASSIFICATION EXPERIMENTS")
    print("="*80)
    print("Using Train/Validation/Test Split (60:20:20)")
    print("With hyperparameter tuning for all training-based methods")
    print("="*80)
    
    # Check if the fixed experiments file exists
    if not os.path.exists("text_classification_experiments.py"):
        print("❌ text_classification_experiments.py not found!")
        print("Please make sure you have the fixed experiment file.")
        return
    
    # Import the fixed version
    try:
        from text_classification_experiments import run_all_experiments, set_seed
        print("✅ Successfully imported fixed experiments")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have the required packages:")
        print("pip install torch scikit-learn pandas numpy matplotlib seaborn transformers")
        print("pip install simcse sentence-transformers  # for SimCSE and SBERT experiments")
        return
    
    # Configuration
    SPLIT_DIR = "split_datasets"  # Your split datasets directory
    DATASETS = ["abortion", "trump", "religion", "capitalism", "lgbtq"]  # All your datasets
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("\nUsage:")
            print("  python run_experiments.py                           # Run all datasets")
            print("  python run_text_experiments.py dataset1 dataset2         # Run specific datasets")
            print("  python run_text_experiments.py --split_dir path dataset  # Use custom split directory")
            print("\nAvailable datasets:")
            print("  abortion, trump, religion, capitalism, lgbtq")
            print("\nHyperparameter tuning included for:")
            print("  • TF-IDF + LogReg: Feature extraction + classifier parameters")
            print("  • SimCSE: Classifier head hyperparameters") 
            print("  • Sentence-BERT: Model selection + classifier hyperparameters")
            print("  • BERT: Full fine-tuning hyperparameters")
            print("  • RoBERTa: Full fine-tuning hyperparameters")
            return
        else:
            # Parse arguments
            args = sys.argv[1:]
            if "--split_dir" in args:
                split_idx = args.index("--split_dir")
                if split_idx + 1 < len(args):
                    SPLIT_DIR = args[split_idx + 1]
                    args = args[:split_idx] + args[split_idx + 2:]
            
            if args:  # If there are remaining arguments, treat as dataset names
                DATASETS = args
    
    print(f"📁 Split directory: {SPLIT_DIR}")
    print(f"📊 Datasets to run: {DATASETS}")
    
    # Check if split directories exist
    missing_dirs = []
    available_datasets = []
    
    for dataset in DATASETS:
        dataset_dir = os.path.join(SPLIT_DIR, dataset)
        if not os.path.exists(dataset_dir):
            missing_dirs.append(dataset_dir)
            continue
            
        # Check for required files
        required_files = ["train.json", "validation.json", "test.json"]
        missing_files = []
        for file_name in required_files:
            file_path = os.path.join(dataset_dir, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            missing_dirs.extend(missing_files)
        else:
            available_datasets.append(dataset)
    
    if missing_dirs:
        print("❌ Missing split files or directories:")
        for path in missing_dirs:
            print(f"   • {path}")
        print(f"\n💡 Make sure you have run split_dataset.py first to create the splits:")
        print(f"   python split_dataset.py --dataset <dataset_name> --output_dir {SPLIT_DIR}")
        print(f"\n   Expected structure:")
        print(f"   {SPLIT_DIR}/")
        print(f"   ├── dataset1/")
        print(f"   │   ├── train.json")
        print(f"   │   ├── validation.json")
        print(f"   │   ├── test.json")
        print(f"   │   └── metadata.json")
        print(f"   └── dataset2/...")
        
        if available_datasets:
            print(f"\n✅ Available datasets: {available_datasets}")
            response = input(f"Continue with available datasets only? (y/n): ")
            if response.lower() != 'y':
                return
            DATASETS = available_datasets
        else:
            return
    
    # Set random seed
    set_seed(42)
    
    print(f"\n🔥 Starting experiments with hyperparameter tuning on {len(DATASETS)} datasets...")
    print("⏱️  This may take a while due to comprehensive hyperparameter search!")
    print("\nHyperparameter tuning details:")
    print("  • TF-IDF + LogReg: ~12 parameter combinations")
    print("  • SimCSE: ~5 classifier configurations")
    print("  • Sentence-BERT: ~3 models × 4 classifier configs = 12 combinations")
    print("  • BERT: ~4 training configurations")
    print("  • RoBERTa: ~4 training configurations")
    
    # Estimate time
    total_configs = len(DATASETS) * (12 + 5 + 12 + 4 + 4)  # Rough estimate
    print(f"  📊 Total configurations to test: ~{total_configs}")
    print(f"  ⏱️  Estimated time: 30-60 minutes per dataset (depending on size and GPU)")
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Experiment cancelled.")
        return
    
    try:
        # Run experiments
        all_results, experiment_folders = run_all_experiments(DATASETS, SPLIT_DIR)
        
        print("\n✅ Experiments completed successfully!")
        print("\n📋 Results locations:")
        for folder in experiment_folders:
            print(f"   • {folder}/")
            print(f"     ├── results.txt (detailed results)")
            print(f"     ├── hyperparameter_search.txt (tuning details)")
            print(f"     └── results_summary.csv (metrics table)")
        
        # Quick summary table
        print(f"\n📊 QUICK RESULTS SUMMARY:")
        print(f"{'Dataset':<12} {'Best Method':<15} {'Test F1':<8} {'Val F1':<8} {'Best Hyperparams'}")
        print("-" * 80)
        
        for dataset, results in all_results.items():
            valid_results = {k: v for k, v in results.items() 
                           if isinstance(v.get('macro_f1'), (int, float)) and v['macro_f1'] > 0}
            
            if valid_results:
                best_model = max(valid_results.keys(), 
                               key=lambda x: valid_results[x]['macro_f1'])
                best_metrics = valid_results[best_model]
                test_f1 = best_metrics['macro_f1']
                val_f1 = best_metrics.get('validation_f1', 'N/A')
                val_f1_str = f"{val_f1:.3f}" if isinstance(val_f1, (int, float)) else str(val_f1)
                
                # Summarize best hyperparams
                if 'best_params' in best_metrics:
                    if isinstance(best_metrics['best_params'], dict):
                        param_summary = ", ".join([f"{k}={v}" for k, v in list(best_metrics['best_params'].items())[:2]])
                        if len(best_metrics['best_params']) > 2:
                            param_summary += "..."
                    else:
                        param_summary = str(best_metrics['best_params'])[:30]
                else:
                    param_summary = "Default"
                
                print(f"{dataset:<12} {best_model:<15} {test_f1:<8.3f} {val_f1_str:<8} {param_summary}")
            else:
                print(f"{dataset:<12} {'No results':<15} {'N/A':<8} {'N/A':<8} {'N/A'}")
        
        # Print validation vs test performance comparison
        print(f"\n🔍 VALIDATION vs TEST PERFORMANCE:")
        print("This shows how well hyperparameter tuning worked:")
        print(f"{'Dataset':<12} {'Method':<15} {'Val F1':<8} {'Test F1':<8} {'Difference'}")
        print("-" * 65)
        
        for dataset, results in all_results.items():
            for method, metrics in results.items():
                if (isinstance(metrics.get('macro_f1'), (int, float)) and 
                    isinstance(metrics.get('validation_f1'), (int, float)) and
                    metrics['macro_f1'] > 0):
                    
                    val_f1 = metrics['validation_f1']
                    test_f1 = metrics['macro_f1']
                    diff = test_f1 - val_f1
                    diff_str = f"{diff:+.3f}"
                    
                    print(f"{dataset:<12} {method:<15} {val_f1:<8.3f} {test_f1:<8.3f} {diff_str}")
        
        print(f"\n💡 Tips for interpreting results:")
        print("  • Small Val-Test difference indicates good generalization")
        print("  • Large positive difference may indicate underfitting")
        print("  • Large negative difference may indicate overfitting")
        print("  • Methods with hyperparameter tuning should show better validation performance")
        
    except KeyboardInterrupt:
        print("\n⚠️  Experiments interrupted by user.")
        print("Partial results may be available in experiment folders.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()