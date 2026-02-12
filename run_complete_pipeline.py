"""
MASTER SCRIPT - BCI Neural Prosthetics EEG Data Engineering
Executes complete pipeline from data acquisition to SQL database
"""

import subprocess
import sys
import time

def run_step(step_number, script_name, description):
    """Run a pipeline step and handle errors"""
    print("\n" + "="*70)
    print(f"EXECUTING STEP {step_number}: {description}")
    print("="*70)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              check=True, 
                              capture_output=True, 
                              text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR in Step {step_number}:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Execute complete pipeline"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║        BCI NEURAL PROSTHETICS EEG DATA ENGINEERING PROJECT       ║
    ║                    Complete Pipeline Execution                   ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    start_time = time.time()
    
    steps = [
        (1, "step1_data_acquisition.py", "Data Acquisition & Setup"),
        (2, "step2_preprocessing.py", "EEG Preprocessing (MNE Pipeline)"),
        (3, "step3_sql_database.py", "SQL Database Creation"),
        (4, "step4_analysis_documentation.py", "Analysis & Documentation")
    ]
    
    for step_num, script, description in steps:
        success = run_step(step_num, script, description)
        if not success:
            print(f"\n❌ Pipeline failed at step {step_num}")
            sys.exit(1)
        time.sleep(1)
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n⏱️  Total execution time: {elapsed_time:.2f} seconds")
    print("\n📂 Project Structure:")
    print("""
    bci_eeg_neural_prosthetics/
    ├── data/
    │   ├── raw/                        # Raw EEG data (NumPy)
    │   ├── processed/                  # Preprocessed EEG data
    │   └── sql_database/               # SQLite database
    ├── results/
    │   ├── preprocessing/              # Before/after comparisons
    │   ├── visualizations/             # Publication-ready plots
    │   └── quality_reports/            # Analysis results
    ├── documentation/                  # Research documentation
    └── step1-4 scripts + master        # Reproducible pipeline
    """)
    
    print("\n🚀 NEXT STEPS:")
    print("   1. Query SQL database for neuroscience analysis")
    print("   2. Implement BCI classification algorithms")
    print("   3. Review documentation/research_documentation.txt")
    print("   4. Use preprocessed data for machine learning")
    print("\n✅ Project ready for neural prosthetics research!")

if __name__ == "__main__":
    main()
