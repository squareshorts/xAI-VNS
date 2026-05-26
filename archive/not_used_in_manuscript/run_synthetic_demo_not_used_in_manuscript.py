from scripts.run_pipeline import run_pipeline

if __name__ == "__main__":
    print("==========================================================")
    print("RUNNING MINIMAL DEMO (SYNTHETIC DATA)")
    print("==========================================================")
    print("NOTE: All results generated here are for pipeline validation only.")
    print("They do not represent scientific results or evidence of clinical VNS efficacy.")
    print("==========================================================\n")
    
    # Run the pipeline forcing synthetic data
    run_pipeline(use_synthetic=True, subject="demo_subject")
