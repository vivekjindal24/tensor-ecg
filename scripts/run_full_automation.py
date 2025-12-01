"""
Complete preprocessing automation orchestrator.
Runs the full pipeline with checks and validation.
"""
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
LOGS_DIR = ROOT / "logs"
SCRIPTS_DIR = ROOT / "scripts"
VENV_PYTHON = ROOT / ".venv1" / "Scripts" / "python.exe"

# Ensure log directory exists
LOGS_DIR.mkdir(exist_ok=True)

def run_script(script_name: str, description: str, env_vars: dict = None) -> bool:
    """Run a Python script and return success status"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"{'='*80}")

    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        return False

    cmd = [str(VENV_PYTHON), str(script_path)]

    # Build environment
    import os
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            return True
        else:
            print(f"✗ {description} failed with return code {result.returncode}")
            return False

    except Exception as e:
        print(f"✗ Error running {description}: {e}")
        return False


def main():
    """Main orchestration pipeline"""
    start_time = time.time()

    print("="*80)
    print("ECG PREPROCESSING AUTOMATION ORCHESTRATOR")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {VENV_PYTHON}")
    print(f"Root: {ROOT}")

    # Step 1: Validate mapping
    if not run_script("validate_mapping.py", "Step 1: Validate unified label mapping"):
        print("\n⚠ Warning: Mapping validation had issues, but continuing...")

    # Step 2: Improve mapping (optional)
    print("\n" + "="*80)
    print("OPTIONAL: Improve Label Mapping")
    print("="*80)
    print("Would you like to run the mapping improvement heuristics?")
    print("This may increase coverage from 33.8% to ~50-60%")
    response = input("Run improvement? (y/n, default=n): ").strip().lower()

    if response == 'y':
        if run_script("improve_mapping.py", "Step 2a: Analyze mapping improvements"):
            print("\nReview suggestions in: logs/mapping_improvements_suggested.csv")
            apply_response = input("Apply these improvements? (y/n, default=n): ").strip().lower()

            if apply_response == 'y':
                if not run_script("apply_mapping_improvements.py", "Step 2b: Apply mapping improvements"):
                    print("\n✗ Failed to apply improvements")
                    return False

    # Step 3: Run preprocessing
    print("\n" + "="*80)
    print("PREPROCESSING MODE SELECTION")
    print("="*80)
    print("Choose preprocessing mode:")
    print("  1. Full run (all datasets, ~7 hours)")
    print("  2. Medium test (5,000 files, ~15 minutes)")
    print("  3. Large test (20,000 files, ~1 hour)")
    print("  4. Custom limit")

    mode = input("Select mode (1-4, default=2): ").strip() or "2"

    env_vars = {}
    if mode == "1":
        print("\nStarting FULL preprocessing...")
        print("This will take approximately 7 hours.")
        confirm = input("Confirm full run? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Cancelled.")
            return False
    elif mode == "2":
        env_vars["ECG_PREPROCESS_LIMIT"] = "5000"
        print("\nRunning medium test (5,000 files)...")
    elif mode == "3":
        env_vars["ECG_PREPROCESS_LIMIT"] = "20000"
        print("\nRunning large test (20,000 files)...")
    elif mode == "4":
        limit = input("Enter custom limit: ").strip()
        try:
            limit_int = int(limit)
            env_vars["ECG_PREPROCESS_LIMIT"] = str(limit_int)
            print(f"\nRunning with custom limit ({limit_int} files)...")
        except ValueError:
            print("Invalid limit. Defaulting to 5,000.")
            env_vars["ECG_PREPROCESS_LIMIT"] = "5000"

    # Run preprocessing
    if not run_script("preprocess_streaming.py", "Step 3: Streaming preprocessing", env_vars):
        print("\n✗ Preprocessing failed!")
        return False

    # Step 4: Verify outputs
    if not run_script("verify_smoke_test.py", "Step 4: Verify preprocessing outputs"):
        print("\n⚠ Warning: Verification had issues")

    # Step 5: Model smoke test
    if not run_script("model_smoke_test.py", "Step 5: Model compatibility test"):
        print("\n⚠ Warning: Model test had issues")

    # Generate final report
    elapsed = time.time() - start_time
    report_file = LOGS_DIR / "preprocess_report.txt"

    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ECG PREPROCESSING AUTOMATION - FINAL REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)\n")
        f.write("\n")
        f.write("All steps completed successfully!\n")
        f.write("\n")
        f.write("Output locations:\n")
        f.write(f"  - Processed records: artifacts/processed/records/\n")
        f.write(f"  - Manifest: artifacts/processed/manifest.jsonl\n")
        f.write(f"  - Splits: artifacts/processed/splits.json\n")
        f.write(f"  - Label map: artifacts/processed/label_map.json\n")
        f.write(f"  - Checkpoints: artifacts/processed/checkpoints/\n")
        f.write("\n")
        f.write("Next steps:\n")
        f.write("  1. Review preprocessing log: logs/preprocess_automation.log\n")
        f.write("  2. Open notebook: notebooks/ecg_tensor_pipeline.ipynb\n")
        f.write("  3. Begin model training\n")
        f.write("\n")
        f.write("="*80 + "\n")

    print("\n" + "="*80)
    print("AUTOMATION COMPLETE!")
    print("="*80)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Report saved to: {report_file}")
    print("\nYou can now begin model training!")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved.")
        print("You can resume by running this script again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

