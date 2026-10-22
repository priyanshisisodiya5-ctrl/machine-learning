"""Machine Learning starter pipeline."""

import sys


def run_pipeline() -> None:
    print("=" * 50)
    print("🤖 Machine Learning Workspace Initialized Successfully!")
    print(f"🐍 Python Version: {sys.version.split()[0]}")
    print("=" * 50)
    print("\nNext steps:")
    print("  1. Add datasets to the data/ directory")
    print("  2. Build training scripts under src/")
    print("  3. Run experiments in notebooks/")


if __name__ == "__main__":
    run_pipeline()
