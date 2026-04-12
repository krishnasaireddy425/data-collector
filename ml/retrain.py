"""
Online Learning: Retrain models when new CSV data is collected.

Checks for new CSV files not in the last training set,
retrains all models, and saves updated versions.

Tracks model performance over time in retrain_log.json.

Usage:
  python3 analysis/retrain.py           # retrain if new data exists
  python3 analysis/retrain.py --force   # force retrain even without new data
"""

import glob
import json
import os
import argparse
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
LOG_PATH = os.path.join(MODEL_DIR, "retrain_log.json")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "price_collector", "data")
CSV_PATTERN = os.path.join(DATA_DIR, "btc-updown-5m-*.csv")


def get_current_epochs():
    """Get set of all epoch numbers from CSV files."""
    files = sorted(glob.glob(CSV_PATTERN))
    epochs = set()
    for f in files:
        try:
            epoch = int(os.path.basename(f).replace("btc-updown-5m-", "").replace(".csv", ""))
            epochs.add(epoch)
        except ValueError:
            continue
    return epochs


def get_last_trained_epochs():
    """Load epochs from last training run."""
    stats_path = os.path.join(MODEL_DIR, "training_stats.json")
    epochs_path = os.path.join(MODEL_DIR, "trained_epochs.json")

    if os.path.exists(epochs_path):
        with open(epochs_path) as f:
            return set(json.load(f))
    return set()


def save_trained_epochs(epochs):
    """Save the set of epochs used in training."""
    epochs_path = os.path.join(MODEL_DIR, "trained_epochs.json")
    with open(epochs_path, "w") as f:
        json.dump(sorted(list(epochs)), f)


def log_retrain(n_samples, n_new, stats):
    """Append to retrain log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_samples,
        "n_new_samples": n_new,
        "accuracies": {k: v["wf_accuracy"] for k, v in stats.items()},
    }

    log = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            log = json.load(f)

    log.append(entry)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    print(f"  Logged retrain #{len(log)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force retrain")
    args = parser.parse_args()

    current_epochs = get_current_epochs()
    last_epochs = get_last_trained_epochs()
    new_epochs = current_epochs - last_epochs

    print(f"Current CSV files: {len(current_epochs)}")
    print(f"Last training set: {len(last_epochs)}")
    print(f"New samples: {len(new_epochs)}")

    if not new_epochs and not args.force:
        print("No new data. Use --force to retrain anyway.")
        return

    print(f"\nRetraining with {len(current_epochs)} total samples...")

    # Import and run training
    from train_model import load_all_windows, train_and_evaluate

    windows = load_all_windows()
    timepoints = [60, 90, 120, 150, 180, 210, 240]
    stats = train_and_evaluate(windows, timepoints)

    # Save which epochs were used
    save_trained_epochs(current_epochs)

    # Log the retrain
    log_retrain(len(windows), len(new_epochs), stats)

    print(f"\nRetrain complete. Models updated with {len(windows)} samples.")


if __name__ == "__main__":
    main()
