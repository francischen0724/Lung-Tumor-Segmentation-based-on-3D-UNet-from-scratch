import os
import shutil
import random

def split_dataset(input_dir, output_base, dataset_name, train_ratio=0.8, seed=42):
    """
    Split a lung ROI dataset into training and testing sets.

    Args:
        input_dir (str): Path to the original dataset (each subdirectory is a case with .pt files).
        output_base (str): Base path where the split structure will be saved.
        dataset_name (str): Name used to label the output (e.g., "MSD" or "NSCLC").
        train_ratio (float): Ratio of data to use for training. Default is 0.8.
        seed (int): Random seed for reproducibility.
    """
    # Get patient folder names
    patients = sorted([p for p in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, p))])
    total = len(patients)

    # Shuffle and split based on the seed
    random.seed(seed)
    random.shuffle(patients)
    train_count = int(train_ratio * total)
    train_patients = patients[:train_count]
    test_patients = patients[train_count:]

    # Define output directories
    train_dir = os.path.join(output_base, dataset_name, "train")
    test_dir = os.path.join(output_base, dataset_name, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Helper function to copy patient subdirectories
    def copy_subset(patients_list, target_dir):
        for p in patients_list:
            src = os.path.join(input_dir, p)
            dst = os.path.join(target_dir, p)
            shutil.copytree(src, dst)

    copy_subset(train_patients, train_dir)
    copy_subset(test_patients, test_dir)

    print(f"✅ {dataset_name} Finish Data Splitting")
    print(f"Train Set: {len(train_patients)} samples -> {train_dir}")
    print(f"Test Set: {len(test_patients)} samples -> {test_dir}")
    print("-" * 40)


if __name__ == "__main__":
    # Input paths (source directories containing lung ROI .pt files)
    # msd_input = "../../datasets_XW/processed_msd1/lungs_roi"
    nsclc_input = "../../datasets_XW/processed_nsclc1/lungs_roi"

    # Output base directory (will contain the split structure)
    output_base = "../../datasets_XW/split1"

    # Perform splitting
    # split_dataset(msd_input, output_base, dataset_name="MSD")
    split_dataset(nsclc_input, output_base, dataset_name="NSCLC")
