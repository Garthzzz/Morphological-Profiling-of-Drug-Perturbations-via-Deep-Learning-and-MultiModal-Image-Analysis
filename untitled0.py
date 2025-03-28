# Step 0: Prepare data - rely on first 9 characters to match aggregator tiff.
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

class AggregateSample:
    """
    A structure to hold a single 'XXXXX_median_aggregated.tiff' file's path
    and its associated label (Metadata_pert_iname).
    """
    def __init__(self, filepath, label_str):
        self.filepath = filepath
        self.label_str = label_str

def get_prefix9(csv_fname):
    """
    Extract the first 9 characters from the csv_fname.
    E.g. 'r16c03f06p01-ch3sk1fk1fl1.tiff' -> 'r16c03f06'
    We'll rely on these 9 chars to build aggregator filename => 'r16c03f06_median_aggregated.tiff'.
    """
    return csv_fname[:9]  # only the first 9 chars

def build_aggregated_mapping(csv_file, root_dir):
    """
    1) Read the CSV
    2) For each row, parse the first 9 chars as the aggregator prefix
    3) aggregator_name = prefix9 + '_median_aggregated.tiff'
    4) If multiple different labels map to the same aggregator, that's a conflict
    5) If aggregator file doesn't exist, mark missing
    6) Return valid_samples, missing_files, conflict_files, label_to_idx
    """
    metadata = pd.read_csv(csv_file)
    
    aggregator_dict = {}  # aggregator_file_name -> set_of_labels
    
    for idx, row in metadata.iterrows():
        csv_fname = row['FileName_OrigRNA']
        label_str = row['Metadata_pert_iname']
        
        # 1) get the first 9 chars
        prefix9 = get_prefix9(csv_fname)
        
        # 2) aggregator name
        agg_fname = prefix9 + "_median_aggregated.tiff"
        
        if agg_fname not in aggregator_dict:
            aggregator_dict[agg_fname] = set()
        aggregator_dict[agg_fname].add(label_str)
    
    valid_samples = []
    missing_files = []
    conflict_files = []
    
    # Now check aggregator_dict
    for agg_fname, label_set in aggregator_dict.items():
        if len(label_set) > 1:
            # conflict: multiple distinct labels
            conflict_files.append(agg_fname)
            continue
        
        # exactly one label
        label_str = list(label_set)[0]
        
        full_path = os.path.join(root_dir, agg_fname)
        if os.path.exists(full_path):
            # valid
            valid_samples.append((full_path, label_str))
        else:
            # missing
            missing_files.append(agg_fname)
    
    # Summaries
    print("\n=== Step 0 Summary ===")
    print(f"Total aggregator entries found in CSV: {len(aggregator_dict)}")
    print(f"  Conflicts (multiple distinct labels): {len(conflict_files)}")
    for cf in conflict_files:
        print("    ", cf)
    print(f"  Missing aggregator files: {len(missing_files)}")
    for mf in missing_files:
        print("    ", mf)
    print(f"  Valid aggregator files (no conflict, exist in folder): {len(valid_samples)}")
    
    # label -> idx
    unique_labels = sorted(set([lab for (_, lab) in valid_samples]))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    
    return valid_samples, label_to_idx, missing_files, conflict_files

def prepare_data_kfold(csv_file, root_dir, n_splits=5):
    """
    Combine all steps: read CSV, build aggregator mapping, filter out conflicts, do StratifiedKFold.
    Returns (all_data, label_to_idx, skf, numeric_labels).
    """
    valid_samples, label_to_idx, missing_files, conflict_files = build_aggregated_mapping(csv_file, root_dir)
    
    # Convert to numeric labels for stratification
    numeric_labels = []
    for (_, lab_str) in valid_samples:
        numeric_labels.append(label_to_idx[lab_str])
    numeric_labels = np.array(numeric_labels)
    
    # Build 5-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # all_data is the list of (filepath, label_str)
    all_data = valid_samples
    
    return all_data, label_to_idx, skf, numeric_labels

# Example usage
data_dir = r"E:\5243\project2"
image_dir = os.path.join(data_dir, "downsampled_data", "downsampled_data")
csv_file = os.path.join(data_dir, "metadata_BR00116991.csv")

all_data, label_to_idx, skf, numeric_labels = prepare_data_kfold(
    csv_file=csv_file, 
    root_dir=image_dir, 
    n_splits=5
)

print("\n=== Final Step 0 Results ===")
print(f"Useable aggregator image files: {len(all_data)}")
print(f"Unique labels: {len(label_to_idx)}")



# Step 0: Prepare data - rely on first 9 characters to match aggregator tiff.
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

class AggregateSample:
    """
    A structure to hold a single 'XXXXX_median_aggregated.tiff' file's path
    and its associated label (Metadata_pert_iname).
    """
    def __init__(self, filepath, label_str):
        self.filepath = filepath
        self.label_str = label_str

def get_prefix9(csv_fname):
    """
    Extract the first 9 characters from the csv_fname.
    E.g. 'r16c03f06p01-ch3sk1fk1fl1.tiff' -> 'r16c03f06'
    We'll rely on these 9 chars to build aggregator filename => 'r16c03f06_median_aggregated.tiff'.
    """
    return csv_fname[:9]  # only the first 9 chars

def build_aggregated_mapping(csv_file, root_dir):
    """
    1) Read the CSV
    2) For each row, parse the first 9 chars as the aggregator prefix
    3) aggregator_name = prefix9 + '_median_aggregated.tiff'
    4) If multiple different labels map to the same aggregator, that's a conflict
    5) If aggregator file doesn't exist, mark missing
    6) Return valid_samples, missing_files, conflict_files, label_to_idx
    """
    metadata = pd.read_csv(csv_file)
    
    aggregator_dict = {}  # aggregator_file_name -> set_of_labels
    
    for idx, row in metadata.iterrows():
        csv_fname = row['FileName_OrigRNA']
        label_str = row['Metadata_pert_iname']
        
        prefix9 = get_prefix9(csv_fname)
        agg_fname = prefix9 + "_median_aggregated.tiff"
        
        if agg_fname not in aggregator_dict:
            aggregator_dict[agg_fname] = set()
        aggregator_dict[agg_fname].add(label_str)
    
    valid_samples = []
    missing_files = []
    conflict_files = []
    
    for agg_fname, label_set in aggregator_dict.items():
        if len(label_set) > 1:
            conflict_files.append(agg_fname)
            continue
        
        label_str = list(label_set)[0]
        full_path = os.path.join(root_dir, agg_fname)
        
        if os.path.exists(full_path):
            valid_samples.append((full_path, label_str))
        else:
            missing_files.append(agg_fname)
    
    print("\n=== Step 0 Summary ===")
    print(f"Total aggregator entries found in CSV: {len(aggregator_dict)}")
    print(f"  Conflicts (multiple distinct labels): {len(conflict_files)}")
    for cf in conflict_files:
        print("    ", cf)
    print(f"  Missing aggregator files: {len(missing_files)}")
    for mf in missing_files:
        print("    ", mf)
    print(f"  Valid aggregator files (no conflict, exist in folder): {len(valid_samples)}")
    
    unique_labels = sorted(set([lab for (_, lab) in valid_samples]))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    
    return valid_samples, label_to_idx, missing_files, conflict_files

def prepare_data_kfold(csv_file, root_dir, n_splits=5):
    valid_samples, label_to_idx, missing_files, conflict_files = build_aggregated_mapping(csv_file, root_dir)
    numeric_labels = []
    for (_, lab_str) in valid_samples:
        numeric_labels.append(label_to_idx[lab_str])
    numeric_labels = np.array(numeric_labels)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_data = valid_samples
    
    return all_data, label_to_idx, skf, numeric_labels

# Example usage
data_dir = r"E:\5243\project2"
image_dir = os.path.join(data_dir, "downsampled_data", "downsampled_data")
csv_file = os.path.join(data_dir, "metadata_BR00116991.csv")

all_data, label_to_idx, skf, numeric_labels = prepare_data_kfold(
    csv_file=csv_file, 
    root_dir=image_dir, 
    n_splits=5
)

print("\n=== Final Step 0 Results ===")
print(f"Useable aggregator image files: {len(all_data)}")
print(f"Unique labels: {len(label_to_idx)}")
