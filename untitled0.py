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

# Step 1: 5-Fold training with Resize + IncrementalPCA + Logistic Regression

import numpy as np
from PIL import Image
import torch
import math

from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import torchvision.transforms as T

transform_resize = T.Compose([
    T.Resize((64, 64)),  # reduce resolution
    T.ToTensor(),
])

def incremental_pca_fit(X_paths, label_to_idx, n_components=100, batch_size=256):
    ipca = IncrementalPCA(n_components=n_components)
    num_imgs = len(X_paths)
    num_batches = math.ceil(num_imgs / batch_size)
    idx_start = 0
    
    for b in range(num_batches):
        batch_paths = X_paths[idx_start:idx_start+batch_size]
        idx_start += batch_size
        
        batch_data_list = []
        for fp in batch_paths:
            img = Image.open(fp).convert('RGB')
            img_t = transform_resize(img)
            img_np = img_t.numpy().ravel()
            batch_data_list.append(img_np)
        
        if len(batch_data_list) == 0:
            continue
        batch_data = np.stack(batch_data_list, axis=0)
        
        ipca.partial_fit(batch_data)
    
    return ipca

def transform_with_ipca(X_paths, ipca, batch_size=256):
    data_proj_list = []
    idx_start = 0
    num_imgs = len(X_paths)
    num_batches = math.ceil(num_imgs / batch_size)
    
    for b in range(num_batches):
        batch_paths = X_paths[idx_start:idx_start+batch_size]
        idx_start += batch_size
        
        batch_data_list = []
        for fp in batch_paths:
            img = Image.open(fp).convert('RGB')
            img_t = transform_resize(img)
            img_np = img_t.numpy().ravel()
            batch_data_list.append(img_np)
        
        if len(batch_data_list) == 0:
            continue
        batch_data = np.stack(batch_data_list, axis=0)
        
        batch_proj = ipca.transform(batch_data)
        data_proj_list.append(batch_proj)
    
    X_proj = np.concatenate(data_proj_list, axis=0)
    return X_proj

def run_5fold_ipca_logreg(all_data, label_to_idx, skf, numeric_labels, 
                          n_components=100, batch_size=256):
    fold_idx = 1
    for train_index, test_index in skf.split(np.zeros(len(numeric_labels)), numeric_labels):
        print(f"\n=== Fold {fold_idx} ===")
        train_subset = [all_data[i] for i in train_index]
        test_subset  = [all_data[i] for i in test_index]
        
        X_train_paths = [fp for (fp,_) in train_subset]
        y_train_list  = [label_to_idx[lab_str] for (_,lab_str) in train_subset]
        
        X_test_paths = [fp for (fp,_) in test_subset]
        y_test_list  = [label_to_idx[lab_str] for (_,lab_str) in test_subset]
        
        y_train = np.array(y_train_list)
        y_test  = np.array(y_test_list)
        
        print("Train data count:", len(X_train_paths))
        print("Test  data count:", len(X_test_paths))
        
        ipca = incremental_pca_fit(X_train_paths, label_to_idx, 
                                   n_components=n_components, 
                                   batch_size=batch_size)
        X_train_pca = transform_with_ipca(X_train_paths, ipca, batch_size=batch_size)
        X_test_pca  = transform_with_ipca(X_test_paths,  ipca, batch_size=batch_size)
        
        print("Train data shape after PCA:", X_train_pca.shape)
        print("Test  data shape after PCA:",  X_test_pca.shape)
        
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_pca, y_train)
        y_pred = clf.predict(X_test_pca)
        
        acc = accuracy_score(y_test, y_pred)
        print(f"Fold {fold_idx} Accuracy: {acc:.4f}")
        
        fold_idx += 1

# usage:
run_5fold_ipca_logreg(all_data, label_to_idx, skf, numeric_labels, 
                      n_components=100, batch_size=256)
