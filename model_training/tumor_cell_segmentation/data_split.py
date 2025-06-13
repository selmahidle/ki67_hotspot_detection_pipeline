import os
import re
import random
from collections import defaultdict
import shutil
from pathlib import Path

STO_SLIDE_IDS = {
    "01gkljsgivng", "32df0kxxew74", "3a4qhnporh8r", "3dgm64mr49sc",
    "6r6k7ih2dd7p", "7txk0ps4sikg", "7xv4x639rfbg", "ajk1dn1fhoi3",
    "c4nea8o6lup1", "cglrr5vd0d9a", "du65g2cx8nl3", "i21y8s14ckkx",
    "iwsnezx0izh7", "j8daraj5jr5w", "kg7fyg5ch6q3", "l1e33nhcim7b",
    "lc1qw8mch4mo", "mg1rmosiczje", "p79wmlxr7p6p", "psgffptayudq",
    "q1zjievq81o1", "qkje7kr6tqt4", "qny2m87fcepo", "rmk6fwjxbj6s",
    "rqyozzu9qnos", "sxgvmkeyj7dq", "szof39pilkad", "unj3jazbqw2hd",
    "wz8shzxklk6wd", "xi089zbbwzng", "yiir8r5utz5h"
}

def parse_id_filename(fname, id_pattern):
    match = id_pattern.search(fname)
    return f"ID_{match.group(1)}" if match else None

def parse_ki67_filename(fname, ki67_pattern):
    match = ki67_pattern.search(fname)
    if not match:
        return None
    num = match.group(1)
    lower = fname.lower()
    if "ki67train" in lower:
        return f"ki67_train_{num}"
    elif "ki67test" in lower:
        return f"ki67_test_{num}"
    else:
        return f"ki67_{num}"

def create_dataset_splits(dataset_foldername="/cluster/home/selmahi/datasets/TumorCell_segment_Train_Test_Selma_300125"):
    random.seed(42)
    id_pattern = re.compile(r"(?i)(?:auto_orig_ID|auto_ID|autoID|ID)(\d+)")
    ki67_pattern = re.compile(r"(?i)(\d+)ki67")

    sto_files = []
    id_files = []
    ki67_files = []
    unknown_files = []
    total_files_walked = 0

    print(f"Scanning dataset folder: {dataset_foldername}")
    for root, dirs, files in os.walk(dataset_foldername):
        rel = os.path.relpath(root, dataset_foldername)
        parts = rel.split(os.sep)
        if len(parts) == 2 and parts[0] in ("train","val","test") and parts[1] in ("images","labels"):
            split, ftype = parts
        else:
            continue

        for fname in files:
            total_files_walked += 1
            full_path = os.path.join(root, fname)
            lower_name = fname.lower()
            extracted_id = None

            if "ki67" in lower_name:
                extracted_id = parse_ki67_filename(fname, ki67_pattern)
                if extracted_id:
                    ki67_files.append((full_path, extracted_id, split, ftype))
                else:
                    unknown_files.append((full_path, None, split, ftype))
            elif "id" in lower_name:
                extracted_id = parse_id_filename(fname, id_pattern)
                if extracted_id:
                    id_files.append((full_path, extracted_id, split, ftype))
                else:
                    unknown_files.append((full_path, None, split, ftype))
            else:
                matched_sto = next((s for s in STO_SLIDE_IDS if s in lower_name), None)
                if matched_sto:
                    sto_files.append((full_path, matched_sto, split, ftype))
                else:
                    unknown_files.append((full_path, None, split, ftype))

    print(f"Finished scanning. Found {total_files_walked} total files.")
    print(f"Categorized: {len(sto_files)} STO, {len(id_files)} ID, {len(ki67_files)} Ki67, {len(unknown_files)} Unknown files.")

    # --- Process STO Files ---
    sto_group = defaultdict(lambda: {"image": [], "label": []})
    for path, slide_id, split, ftype in sto_files:
        if ftype == "images":
            sto_group[slide_id]["image"].append(path)
        else:
            sto_group[slide_id]["label"].append(path)

    unique_sto_ids = sorted(sto_group.keys())
    train_sto_ids = unique_sto_ids[:15]
    val_sto_ids   = unique_sto_ids[15:23]
    test_sto_ids  = unique_sto_ids[23:]
    print(f"\nSTO Split: {len(train_sto_ids)} train, {len(val_sto_ids)} val, {len(test_sto_ids)} test slide IDs")

    def add_files(slide_ids, group, out_img, out_lbl):
        for sid in slide_ids:
            out_img.extend(group[sid]["image"])
            out_lbl.extend(group[sid]["label"])

    sto_train_images, sto_train_labels = [], []
    sto_val_images,   sto_val_labels   = [], []
    sto_test_images,  sto_test_labels  = [], []
    add_files(train_sto_ids, sto_group, sto_train_images, sto_train_labels)
    add_files(val_sto_ids,   sto_group, sto_val_images,   sto_val_labels)
    add_files(test_sto_ids,  sto_group, sto_test_images,  sto_test_labels)

    # --- Process ID Files ---
    id_group = defaultdict(lambda: {"image": [], "label": []})
    for path, slide_id, split, ftype in id_files:
        if ftype == "images":
            id_group[slide_id]["image"].append(path)
        else:
            id_group[slide_id]["label"].append(path)

    unique_id_ids = sorted(id_group.keys())
    random.shuffle(unique_id_ids)
    n_id = len(unique_id_ids)
    t1 = int(0.70 * n_id)
    t2 = t1 + int(0.15 * n_id)
    train_id_ids = unique_id_ids[:t1]
    val_id_ids   = unique_id_ids[t1:t2]
    test_id_ids  = unique_id_ids[t2:]
    print(f"\nID Split: {len(train_id_ids)} train, {len(val_id_ids)} val, {len(test_id_ids)} test slide IDs")

    id_train_images, id_train_labels = [], []
    id_val_images,   id_val_labels   = [], []
    id_test_images,  id_test_labels  = [], []
    add_files(train_id_ids, id_group, id_train_images, id_train_labels)
    add_files(val_id_ids,   id_group, id_val_images,   id_val_labels)
    add_files(test_id_ids,  id_group, id_test_images,  id_test_labels)

    # --- Process Ki67 Files ---
    ki67_group = defaultdict(lambda: {"image": [], "label": []})
    for path, slide_id, split, ftype in ki67_files:
        if ftype == "images":
            ki67_group[slide_id]["image"].append(path)
        else:
            ki67_group[slide_id]["label"].append(path)

    unique_ki67_ids = sorted(ki67_group.keys())
    random.shuffle(unique_ki67_ids)
    n_k = len(unique_ki67_ids)
    k1 = int(0.70 * n_k)
    k2 = k1 + int(0.15 * n_k)
    train_ki67_ids = unique_ki67_ids[:k1]
    val_ki67_ids   = unique_ki67_ids[k1:k2]
    test_ki67_ids  = unique_ki67_ids[k2:]
    print(f"\nKi67 Split: {len(train_ki67_ids)} train, {len(val_ki67_ids)} val, {len(test_ki67_ids)} test slide IDs")

    ki67_train_images, ki67_train_labels = [], []
    ki67_val_images,   ki67_val_labels   = [], []
    ki67_test_images,  ki67_test_labels  = [], []
    add_files(train_ki67_ids, ki67_group, ki67_train_images, ki67_train_labels)
    add_files(val_ki67_ids,   ki67_group, ki67_val_images,   ki67_val_labels)
    add_files(test_ki67_ids,  ki67_group, ki67_test_images,  ki67_test_labels)

    final_train_images = sto_train_images + id_train_images + ki67_train_images
    final_val_images   = sto_val_images   + id_val_images   + ki67_val_images
    final_test_images  = sto_test_images  + id_test_images  + ki67_test_images

    final_train_labels = sto_train_labels + id_train_labels + ki67_train_labels
    final_val_labels   = sto_val_labels   + id_val_labels   + ki67_val_labels
    final_test_labels  = sto_test_labels  + id_test_labels  + ki67_test_labels

    print("\n--- File Usage Summary ---")
    used = (len(final_train_images)+len(final_val_images)+len(final_test_images) +
            len(final_train_labels)+len(final_val_labels)+len(final_test_labels))
    print(f"Total files walked: {total_files_walked}, included: {used}")

    return {
        "train_images": final_train_images,
        "val_images":   final_val_images,
        "test_images":  final_test_images,
        "train_labels": final_train_labels,
        "val_labels":   final_val_labels,
        "test_labels":  final_test_labels
    }
