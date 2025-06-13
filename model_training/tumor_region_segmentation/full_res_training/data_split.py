import os
import re
import random
from collections import defaultdict 

def create_dataset_splits(dataset_foldername):
    random.seed(42)

    train_images_path = os.path.join(dataset_foldername, "TrainImages")
    val_images_path   = os.path.join(dataset_foldername, "ValidationImages")
    train_labels_path = os.path.join(dataset_foldername, "TrainLabels")
    val_labels_path   = os.path.join(dataset_foldername, "ValidationLabels")

    sto_pattern = re.compile(r"sto(.*?)d2x", re.IGNORECASE)
    id_pattern = re.compile(r"(?i)(?:auto)?id(\d+)")
    ki67_pattern = re.compile(r"(?i)(?:auto)?(\d+)ki67")

    def label_basename(fname):
        root, _ = os.path.splitext(fname)
        if root.startswith("Labels_"):
            root = root[7:]
        return root

    def parse_id_filename(fname, pattern):
        match = pattern.search(fname)
        if match:
            digits = match.group(1)
            if digits is not None and digits.isdigit():
                 return "ID_" + digits
        return None

    def parse_ki67_filename(fname, pattern):
        match = pattern.search(fname)
        if match:
            num = match.group(1)
            fname_lower = fname.lower()
            if "ki67train" in fname_lower:
                return f"ki67_train_{num}"
            elif "ki67test" in fname_lower:
                return f"ki67_test_{num}"
            else:
                return f"ki67_{num}"
        return None

    image_subfolders = [train_images_path, val_images_path]
    label_subfolders = [train_labels_path, val_labels_path]

    # ============================
    # PART A: STO Slides
    # ============================
    sto_slide_ids = set()
    sto_files_temp = []

    for folder_list in [image_subfolders, label_subfolders]:
        for directory in folder_list:
            if not os.path.exists(directory): continue
            for fname in os.listdir(directory):
                full_path = os.path.join(directory, fname)
                if os.path.isfile(full_path):
                    name_to_match = label_basename(fname) if "label" in directory.lower() else fname
                    match = sto_pattern.search(name_to_match)
                    if match:
                        sid = match.group(1)
                        sto_slide_ids.add(sid)
                        sto_files_temp.append((full_path, sid))

    sorted_sto_slides = sorted(list(sto_slide_ids))
    sto_train_ids = sorted_sto_slides[:15]
    sto_val_ids   = sorted_sto_slides[15:23]
    sto_test_ids  = sorted_sto_slides[23:]

    sto_train_images, sto_val_images, sto_test_images = [], [], []
    sto_train_labels, sto_val_labels, sto_test_labels = [], [], []
    for path, sid in sto_files_temp:
        if sid in sto_train_ids:
            if "image" in path.lower(): sto_train_images.append(path)
            elif "label" in path.lower(): sto_train_labels.append(path)
        elif sid in sto_val_ids:
            if "image" in path.lower(): sto_val_images.append(path)
            elif "label" in path.lower(): sto_val_labels.append(path)
        elif sid in sto_test_ids:
            if "image" in path.lower(): sto_test_images.append(path)
            elif "label" in path.lower(): sto_test_labels.append(path)


    # ============================
    # PART B: NON-STO slides
    # ============================
    id_image_dict = defaultdict(list)
    ki67_image_dict = defaultdict(list)
    id_label_dict = defaultdict(list)
    ki67_label_dict = defaultdict(list)

    for img_dir in image_subfolders:
        if not os.path.exists(img_dir): continue
        for fname in os.listdir(img_dir):
            if sto_pattern.search(fname): continue
            full_path = os.path.join(img_dir, fname)
            if os.path.isfile(full_path):
                parsed_id = parse_ki67_filename(fname, ki67_pattern)
                if parsed_id:
                    ki67_image_dict[parsed_id].append(full_path)
                else:
                    parsed_id = parse_id_filename(fname, id_pattern)
                    if parsed_id:
                        id_image_dict[parsed_id].append(full_path)

    for lbl_dir in label_subfolders:
        if not os.path.exists(lbl_dir): continue
        for fname in os.listdir(lbl_dir):
             base_for_sto_check = label_basename(fname)
             if sto_pattern.search(base_for_sto_check): continue 
             full_path = os.path.join(lbl_dir, fname)
             if os.path.isfile(full_path):
                base = label_basename(fname)
                parsed_id = parse_ki67_filename(base, ki67_pattern)
                if parsed_id:
                    ki67_label_dict[parsed_id].append(full_path)
                else:
                    parsed_id = parse_id_filename(base, id_pattern)
                    if parsed_id:
                        id_label_dict[parsed_id].append(full_path)

    id_image_slides = set(id_image_dict.keys())
    id_label_slides = set(id_label_dict.keys())
    matched_id_slides = sorted(list(id_image_slides.intersection(id_label_slides)))
    random.shuffle(matched_id_slides)
    num_id_slides = len(matched_id_slides)
    id_train_end = int(0.70 * num_id_slides)
    id_val_end   = int(0.85 * num_id_slides)
    id_train_slide_ids = matched_id_slides[:id_train_end]
    id_val_slide_ids   = matched_id_slides[id_train_end:id_val_end]
    id_test_slide_ids  = matched_id_slides[id_val_end:]

    ki67_image_slides = set(ki67_image_dict.keys())
    ki67_label_slides = set(ki67_label_dict.keys())
    matched_ki67_slides = sorted(list(ki67_image_slides.intersection(ki67_label_slides)))
    random.shuffle(matched_ki67_slides)
    num_ki67_slides = len(matched_ki67_slides)
    ki67_train_end = int(0.70 * num_ki67_slides)
    ki67_val_end   = int(0.85 * num_ki67_slides)
    ki67_train_slide_ids = matched_ki67_slides[:ki67_train_end]
    ki67_val_slide_ids   = matched_ki67_slides[ki67_train_end:ki67_val_end]
    ki67_test_slide_ids  = matched_ki67_slides[ki67_val_end:]

    def add_paths_for_slides(sids, img_dict, lbl_dict, out_images, out_labels):
        for sid in sids:
            out_images.extend(img_dict.get(sid, []))
            out_labels.extend(lbl_dict.get(sid, []))

    id_train_images, id_val_images, id_test_images = [], [], []
    id_train_labels, id_val_labels, id_test_labels = [], [], []
    add_paths_for_slides(id_train_slide_ids, id_image_dict, id_label_dict, id_train_images, id_train_labels)
    add_paths_for_slides(id_val_slide_ids,   id_image_dict, id_label_dict, id_val_images, id_val_labels)
    add_paths_for_slides(id_test_slide_ids,  id_image_dict, id_label_dict, id_test_images, id_test_labels)

    ki67_train_images, ki67_val_images, ki67_test_images = [], [], []
    ki67_train_labels, ki67_val_labels, ki67_test_labels = [], [], []
    add_paths_for_slides(ki67_train_slide_ids, ki67_image_dict, ki67_label_dict, ki67_train_images, ki67_train_labels)
    add_paths_for_slides(ki67_val_slide_ids,   ki67_image_dict, ki67_label_dict, ki67_val_images, ki67_val_labels)
    add_paths_for_slides(ki67_test_slide_ids,  ki67_image_dict, ki67_label_dict, ki67_test_images, ki67_test_labels)

    # ============================
    # PART C: Combine STO + ID + Ki67
    # ============================
    final_train_images = sto_train_images + id_train_images + ki67_train_images
    final_val_images   = sto_val_images   + id_val_images   + ki67_val_images
    final_test_images  = sto_test_images  + id_test_images  + ki67_test_images

    final_train_labels = sto_train_labels + id_train_labels + ki67_train_labels
    final_val_labels   = sto_val_labels   + id_val_labels   + ki67_val_labels
    final_test_labels  = sto_test_labels  + id_test_labels  + ki67_test_labels

    return {
        "train_images": final_train_images,
        "val_images":   final_val_images,
        "test_images":  final_test_images,
        "train_labels": final_train_labels,
        "val_labels":   final_val_labels,
        "test_labels":  final_test_labels
    }


def main():
    splits_data = create_dataset_splits("/cluster/home/selmahi/datasets/250325_mib1_selma_4096_ds2_5x_sematic_seg_tumor")

    sto_pattern = re.compile(r"sto(.*?)d2x", re.IGNORECASE)
    id_pattern = re.compile(r"(?i)(?:auto)?id(\d+)")
    ki67_pattern = re.compile(r"(?i)(?:auto)?(\d+)ki67")


    def extract_sto_slide_id(filename):
        match = sto_pattern.search(filename) 
        if match:
            return match.group(1)
        return None

    def extract_id_slide_id_key(filename):
        match = id_pattern.search(filename) 
        if match:
            digits = match.group(1)
            if digits is not None and digits.isdigit():
                 return "ID_" + digits 
        return None


    def extract_ki67_slide_id_key(filename):
        match = ki67_pattern.search(filename)
        if match:
            num = match.group(1)
            fname_lower = filename.lower()
            if "ki67train" in fname_lower:
                return f"ki67_train_{num}"
            elif "ki67test" in fname_lower:
                return f"ki67_test_{num}"
            else:
                return f"ki67_{num}"
        return None

    # --- Collect all image files from the splits ---
    all_image_files = []
    all_image_files.extend(splits_data["train_images"])
    all_image_files.extend(splits_data["val_images"])
    all_image_files.extend(splits_data["test_images"])

    # --- Determine unique slide IDs for each category from the output files ---
    sto_unique_slide_ids = set()
    id_type_unique_slide_ids = set()
    ki67_unique_slide_ids = set()

    for file_path in all_image_files:
        filename = os.path.basename(file_path)

        # Check for STO type
        sto_id = extract_sto_slide_id(filename)
        if sto_id:
            sto_unique_slide_ids.add(sto_id)
            continue

        # Check for Ki67 type
        ki67_id_key = extract_ki67_slide_id_key(filename)
        if ki67_id_key:
            ki67_unique_slide_ids.add(ki67_id_key)
            continue

        # Check for ID type
        id_slide_key = extract_id_slide_id_key(filename)
        if id_slide_key:
            id_type_unique_slide_ids.add(id_slide_key)

    print("--- Unique Slide Counts (derived from output file lists) ---")
    print(f"Number of unique STO slides: {len(sto_unique_slide_ids)}")
    print(f"Number of unique ID slides: {len(id_type_unique_slide_ids)}")
    print(f"Number of unique Ki67 (Acrobat) slides: {len(ki67_unique_slide_ids)}")
    total_slides = len(sto_unique_slide_ids) + len(id_type_unique_slide_ids) + len(ki67_unique_slide_ids)
    print(f"Total unique slides represented in splits: {total_slides}")

    print("\n--- Patch Counts (from returned lists) ---")
    num_train_images = len(splits_data["train_images"])
    num_val_images   = len(splits_data["val_images"])
    num_test_images  = len(splits_data["test_images"])
    total_images = num_train_images + num_val_images + num_test_images

    num_train_labels = len(splits_data["train_labels"])
    num_val_labels   = len(splits_data["val_labels"])
    num_test_labels  = len(splits_data["test_labels"])
    total_labels = num_train_labels + num_val_labels + num_test_labels

    print(f"Train image patches: {num_train_images}")
    print(f"Validation image patches: {num_val_images}")
    print(f"Test image patches: {num_test_images}")
    print(f"Total image patches: {total_images}")

    print(f"\nTrain label patches: {num_train_labels}")
    print(f"Validation label patches: {num_val_labels}")
    print(f"Test label patches: {num_test_labels}")
    print(f"Total label patches: {total_labels}")

    if total_images == total_labels:
        print("\nNote: Total image patches match total label patches.")
    else:
        print(f"\nWARNING: Mismatch! Total image patches ({total_images}) vs Total label patches ({total_labels}).")

if __name__ == "__main__":
    main()