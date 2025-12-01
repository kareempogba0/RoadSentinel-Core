"""
Data Cleaning Script for Car Accident Detection Dataset
This script checks for corrupted images and missing label files.
Results are saved to a timestamped report file.
"""

import os
import glob
from pathlib import Path
import sys
from datetime import datetime

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Global variable to store output file handle
output_file = None


def log_message(message, print_to_console=True):
    """
    Log message to both console and output file
    
    Args:
        message (str): Message to log
        print_to_console (bool): Whether to print to console
    """
    global output_file
    if print_to_console:
        print(message)
    if output_file:
        output_file.write(message + '\n')


def clean_data(data_path):
    """
    Clean the dataset by checking for:
    1. Empty/corrupted image files (0 bytes)
    2. Missing label files for images
    
    Args:
        data_path (str): Path to the dataset folder (e.g., './data/car-accident-detection-1/train')
    """
    # Find all image files recursively
    images = glob.glob(f"{data_path}/**/images/*.jpg", recursive=True)
    log_message(f"=" * 60)
    log_message(f"Starting Data Cleaning Process")
    log_message(f"=" * 60)
    log_message(f"Checking {len(images)} images in {data_path}...\n")
    
    corrupted_count = 0
    missing_label_count = 0
    valid_count = 0
    corrupted_files = []
    missing_label_files = []
    
    for img in images:
        img_path = Path(img)
        
        # Check 1: Is file empty or corrupted? (Outlier/Cleaning)
        file_size = os.path.getsize(img)
        if file_size == 0:
            log_message(f"[!] CORRUPTED: {img_path.name} (0 bytes) - DELETING")
            os.remove(img)
            corrupted_count += 1
            corrupted_files.append(str(img_path.name))
            continue
        
        # Check 2: Does label exist? (Missing Value)
        # Convert image path to corresponding label path
        label_path = str(img_path).replace("images", "labels").replace(".jpg", ".txt")
        
        if not os.path.exists(label_path):
            log_message(f"[!] MISSING LABEL: {img_path.name}")
            log_message(f"    Creating empty label file: {Path(label_path).name}")
            # Create empty label file (indicates background/no accident image)
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            open(label_path, 'a').close()
            missing_label_count += 1
            missing_label_files.append(str(img_path.name))
        else:
            valid_count += 1
    
    # Summary Report
    log_message("\n" + "=" * 60)
    log_message("Data Cleaning Summary")
    log_message("=" * 60)
    log_message(f"[+] Valid images with labels:     {valid_count}")
    log_message(f"[-] Corrupted files deleted:      {corrupted_count}")
    log_message(f"[*] Missing labels created:       {missing_label_count}")
    log_message(f"[#] Total images processed:       {len(images)}")
    log_message("=" * 60)
    log_message("[DONE] Data cleaning complete!\n")
    
    return {
        'valid': valid_count,
        'corrupted': corrupted_count,
        'missing_labels': missing_label_count,
        'total': len(images),
        'corrupted_files': corrupted_files,
        'missing_label_files': missing_label_files
    }


def clean_all_splits(base_path):
    """
    Clean all dataset splits (train, valid, test)
    
    Args:
        base_path (str): Base path to the dataset folder
    """
    splits = ['train', 'valid', 'test']
    all_results = {}
    
    log_message("\n" + ">" * 60)
    log_message("CLEANING ALL DATASET SPLITS")
    log_message(">" * 60 + "\n")
    
    for split in splits:
        split_path = os.path.join(base_path, split)
        if os.path.exists(split_path):
            log_message(f"\n[>>] Processing {split.upper()} split...")
            results = clean_data(split_path)
            all_results[split] = results
        else:
            log_message(f"[!] {split.upper()} split not found at {split_path}")
    
    # Overall Summary
    log_message("\n" + "=" * 60)
    log_message("OVERALL SUMMARY - ALL SPLITS")
    log_message("=" * 60)
    total_valid = sum(r['valid'] for r in all_results.values())
    total_corrupted = sum(r['corrupted'] for r in all_results.values())
    total_missing = sum(r['missing_labels'] for r in all_results.values())
    total_images = sum(r['total'] for r in all_results.values())
    
    log_message(f"[+] Total valid images:           {total_valid}")
    log_message(f"[-] Total corrupted files:        {total_corrupted}")
    log_message(f"[*] Total missing labels created: {total_missing}")
    log_message(f"[#] Total images processed:       {total_images}")
    log_message("=" * 60)
    
    # Detailed file lists
    log_message("\n" + "=" * 60)
    log_message("DETAILED FILE LISTS")
    log_message("=" * 60)
    
    # List corrupted files
    log_message("\n[CORRUPTED FILES]")
    if total_corrupted > 0:
        for split, results in all_results.items():
            if results['corrupted_files']:
                log_message(f"\n  {split.upper()} split ({len(results['corrupted_files'])} files):")
                for file in results['corrupted_files']:
                    log_message(f"    - {file}")
    else:
        log_message("  None found - All images are valid!")
    
    # List missing label files
    log_message("\n[MISSING LABEL FILES - Created Empty Labels]")
    if total_missing > 0:
        for split, results in all_results.items():
            if results['missing_label_files']:
                log_message(f"\n  {split.upper()} split ({len(results['missing_label_files'])} files):")
                for file in results['missing_label_files']:
                    log_message(f"    - {file}")
    else:
        log_message("  None found - All images have labels!")
    
    log_message("\n" + "=" * 60)
    
    return all_results




if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = "./data_cleaning_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"data_cleaning_report_{timestamp}.txt")
    
    # Open output file
    output_file = open(output_filename, 'w', encoding='utf-8')
    
    # Write header to output file
    log_message("=" * 60, print_to_console=False)
    log_message("DATA CLEANING REPORT", print_to_console=False)
    log_message("Car Accident Detection Dataset", print_to_console=False)
    log_message("=" * 60, print_to_console=False)
    log_message(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", print_to_console=False)
    log_message("=" * 60, print_to_console=False)
    log_message("", print_to_console=False)
    
    # Point this to your downloaded dataset folder
    dataset_base_path = "./data/car-accident-detection-1"
    
    try:
        # Clean all splits (train, valid, test)
        results = clean_all_splits(dataset_base_path)
        
        # Write footer
        log_message("\n" + "=" * 60)
        log_message(f"Report saved to: {output_filename}")
        log_message("=" * 60)
        
    finally:
        # Close output file
        if output_file:
            output_file.close()
    
    print(f"\n[SUCCESS] Detailed report saved to: {output_filename}")

