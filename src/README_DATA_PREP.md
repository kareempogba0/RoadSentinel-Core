# Car Accident Detection - Data Preparation

This directory contains scripts for data cleaning and exploratory data analysis (EDA) for the car accident detection dataset.

## Files Created

### 1. `data_cleaning.py`

A comprehensive data cleaning script that:

- ✅ Checks for corrupted images (0 byte files)
- ✅ Identifies missing label files
- ✅ Creates empty label files for images without labels (background images)
- ✅ Provides detailed summary reports
- ✅ **Automatically saves results to timestamped report files**

**Usage:**

```bash
python data_cleaning.py
```

**Output:**

- Console output with real-time progress
- Detailed report saved to `data_cleaning_reports/data_cleaning_report_YYYYMMDD_HHMMSS.txt`

**Report includes:**

- Timestamp and dataset information
- Processing details for each split (train, valid, test)
- Summary statistics
- Complete list of corrupted files (if any)
- Complete list of missing label files (if any)

**Results from latest run:**

- Total images processed: **620**
- Valid images with labels: **620**
- Missing labels created: **0** (all fixed from previous run)
- Corrupted files deleted: **0**

### 2. `notebooks/EDA.ipynb`

A comprehensive Jupyter notebook for exploratory data analysis that includes:

#### Analysis Sections:

1. **Dataset Overview** - Configuration and statistics
2. **Class Distribution** - Accident vs non-accident images
3. **Visualizations**:
   - Stacked bar chart showing distribution across splits
   - Pie chart for overall composition
   - Bounding box size distributions
   - Position heatmaps
4. **Bounding Box Analysis** - Width, height, area statistics
5. **Image Dimension Analysis** - Sample-based dimension checks
6. **Summary & Insights** - Actionable recommendations

#### Key Features:

- 📊 Multiple professional visualizations
- 📈 Statistical analysis of bounding boxes
- 🎯 Data quality checks
- 💡 Actionable insights for model training

**Usage:**

```bash
# Navigate to the src directory
cd f:\Abdelrahman\CS415\src

# Launch Jupyter Notebook
jupyter notebook notebooks/EDA.ipynb
```

## Dataset Structure

```
data/car-accident-detection-1/
├── train/
│   ├── images/    (434 images)
│   └── labels/    (434 labels)
├── valid/
│   ├── images/    (124 images)
│   └── labels/    (124 labels)
├── test/
│   ├── images/    (62 images)
│   └── labels/    (62 labels)
└── data.yaml      (Dataset configuration)
```

## Requirements Met

### ✅ Step 2: Data Cleaning Script

- **Requirement**: Check for images with 0 byte size or missing label files
- **Implementation**: `data_cleaning.py`
- **Features**:
  - Detects and removes corrupted (0 byte) images
  - Identifies missing label files
  - Creates empty label files for background images
  - Comprehensive logging and summary reports
  - Processes all splits (train, valid, test)
  - **Saves detailed reports to timestamped files**

### ✅ Step 3: EDA Report

- **Requirement**: Class distribution visualization (instead of correlation matrix for images)
- **Implementation**: `notebooks/EDA.ipynb`
- **Features**:
  - Class distribution bar charts and pie charts
  - Bounding box analysis (width, height, area, position)
  - Image dimension statistics
  - Data quality insights
  - Professional visualizations with seaborn/matplotlib

## Output Files

### Data Cleaning Reports

All data cleaning runs automatically generate timestamped reports in:

```
data_cleaning_reports/
└── data_cleaning_report_YYYYMMDD_HHMMSS.txt
```

Each report contains:

- Full processing log
- Summary statistics for each split
- Overall summary across all splits
- Detailed lists of any issues found

## Next Steps

1. **Run the EDA notebook** to visualize the data:

   ```bash
   jupyter notebook notebooks/EDA.ipynb
   ```

2. **Review the cleaning reports** in `data_cleaning_reports/` to verify data quality

3. **Review the insights** from the EDA to inform your model training strategy

4. **Proceed with model training** using the cleaned dataset

## Notes

- The dataset has **1 class** (Car Accident Detection)
- Images use YOLO format labels (class_id, center_x, center_y, width, height)
- All coordinates are normalized (0-1)
- Empty label files indicate background/non-accident images
- All cleaning operations are logged to both console and report files
