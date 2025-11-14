"""
atlas_volume_analysis.py

Automated volumetric analysis of the King's College London
Spina Bifida Aperta fetal brain MRI atlas.

Author: Jawahar Sri Prakash Thiyagarajan
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind


# -------------------------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------------------------

# CHANGE THIS to the folder where the atlas is stored on your computer
DATA_DIR = r"E:\BRAIN_IIT\kings college london\SpinaBifidaAtlas_v2"

# Output folders (relative to repository root or current working directory)

# If running as script (.py), __file__ exists.
# If running in Jupyter / interactive, it does not -> use current working dir.
try:
    ROOT = Path(__file__).resolve().parents[1]  # repo root if script in /code
except NameError:
    ROOT = Path(os.getcwd())                    # Jupyter / interactive fallback

RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)


# -------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -------------------------------------------------------------------------

def list_case_folders(data_dir: str) -> list:
    """Return a sorted list of fetal atlas folders in the given directory."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir}")

    folders = [
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f))
    ]
    folders = sorted(folders)
    print(f"[INFO] Found {len(folders)} folders in atlas directory:")
    for f in folders:
        print("   ", f)
    return folders


def load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI file and return the data array."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    img = nib.load(path)
    return img.get_fdata()


def compute_region_and_total_volumes(data_dir: str, folders: list) -> pd.DataFrame:
    """
    For each folder, compute voxel counts per region from parcellation.nii.gz
    and return a long-format DataFrame with one row per region.
    """
    records = []

    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        parc_path = os.path.join(folder_path, "parcellation.nii.gz")

        if not os.path.exists(parc_path):
            print(f"[WARN] Skipping {folder} (no parcellation.nii.gz)")
            continue

        parc_data = load_nifti(parc_path)

        # Unique labels; exclude 0 (background)
        region_labels = np.unique(parc_data)
        region_labels = region_labels[region_labels != 0]

        # Extract gestational age from folder name (e.g., GA21 -> 21)
        ga_digits = "".join(c for c in folder if c.isdigit())
        try:
            ga = int(ga_digits)
        except ValueError:
            ga = np.nan

        operated_flag = "operated" in folder.lower()

        for label in region_labels:
            count = int(np.sum(parc_data == label))
            rec = {
                "Folder": folder,
                "Gestational_Age": ga,
                "Operated": operated_flag,
                "Region_Label": int(label),
                "Voxel_Count": count,
            }
            records.append(rec)

    if not records:
        raise RuntimeError("No region volumes computed. Check atlas structure.")

    df = pd.DataFrame(records)
    return df


def plot_central_slices(data_dir: str, folder: str) -> None:
    """
    Plot central slices of srr, mask and parcellation for one folder
    and save the figure as PNG.
    """
    folder_path = os.path.join(data_dir, folder)
    srr_path = os.path.join(folder_path, "srr.nii.gz")
    mask_path = os.path.join(folder_path, "mask.nii.gz")
    parc_path = os.path.join(folder_path, "parcellation.nii.gz")

    srr = load_nifti(srr_path)
    mask = load_nifti(mask_path)
    parc = load_nifti(parc_path)

    central_slice = srr.shape[2] // 2

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(srr[:, :, central_slice], cmap="gray")
    plt.title("Super-Resolution MRI")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask[:, :, central_slice], cmap="gray")
    plt.title("Brain Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(parc[:, :, central_slice], cmap="nipy_spectral")
    plt.title("Parcellation")
    plt.axis("off")

    out_path = FIG_DIR / f"central_slices_{folder}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Saved central slices figure -> {out_path}")


def summarize_total_volumes(df_regions: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse region-wise volumes to total brain volume per folder.
    """
    total = (
        df_regions
        .groupby(["Folder", "Gestational_Age", "Operated"])["Voxel_Count"]
        .sum()
        .reset_index()
        .rename(columns={"Voxel_Count": "Total_Brain_Volume"})
    )
    return total


def stats_and_plots(total_df: pd.DataFrame) -> None:
    """
    Compute statistics, correlation, regression and plots.
    Saves plots to FIG_DIR and prints stats.
    """
    total_df = total_df.dropna(subset=["Gestational_Age", "Total_Brain_Volume"])
    total_df = total_df.sort_values("Gestational_Age")

    x = total_df["Gestational_Age"].to_numpy()
    y = total_df["Total_Brain_Volume"].to_numpy()

    if len(x) < 2:
        print("[WARN] Not enough cases for statistics.")
        return

    mean_vol = y.mean()
    std_vol = y.std(ddof=1)
    corr, p_corr = pearsonr(x, y)

    print("\n[STATS] Summary")
    print(f"Mean brain volume        : {mean_vol:.2f} voxels")
    print(f"Std. dev. brain volume   : {std_vol:.2f} voxels")
    print(f"Pearson r (GA vs volume) : {corr:.2f} (p = {p_corr:.4e})")

    # Linear regression (1st-degree polynomial)
    slope, intercept = np.polyfit(x, y, 1)
    fit_line = slope * x + intercept
    print(f"Slope (weekly increase)  : {slope:.2f} voxels/week")

    # Growth between consecutive cases (sorted by GA)
    growth = np.diff(y)
    if len(growth) > 0:
        mean_growth = growth.mean()
        print(f"Mean Δvolume per step    : {mean_growth:.2f} voxels")

    # Plot: brain volume vs GA
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label="Cases")
    plt.plot(x, fit_line, color="red", label=f"Linear fit (slope={slope:.1f})")
    plt.xlabel("Gestational Age (weeks)")
    plt.ylabel("Total Brain Volume (voxels)")
    plt.title("Brain Volume vs Gestational Age")
    plt.legend()
    plt.tight_layout()
    out_path = FIG_DIR / "brain_volume_vs_age.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Saved volume vs age plot -> {out_path}")

    # Histogram of volumes
    plt.figure(figsize=(7, 5))
    plt.hist(y, bins=8)
    plt.xlabel("Total Brain Volume (voxels)")
    plt.ylabel("Number of Cases")
    plt.title("Distribution of Fetal Brain Volumes")
    plt.tight_layout()
    out_path = FIG_DIR / "brain_volume_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Saved volume distribution plot -> {out_path}")

    # Outlier detection: |z| > 2
    z_scores = (y - mean_vol) / std_vol
    outlier_mask = np.abs(z_scores) > 2
    outliers = total_df[outlier_mask]

    if outliers.empty:
        print("[INFO] No outliers detected with |z| > 2.")
    else:
        print("[WARN] Outliers detected (|z| > 2):")
        print(outliers[["Folder", "Gestational_Age", "Total_Brain_Volume"]])

    # Operated vs not-operated comparison (if both groups exist)
    operated = total_df[total_df["Operated"]]["Total_Brain_Volume"].to_numpy()
    not_operated = total_df[~total_df["Operated"]]["Total_Brain_Volume"].to_numpy()

    if len(operated) > 1 and len(not_operated) > 1:
        t_stat, p_val = ttest_ind(operated, not_operated, equal_var=False)
        print("\n[STATS] Operated vs Not-operated (Welch t-test)")
        print(f"Operated mean      : {operated.mean():.2f}")
        print(f"Not-operated mean  : {not_operated.mean():.2f}")
        print(f"t = {t_stat:.2f}, p = {p_val:.4f}")
    else:
        print("\n[INFO] Not enough not-operated and operated cases for t-test.")


# -------------------------------------------------------------------------
# 3. MAIN
# -------------------------------------------------------------------------

def main():
    print(f"[INFO] Using data directory: {DATA_DIR}")

    folders = list_case_folders(DATA_DIR)
    if not folders:
        print("[ERROR] No folders found. Check DATA_DIR path and contents.")
        return

    # Pick a folder for example visualization (prefer GA21 if present)
    example_folder = None
    for f in folders:
        if "GA21" in f:
            example_folder = f
            break
    if example_folder is None:
        example_folder = folders[0]

    print(f"[INFO] Plotting central slices for: {example_folder}")
    try:
        plot_central_slices(DATA_DIR, example_folder)
    except Exception as e:
        print(f"[WARN] Could not plot central slices for {example_folder}: {e}")

    print("[INFO] Computing region and total volumes...")
    df_regions = compute_region_and_total_volumes(DATA_DIR, folders)

    # Save region-wise CSV
    region_csv = RESULTS_DIR / "spina_bifida_region_volumes.csv"
    df_regions.to_csv(region_csv, index=False)
    print(f"[OK] Saved region-wise volumes -> {region_csv}")

    # Total volume per case
    total_df = summarize_total_volumes(df_regions)
    total_csv = RESULTS_DIR / "spina_bifida_total_volumes.csv"
    total_df.to_csv(total_csv, index=False)
    print(f"[OK] Saved total brain volumes -> {total_csv}")

    # Statistics and plots
    stats_and_plots(total_df)

    print("\n[DONE] Analysis complete.")


if __name__ == "__main__":
    main()
