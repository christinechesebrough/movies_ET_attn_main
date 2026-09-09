# Comparison: `compute_eye_measures.py` vs `prePCA_agg_norm.py`

## Overview

Both scripts compute derived eye movement measures from preprocessed gaze data, but they serve different purposes in the analysis pipeline.

## Key Differences

### 1. **Primary Purpose**

**`compute_eye_measures.py`**:
- Computes individual eye movement measures per patient
- Saves updated `.npz` files with additional computed measures
- Allows selective computation via flags (`compute_verg`, `compute_pupil`, etc.)
- **Output**: Updated `.npz` files per patient

**`prePCA_agg_norm.py`**:
- Aggregates all measures across patients into a single DataFrame
- Normalizes measures for PCA analysis
- Always computes all measures (no selective flags)
- **Output**: Combined CSV files (raw and normalized) for all patients

### 2. **Data Structure & Output**

| Aspect | `compute_eye_measures.py` | `prePCA_agg_norm.py` |
|--------|-------------------------|---------------------|
| **Output Format** | Individual `.npz` files | Combined CSV files |
| **Data Organization** | Per-patient files | All patients in one DataFrame |
| **Normalization** | None | Z-score normalization |
| **Aggregation** | No | Yes (all patients combined) |

### 3. **Computed Measures**

Both scripts compute similar measures, but with some differences:

#### **Common Measures**:
- ✅ Rolling vergence (mean, std, absolute)
- ✅ Rolling saccade rate
- ✅ Rolling saccade dispersion (mean, std)
- ✅ Rolling blink rate
- ✅ Rolling blink duration
- ✅ Rolling pupil (mean, std)
- ✅ ISC (loaded from pre-computed files)

#### **Differences**:

**`compute_eye_measures.py`**:
- Computes `saccade_dispersions_dist` from `saccade_pos` (Euclidean distance between xy pairs)
- Saves `fixation_int` and `saccade_onset_int` to npz files
- Has conditional computation flags

**`prePCA_agg_norm.py`**:
- Only computes saccade dispersion from saccade-to-fixation distance
- Always computes all measures
- Aggregates into DataFrame format

### 4. **Vergence Calculation**

**Critical Difference**:

```python
# compute_eye_measures.py
vergence = verg_dat['vis_fd_interp'].values  # Visual focus displacement

# prePCA_agg_norm.py  
vergence = verg_dat['dva_gaze_disp_x_interp']  # Gaze disparity in DVA
```

**Impact**: These are different measures! `vis_fd_interp` is visual focus displacement, while `dva_gaze_disp_x_interp` is gaze disparity in degrees of visual angle.

### 5. **Rolling Window Parameters**

Both use identical parameters:
- **Window size**: 10 seconds (`window_samples = 10*fs_eye`)
- **Overlap**: 7.5 seconds (`overlap_samples = 7.5*fs_eye`)
- **Step size**: 2.5 seconds (`step_size_samples = window_samples - overlap_samples`)
- **Alignment**: Aligned to ISC time (`isc_aligned_time`)

### 6. **Data Alignment & Timing**

Both scripts:
- Cut data around movie times (`t_start` to `t_end` from `t_pupil`)
- Zero out time vectors (`t = t - t[0]`)
- Align saccade/fixation/blink times by subtracting `t_start`
- Verify timing alignment between vergence and gaze data

### 7. **Saccade Dispersion Calculation**

**Both scripts** compute dispersion as Euclidean distance between:
- Saccade onset position
- Fixation onset position

**Additional in `compute_eye_measures.py`**:
- Also computes `saccade_dispersions_dist` from `saccade_pos` array
- Extracts xy pairs from each row: `xy1 = row[0:2]`, `xy2 = row[2:4]`

### 8. **Pupil Data Processing**

Both scripts:
- Handle different sampling rates between pupil and gaze data
- Compute rolling windows aligned to `isc_aligned_time`
- Calculate mean and std within each window

**Difference**: `prePCA_agg_norm.py` includes pupil in normalization, while `compute_eye_measures.py` doesn't normalize.

### 9. **Code Structure**

**`compute_eye_measures.py`**:
- Conditional computation blocks (`if compute_verg:`, `if compute_sacc:`, etc.)
- Saves updated npz files per patient
- More modular, allows selective computation

**`prePCA_agg_norm.py`**:
- Always computes all measures
- Aggregates into dictionaries (`all_subs_*`)
- Creates DataFrame and normalizes
- Prepares data for PCA

## Recommendations for Standardization

### 1. **Unify Vergence Measure**
**Decision needed**: Which vergence measure should be standard?
- `vis_fd_interp` (visual focus displacement) - used in `compute_eye_measures.py`
- `dva_gaze_disp_x_interp` (gaze disparity in DVA) - used in `prePCA_agg_norm.py`

**Recommendation**: Use `vis_fd_interp` as it's the more comprehensive measure of vergence.

### 2. **Standardize Output Format**
**Option A**: Keep both formats
- Use `compute_eye_measures.py` for per-patient analysis
- Use `prePCA_agg_norm.py` for group-level analysis

**Option B**: Create unified script
- Compute measures per patient (like `compute_eye_measures.py`)
- Optionally aggregate into DataFrame (like `prePCA_agg_norm.py`)
- Save both formats

### 3. **Unify Saccade Dispersion**
Both scripts compute dispersion similarly, but `compute_eye_measures.py` has an additional measure. **Recommendation**: Include both dispersion measures in standardized version.

### 4. **Standardize Normalization**
**Decision needed**: When should normalization occur?
- Per-patient (before aggregation)
- Across all patients (after aggregation)
- Both (with flags)

**Recommendation**: Normalize per-patient first, then optionally normalize across patients for PCA.

### 5. **Create Unified Function Structure**

```python
def compute_rolling_measures(patient_data, window_params):
    """
    Compute all rolling eye movement measures for a single patient.
    
    Returns:
        dict with all computed measures
    """
    pass

def aggregate_patient_measures(all_patient_measures):
    """
    Aggregate measures from all patients into DataFrame.
    """
    pass

def normalize_measures(df, normalization_type='zscore'):
    """
    Normalize measures (per-patient or across patients).
    """
    pass
```

## Proposed Standardized Approach

1. **Single computation function** that computes all measures per patient
2. **Flexible output**: Save per-patient npz files AND/OR aggregated DataFrame
3. **Unified vergence measure**: Use `vis_fd_interp` consistently
4. **Standardized normalization**: Per-patient z-score, then optional across-patient normalization
5. **Consistent window parameters**: Use same rolling window approach
6. **Clear documentation**: Document which measures are computed and their units

## Questions to Resolve

1. Which vergence measure should be standard? (`vis_fd_interp` vs `dva_gaze_disp_x_interp`)
2. Should normalization happen per-patient or across patients?
3. Should both dispersion measures be included? (saccade-to-fixation distance AND saccade_pos distance)
4. Should the standardized script support selective computation (flags) or always compute all measures?
5. What should be the default output format? (npz files, CSV, or both?)


