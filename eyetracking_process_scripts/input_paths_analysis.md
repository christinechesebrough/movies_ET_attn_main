# Input Paths Analysis: Unique File Paths Per Patient

## Summary

**Total Unique Input Files Per Patient: 4**
1. Eye prep npz file (`*_et_prep.npz`)
2. Vergence CSV file (`*_et_prep.csv`)
3. Blink events CSV file (`*_blink_events.csv`)
4. ISC data npz file (loaded once, not per patient)

## Redundancies Found

### `compute_eye_measures.py`

**Issue 1: Eye prep npz loaded TWICE (lines 97-99)**
```python
eye_data = np.load('{:s}/{:s}'.format(eye_pat_dir, eye_file))
eye_file = eye_list[0]  # Redundant assignment
eye_data = np.load('{:s}/{:s}'.format(eye_pat_dir, eye_file))  # Redundant load
```

**Issue 2: Eye prep npz loaded AGAIN later (lines 265-268)**
```python
et_files = os.listdir('{:s}/Eye_prep/'.format(pat_dir))
et_file = [file for file in et_files if f'et_prep.npz' in file][0]
et_data_path = '{:s}/Eye_prep/{:s}'.format(pat_dir, et_file)
et_data = np.load(et_data_path)  # Same file loaded again!
```

**Total**: Eye prep npz file is loaded **3 times** for the same patient!

### `prePCA_agg_norm.py`

**Issue 1: Eye prep npz loaded TWICE (lines 141 and 236)**
```python
# First load (line 141)
eye_data = np.load('{:s}/{:s}'.format(eye_pat_dir, eye_file))

# Later, loaded again with different variable name (line 236)
et_data = np.load(et_file)  # Same file, different variable!
```

**Total**: Eye prep npz file is loaded **2 times** for the same patient!

## Detailed Path Analysis

### 1. Eye Prep NPZ File (`*_et_prep.npz`)

**Purpose**: Contains gaze data, pupil data, saccade/fixation timings

**In `compute_eye_measures.py`**:
- **Line 92-99**: Loaded as `eye_data` (with duplicate load)
- **Line 265-268**: Loaded again as `et_data` (redundant!)

**In `prePCA_agg_norm.py`**:
- **Line 128-141**: Loaded as `eye_data` (with special cases for NS190, NS191)
- **Line 210-236**: Loaded again as `et_data` (redundant!)

**Search Patterns**:
- `compute_eye_measures.py`: Filters by `'_et_prep' in f` then `vid in f`
- `prePCA_agg_norm.py`: Uses `glob.glob` with pattern `*{vid}*et_prep.npz` OR special hardcoded paths for NS190/NS191

### 2. Vergence CSV File (`*_et_prep.csv`)

**Purpose**: Contains vergence measures (should use `vis_fd_interp`)

**In `compute_eye_measures.py`**:
- **Line 111-118**: Loaded once as `verg_dat`
- **Search pattern**: Different for `despicable_me_english` vs `inscapes`

**In `prePCA_agg_norm.py`**:
- **Line 151-188**: Loaded once as `verg_dat`
- **Search pattern**: Uses list comprehension with different patterns per video
- **Special cases**: Hardcoded paths for NS190/NS191

**Current Issue**: 
- `compute_eye_measures.py` uses `vis_fd_interp` ✅ (CORRECT)
- `prePCA_agg_norm.py` uses `dva_gaze_disp_x_interp` ❌ (WRONG - should be `vis_fd_interp`)

### 3. Blink Events CSV File (`*_blink_events.csv`)

**Purpose**: Contains blink onset, end, and duration

**In `compute_eye_measures.py`**:
- **Line 143-146**: Loaded once as `blink_data`
- **Search pattern**: `f'{vid}_blink_events.csv' in f`

**In `prePCA_agg_norm.py`**:
- **Line 258-261**: Loaded once as `blink_data`
- **Search pattern**: `f'{vid}_blink_events.csv' in f`

**Status**: ✅ No redundancy, consistent across scripts

### 4. ISC Data NPZ File

**Purpose**: Pre-computed ISC data

**In `compute_eye_measures.py`**:
- **Line 54-57**: Loaded once at script start (not per patient)
- Different paths for different videos

**In `prePCA_agg_norm.py`**:
- **Line 73-78**: Loaded once at script start (not per patient)
- Different paths for different videos

**Status**: ✅ No redundancy, loaded once per script run

## Path Construction Issues

### Inconsistent Directory Construction

**`compute_eye_measures.py`**:
```python
# Line 92: Uses string formatting
eye_pat_dir = '{:s}/{:s}/Eye_prep'.format(data_dir, pat)

# Line 110: Uses string formatting
pat_dir = '{:s}/{:s}'.format(data_dir, pat)

# Line 76: Uses os.path.join
pat_dir = os.path.join(data_dir, pat)
```

**`prePCA_agg_norm.py`**:
```python
# Line 114: Uses os.path.join
pat_dir = os.path.join(data_dir, pat)

# Line 129: Uses string formatting
eye_pat_dir = '{:s}/{:s}/Eye_prep'.format(mne_data_dir, pat)

# Line 152: Uses string formatting
pat_dir = '{:s}/{:s}'.format(data_dir, pat)

# Line 212: Uses string formatting
sub_et_pat_dir = '{:s}/Eye_prep'.format(pat_dir)
```

**Issue**: Mix of `os.path.join()` and string formatting makes paths inconsistent and harder to maintain.

## Recommendations

### 1. **Standardize to `vis_fd_interp`**
- ✅ Confirmed: `vis_fd_interp` should be the standard vergence measure
- Fix `prePCA_agg_norm.py` line 190 to use `vis_fd_interp` instead of `dva_gaze_disp_x_interp`

### 2. **Eliminate Redundant File Loads**

**For `compute_eye_measures.py`**:
- Remove duplicate load on lines 98-99
- Reuse `eye_data` instead of loading again as `et_data` on lines 265-268

**For `prePCA_agg_norm.py`**:
- Reuse `eye_data` instead of loading again as `et_data` on line 236

### 3. **Standardize Path Construction**

Use `os.path.join()` consistently:
```python
# Standard pattern
pat_dir = os.path.join(data_dir, pat)
eye_prep_dir = os.path.join(pat_dir, 'Eye_prep')
```

### 4. **Create Unified File Loading Function**

```python
def load_patient_eye_data(pat_dir, vid, pat=None):
    """
    Load all eye tracking data files for a patient.
    
    Returns:
        dict with keys: 'eye_data', 'verg_data', 'blink_data'
    """
    eye_prep_dir = os.path.join(pat_dir, 'Eye_prep')
    
    # Load eye prep npz (only once!)
    eye_files = [f for f in os.listdir(eye_prep_dir) 
                 if '_et_prep.npz' in f and vid in f]
    eye_file = eye_files[0]
    eye_data = np.load(os.path.join(eye_prep_dir, eye_file))
    
    # Load vergence CSV
    verg_files = [f for f in os.listdir(eye_prep_dir)
                  if vid in f and f.endswith('_et_prep.csv')]
    verg_file = verg_files[0]
    verg_data = pd.read_csv(os.path.join(eye_prep_dir, verg_file))
    
    # Load blink events CSV
    blink_files = [f for f in os.listdir(eye_prep_dir)
                   if f'{vid}_blink_events.csv' in f]
    blink_file = blink_files[0]
    blink_data = pd.read_csv(os.path.join(eye_prep_dir, blink_file))
    
    return {
        'eye_data': eye_data,
        'verg_data': verg_data,
        'blink_data': blink_data
    }
```

## Summary of Unique Input Paths

| File Type | Purpose | Loaded Per Patient? | Redundancies |
|-----------|---------|---------------------|--------------|
| Eye prep npz | Gaze, pupil, saccades | Yes | ❌ 2-3 times |
| Vergence CSV | Vergence measures | Yes | ✅ Once (but wrong column) |
| Blink CSV | Blink events | Yes | ✅ Once |
| ISC npz | Pre-computed ISC | No (once per script) | ✅ Once |

**Total unique inputs per patient: 3** (excluding ISC which is loaded once)


