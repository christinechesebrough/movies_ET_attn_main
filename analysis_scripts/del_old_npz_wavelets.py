import os, re, sys
import glob

machine_path = 'media/christine'

sys.path.insert(0, f'/{machine_path}/Samsung/EPIPE-movie_nwb/Python')
sys.path.insert(0, f'/{machine_path}/Samsung/iEEG2NWB-main')

#vids = ['inscapes','despicable_me_english']#,'despicable_me_english']
vids = ['despicable_me_hungarian','despicable_me_english','inscapes']#,'despicable_me_english']#,'despicable_me_english']

drive = 'Samsung'
ref = 'avg'
region = 'all'

output = 'wavelet'
pow_type = 'raw'

tf_freq_range = (1, 150) 

vids = ['despicable_me_hungarian','despicable_me_english','inscapes']

for vid in vids:
    fig_dir = f'/{machine_path}/Data/Movie_data/{output}_{vid}_all_cortContacts_tf_10s_{tf_freq_range}_26Aug26'
    
    if os.path.exists(fig_dir):
        # Find all .npz files nested inside any subdirectory of fig_dir
        npz_files = glob.glob(os.path.join(fig_dir, '*', '*.npz'))
        
        for file_path in npz_files:
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")
#            print(file_path)
    else:
        print(f"Directory not found: {fig_dir}")
