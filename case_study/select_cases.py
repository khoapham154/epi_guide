import pandas as pd
import json
import os
import shutil

df = pd.read_csv('external_data/MME/classification_gold.csv')

target_ids = [
    'PMC9302225__Patient_2',
    'PMC9302225__Patient_4',
    'PMC8594770__P7',
    'PMC6029564__Patient_18',
    'PMC6029564__Patient_9',
    'PMC6029564__Patient_19',
    'PMC10791031__15',
    'PMC6331207__35y_F_Pilomotor',
    'PMC10627632__Patient 5',
    'PMC6015084__Patient_10'
]

selected_indices = []
for tid in target_ids:
    idx = df.index[df['patient_id'] == tid].tolist()
    if idx:
        selected_indices.append(idx[0])
    else:
        print(f"Warning: Could not find patient {tid}")

selected_df = df.loc[selected_indices].copy()

# Add 5 random high-quality cases that have both MRI and EEG if possible
remaining_df = df.drop(selected_indices)
has_mri = remaining_df['mri_images'].notna() & (remaining_df['mri_images'] != '[]')
has_eeg = remaining_df['eeg_images'].notna() & (remaining_df['eeg_images'] != '[]')

new_cases_df = remaining_df[has_mri & has_eeg].head(5)
if len(new_cases_df) < 5:
    needed = 5 - len(new_cases_df)
    extra = remaining_df[has_mri | has_eeg].head(needed)
    new_cases_df = pd.concat([new_cases_df, extra])

final_df = pd.concat([selected_df, new_cases_df]).reset_index(drop=True)

# Assign groups based on the prompt "Try to make categories A, B (Group) like this comprehensive as well"
groups = []
for i in range(len(final_df)):
    if i < 2:
        groups.append('A - Discriminative Classifiers Overriding Text Hallucination')
    elif i < 5: # A-3, B-1, B-2
        groups.append('B - Multi-Modal Fusion with Post-Surgical MRI')
    elif i < 7: # C-1, C-2
        groups.append('C - Surgery Outcome via Discriminative Ensemble')
    elif i < 10: # D-1, D-2, D-3
        groups.append('D - Rare/Complex Syndromes Correctly Identified')
    else:
        groups.append('E - Comprehensive Novel Cases (Multi-Modal Fusion Analysis)')
final_df['Group'] = groups

print(f"Selected {len(final_df)} cases.")

# Save to CSV
out_dir = 'case_study'
out_csv = os.path.join(out_dir, 'comprehensive_15_cases.csv')
columns_for_clinicians = ['Group', 'pmc_id', 'patient_id', 'age', 'sex', 'semiology_text', 'mri_report_text', 'eeg_report_text', 'epilepsy_type_label', 'seizure_type_label', 'ez_localization_label', 'aed_response_label', 'surgery_outcome_label']
final_df[columns_for_clinicians].to_csv(out_csv, index=False)
print(f"Saved dataset to {out_csv}")

# Copy images
mri_dir = os.path.join(out_dir, 'mri')
eeg_dir = os.path.join(out_dir, 'eeg')
os.makedirs(mri_dir, exist_ok=True)
os.makedirs(eeg_dir, exist_ok=True)

def copy_images(images_str, pmc, pat, target_dir, prefix_modality):
    if pd.isna(images_str) or images_str == '[]':
        return
    try:
        images = eval(images_str) # the str is a python repr of list of dicts or standard json
    except:
        try:
            images = json.loads(images_str.replace("'", '"'))
        except:
            return
            
    if not isinstance(images, list):
        return
        
    for i, img in enumerate(images):
        src_path = img.get('path')
        if src_path and os.path.exists(src_path):
            ext = os.path.splitext(src_path)[1]
            pat_clean = str(pat).replace(' ', '')
            dst_name = f"{pmc}_{pat_clean}_{prefix_modality}_{i}{ext}"
            dst_path = os.path.join(target_dir, dst_name)
            shutil.copy2(src_path, dst_path)

for _, row in final_df.iterrows():
    pmc = row['pmc_id']
    pat = row['patient_id'] if pd.notna(row['patient_id']) else 'Unknown'
    copy_images(row['mri_images'], pmc, pat, mri_dir, 'MRI')
    copy_images(row['eeg_images'], pmc, pat, eeg_dir, 'EEG')

print("Image extraction completed.")
