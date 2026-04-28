"""
Step 2: EEG Preprocessing Pipeline using MNE
Applies signal filtering, re-referencing, and artifact handling
"""

import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def load_trial_to_mne(trial_id, trial_metadata, channel_info):
    """Load a single trial into an MNE Raw object."""
    trial_info = trial_metadata[trial_metadata['trial_id'] == trial_id].iloc[0]
    trial_idx = trial_metadata[trial_metadata['trial_id'] == trial_id].index[0]

    data = np.load(f'data/raw/eeg_trial_{trial_idx:04d}.npy')

    ch_names = channel_info['channel_name'].tolist()
    ch_types = ['eeg'] * len(ch_names)
    sfreq = trial_info['sampling_rate']

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore', verbose=False)

    return raw, trial_info


def apply_bandpass_filter(raw, l_freq=0.5, h_freq=50.0):
    """Apply bandpass filter to remove drift and high-frequency noise."""
    raw_filtered = raw.copy()
    raw_filtered.filter(l_freq=l_freq, h_freq=h_freq,
                        fir_design='firwin', verbose=False)
    return raw_filtered


def apply_notch_filter(raw, freqs=[50, 60]):
    """Apply notch filter to remove powerline noise."""
    raw_notched = raw.copy()
    raw_notched.notch_filter(freqs=freqs, verbose=False)
    return raw_notched


def apply_rereferencing(raw, ref_type='average'):
    """Apply re-referencing scheme."""
    raw_reref = raw.copy()
    raw_reref.set_eeg_reference('average', projection=True, verbose=False)
    raw_reref.apply_proj(verbose=False)
    return raw_reref


def detect_bad_channels(raw, threshold=3.0):
    """Detect bad channels using amplitude criteria."""
    data = raw.get_data()
    channel_stds = np.std(data, axis=1)
    z_scores = (channel_stds - np.mean(channel_stds)) / np.std(channel_stds)

    bad_channels = [raw.ch_names[i] for i in range(len(z_scores))
                    if abs(z_scores[i]) > threshold]
    return bad_channels


def apply_ica_artifact_removal(raw, n_components=15):
    """Apply ICA for eye blink and muscle artifact removal."""
    ica = ICA(n_components=n_components, random_state=42,
              max_iter=200, verbose=False)

    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    ica.fit(raw_for_ica, verbose=False)

    ica.exclude = list(range(min(2, n_components)))

    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)

    return raw_clean, ica


def preprocess_trial(trial_id, trial_metadata, channel_info,
                     apply_ica=False, verbose=True):
    """Complete preprocessing pipeline for a single trial."""
    if verbose:
        print(f"  Processing: {trial_id}")

    raw, trial_info = load_trial_to_mne(trial_id, trial_metadata, channel_info)

    preprocessing_log = {
        'trial_id': trial_id,
        'original_n_channels': len(raw.ch_names)
    }

    raw = apply_bandpass_filter(raw, l_freq=0.5, h_freq=50.0)
    preprocessing_log['bandpass_filter'] = '0.5-50 Hz'

    raw = apply_notch_filter(raw, freqs=[50, 60])
    preprocessing_log['notch_filter'] = '50, 60 Hz'

    bad_channels = detect_bad_channels(raw, threshold=3.0)
    raw.info['bads'] = bad_channels
    preprocessing_log['bad_channels'] = len(bad_channels)
    preprocessing_log['bad_channel_names'] = ','.join(bad_channels) if bad_channels else 'none'

    if len(bad_channels) > 0:
        raw.interpolate_bads(reset_bads=True, verbose=False)
    preprocessing_log['interpolated'] = len(bad_channels) > 0

    raw = apply_rereferencing(raw, ref_type='average')
    preprocessing_log['reference'] = 'average'

    if apply_ica and len(raw.ch_names) >= 15:
        raw, ica = apply_ica_artifact_removal(raw, n_components=15)
        preprocessing_log['ica_applied'] = True
        preprocessing_log['ica_components_removed'] = len(ica.exclude)
    else:
        preprocessing_log['ica_applied'] = False

    preprocessed_data = raw.get_data()
    preprocessing_log['final_n_channels'] = preprocessed_data.shape[0]
    preprocessing_log['final_n_timepoints'] = preprocessed_data.shape[1]

    return preprocessed_data, preprocessing_log, raw


def create_preprocessing_visualizations(raw_before, raw_after, trial_id):
    """Create before/after preprocessing visualizations."""
    channels_to_plot = ['C3', 'Cz', 'C4', 'Pz']
    available_channels = [ch for ch in channels_to_plot if ch in raw_before.ch_names]

    if not available_channels:
        return

    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    raw_before.plot(picks=available_channels, duration=2.0,
                    n_channels=len(available_channels),
                    scalings='auto', show=False, axes=axes[0])
    axes[0].set_title(f'{trial_id} - Before Preprocessing', fontsize=12, fontweight='bold')

    raw_after.plot(picks=available_channels, duration=2.0,
                   n_channels=len(available_channels),
                   scalings='auto', show=False, axes=axes[1])
    axes[1].set_title(f'{trial_id} - After Preprocessing', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'results/preprocessing/{trial_id}_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function."""
    print("-" * 70)
    print("STEP 2: EEG PREPROCESSING PIPELINE (MNE)")
    print("-" * 70)

    trial_metadata = pd.read_csv('data/raw/trial_metadata.csv')
    channel_info = pd.read_csv('data/raw/channel_info.csv')

    print(f"\nPreprocessing {len(trial_metadata)} EEG trials...")
    print("Pipeline: Filtering -> Bad channel detection -> Interpolation -> Re-referencing")

    preprocessing_logs = []
    sample_trials = trial_metadata['trial_id'].tolist()[:20]

    print(f"\nProcessing {len(sample_trials)} trials...\n")

    for idx, trial_id in enumerate(sample_trials):
        raw_before, trial_info = load_trial_to_mne(trial_id, trial_metadata, channel_info)

        preprocessed_data, log, raw_after = preprocess_trial(
            trial_id, trial_metadata, channel_info,
            apply_ica=False,
            verbose=True
        )

        preprocessing_logs.append(log)

        trial_idx = trial_metadata[trial_metadata['trial_id'] == trial_id].index[0]
        np.save(f'data/processed/preprocessed_eeg_trial_{trial_idx:04d}.npy',
                preprocessed_data)

        if idx < 3:
            create_preprocessing_visualizations(raw_before, raw_after, trial_id)
            print(f"    Visualization saved for {trial_id}")

    log_df = pd.DataFrame(preprocessing_logs)
    log_df.to_csv('results/quality_reports/preprocessing_log.csv', index=False)

    print("\n" + "-" * 70)
    print("PREPROCESSING SUMMARY")
    print("-" * 70)
    print(f"  Trials processed:              {len(preprocessing_logs)}")
    print(f"  Avg bad channels per trial:    {log_df['bad_channels'].mean():.2f}")
    print(f"  Trials with bad channels:      {(log_df['bad_channels'] > 0).sum()}")
    print(f"  ICA applied:                   {log_df['ica_applied'].sum()} trials")

    summary_stats = {
        'total_trials_processed': len(preprocessing_logs),
        'avg_bad_channels': float(log_df['bad_channels'].mean()),
        'trials_with_bad_channels': int((log_df['bad_channels'] > 0).sum()),
        'ica_applied_count': int(log_df['ica_applied'].sum()),
        'filter_lowcut': 0.5,
        'filter_highcut': 50.0,
        'notch_frequencies': [50, 60],
        'reference_type': 'average'
    }

    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv('documentation/preprocessing_summary.csv', index=False)

    print("\n" + "-" * 70)
    print("PREPROCESSING COMPLETE")
    print("-" * 70)
    print("\nOutput files:")
    print("  data/processed/preprocessed_eeg_trial_*.npy")
    print("  results/quality_reports/preprocessing_log.csv")
    print("  results/preprocessing/*_comparison.png")
    print("  documentation/preprocessing_summary.csv")


if __name__ == "__main__":
    main()
