"""
Step 1: BCI EEG Data Acquisition and Setup
Downloads public BCI dataset and sets up project structure
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import urllib.request
import json


def setup_directories():
    """Create project directory structure."""
    directories = [
        'data/raw',
        'data/processed',
        'data/sql_database',
        'results/preprocessing',
        'results/visualizations',
        'results/quality_reports',
        'documentation',
        'logs'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("Project directories created.")


def generate_synthetic_eeg_data(n_subjects=10, n_trials=50, n_channels=64,
                                 n_timepoints=1000, sampling_rate=250):
    """
    Generate synthetic EEG data simulating BCI motor imagery task.

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    n_trials : int
        Trials per subject per condition.
    n_channels : int
        Number of EEG channels.
    n_timepoints : int
        Time points per trial.
    sampling_rate : int
        Sampling frequency in Hz.
    """
    print("\nGenerating synthetic BCI EEG dataset...")

    np.random.seed(42)

    # EEG channel names (10-20 system)
    channel_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'CP5', 'CP1', 'CP2', 'CP6',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'PO9', 'O1', 'Oz', 'O2', 'PO10',
        'AF7', 'AF3', 'AF4', 'AF8',
        'F5', 'F1', 'F2', 'F6',
        'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10',
        'C5', 'C1', 'C2', 'C6',
        'TP9', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'TP10',
        'P5', 'P1', 'P2', 'P6',
        'PO7', 'PO3', 'POz', 'PO4', 'PO8',
        'Iz'
    ][:n_channels]

    # Task conditions for motor imagery BCI
    conditions = ['left_hand', 'right_hand', 'feet', 'rest']

    all_data = []
    subject_info = []

    for subject_id in range(1, n_subjects + 1):
        print(f"  Generating subject {subject_id}/{n_subjects}...")

        subject_data = []

        for condition in conditions:
            for trial in range(n_trials):
                eeg_trial = np.zeros((n_channels, n_timepoints))

                for ch in range(n_channels):
                    # 1/f background noise
                    freqs = np.fft.fftfreq(n_timepoints, 1/sampling_rate)
                    power = 1 / (np.abs(freqs) + 1)
                    phases = np.random.uniform(0, 2*np.pi, n_timepoints)
                    fft_signal = power * np.exp(1j * phases)
                    background = np.fft.ifft(fft_signal).real

                    # Alpha rhythm (8-12 Hz), stronger in occipital channels
                    t = np.arange(n_timepoints) / sampling_rate
                    if 'O' in channel_names[ch] or 'P' in channel_names[ch]:
                        alpha = 5 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2*np.pi))
                    else:
                        alpha = 2 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2*np.pi))

                    # Mu rhythm suppression in motor areas during motor imagery
                    mu_suppression = 0
                    if condition in ['left_hand', 'right_hand', 'feet']:
                        if 'C' in channel_names[ch]:
                            mu_suppression = -3 * np.sin(2 * np.pi * 11 * t)

                            if condition == 'left_hand' and channel_names[ch] in ['C4', 'CP4']:
                                mu_suppression *= 1.5
                            elif condition == 'right_hand' and channel_names[ch] in ['C3', 'CP3']:
                                mu_suppression *= 1.5

                    eeg_trial[ch, :] = background * 10 + alpha + mu_suppression
                    eeg_trial[ch, :] *= np.random.uniform(8, 12)

                trial_info = {
                    'subject_id': f'S{subject_id:02d}',
                    'trial_id': f'S{subject_id:02d}_T{len(subject_data):03d}',
                    'condition': condition,
                    'trial_number': trial,
                    'sampling_rate': sampling_rate,
                    'n_channels': n_channels,
                    'n_timepoints': n_timepoints,
                    'duration_sec': n_timepoints / sampling_rate
                }

                subject_data.append({
                    'info': trial_info,
                    'data': eeg_trial
                })

        all_data.extend(subject_data)

        subject_info.append({
            'subject_id': f'S{subject_id:02d}',
            'age': np.random.randint(20, 65),
            'sex': np.random.choice(['M', 'F']),
            'handedness': np.random.choice(['R', 'L'], p=[0.9, 0.1]),
            'n_trials': len(subject_data),
            'n_sessions': 1
        })

    print(f"\nGenerated {len(all_data)} EEG trials across {n_subjects} subjects.")

    channel_info = pd.DataFrame({
        'channel_id': range(len(channel_names)),
        'channel_name': channel_names,
        'channel_type': 'eeg'
    })
    channel_info.to_csv('data/raw/channel_info.csv', index=False)

    subject_info_df = pd.DataFrame(subject_info)
    subject_info_df.to_csv('data/raw/subject_info.csv', index=False)

    trial_metadata = pd.DataFrame([trial['info'] for trial in all_data])
    trial_metadata.to_csv('data/raw/trial_metadata.csv', index=False)

    print("Saving EEG data arrays...")
    for idx, trial in enumerate(all_data):
        np.save(f'data/raw/eeg_trial_{idx:04d}.npy', trial['data'])

    print(f"Saved {len(all_data)} EEG trial files.")

    return all_data, subject_info_df, channel_info, trial_metadata


def create_dataset_description():
    """Create dataset description file."""
    description = {
        'dataset_name': 'BCI Motor Imagery EEG Dataset',
        'task': 'Motor imagery (left hand, right hand, feet, rest)',
        'modality': 'EEG',
        'n_subjects': 10,
        'n_channels': 64,
        'sampling_rate': 250,
        'reference': 'average reference',
        'ground': 'AFz',
        'coordinate_system': '10-20 international system',
        'eeg_manufacturer': 'Simulated data for demonstration',
        'notes': 'Synthetic dataset for BCI neural prosthetics research pipeline demonstration'
    }

    with open('documentation/dataset_description.json', 'w') as f:
        json.dump(description, f, indent=2)

    print("Dataset description saved.")


def main():
    """Main execution function."""
    print("-" * 70)
    print("STEP 1: BCI EEG DATA ACQUISITION AND SETUP")
    print("-" * 70)

    setup_directories()

    all_data, subject_info, channel_info, trial_metadata = generate_synthetic_eeg_data(
        n_subjects=10,
        n_trials=50,
        n_channels=64,
        n_timepoints=1000,
        sampling_rate=250
    )

    create_dataset_description()

    print("\n" + "-" * 70)
    print("DATA ACQUISITION COMPLETE")
    print("-" * 70)
    print("\nDataset summary:")
    print(f"  Subjects:      {len(subject_info)}")
    print(f"  Total trials:  {len(trial_metadata)}")
    print(f"  Channels:      {len(channel_info)}")
    print(f"  Conditions:    {trial_metadata['condition'].nunique()}")
    print(f"  Sampling rate: 250 Hz")
    print(f"  Trial duration: 4.0 seconds")

    print("\nGenerated files:")
    print("  data/raw/eeg_trial_*.npy")
    print("  data/raw/subject_info.csv")
    print("  data/raw/channel_info.csv")
    print("  data/raw/trial_metadata.csv")
    print("  documentation/dataset_description.json")


if __name__ == "__main__":
    main()
