"""
Step 3: SQL Database Creation - Structuring EEG Trials
Creates relational database for neuroscience analysis
"""

import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json

# Create SQLAlchemy base
Base = declarative_base()

# Define database schema

class Subject(Base):
    """Subject/participant table"""
    __tablename__ = 'subjects'
    
    subject_id = Column(String, primary_key=True)
    age = Column(Integer)
    sex = Column(String)
    handedness = Column(String)
    n_trials = Column(Integer)
    n_sessions = Column(Integer)
    
    # Relationships
    trials = relationship("Trial", back_populates="subject")

class Channel(Base):
    """EEG channel information table"""
    __tablename__ = 'channels'
    
    channel_id = Column(Integer, primary_key=True)
    channel_name = Column(String, unique=True)
    channel_type = Column(String)
    x_coord = Column(Float, nullable=True)
    y_coord = Column(Float, nullable=True)
    z_coord = Column(Float, nullable=True)

class Trial(Base):
    """EEG trial metadata table"""
    __tablename__ = 'trials'
    
    trial_id = Column(String, primary_key=True)
    subject_id = Column(String, ForeignKey('subjects.subject_id'))
    condition = Column(String)
    trial_number = Column(Integer)
    sampling_rate = Column(Float)
    n_channels = Column(Integer)
    n_timepoints = Column(Integer)
    duration_sec = Column(Float)
    preprocessing_status = Column(String)
    bad_channels_count = Column(Integer, default=0)
    bad_channels_list = Column(Text, nullable=True)
    
    # Relationships
    subject = relationship("Subject", back_populates="trials")
    eeg_data = relationship("EEGData", back_populates="trial")
    features = relationship("TrialFeatures", back_populates="trial", uselist=False)

class EEGData(Base):
    """EEG time series data table (compressed/aggregated)"""
    __tablename__ = 'eeg_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trial_id = Column(String, ForeignKey('trials.trial_id'))
    channel_id = Column(Integer, ForeignKey('channels.channel_id'))
    mean_amplitude = Column(Float)
    std_amplitude = Column(Float)
    min_amplitude = Column(Float)
    max_amplitude = Column(Float)
    rms_amplitude = Column(Float)
    
    # Relationships
    trial = relationship("Trial", back_populates="eeg_data")

class TrialFeatures(Base):
    """Extracted features per trial for analysis"""
    __tablename__ = 'trial_features'
    
    trial_id = Column(String, ForeignKey('trials.trial_id'), primary_key=True)
    
    # Power spectral density features
    delta_power = Column(Float)  # 0.5-4 Hz
    theta_power = Column(Float)  # 4-8 Hz
    alpha_power = Column(Float)  # 8-13 Hz
    beta_power = Column(Float)   # 13-30 Hz
    gamma_power = Column(Float)  # 30-50 Hz
    
    # Motor cortex features (C3, C4 channels)
    c3_alpha_power = Column(Float)
    c4_alpha_power = Column(Float)
    motor_asymmetry = Column(Float)  # (C3-C4)/(C3+C4)
    
    # Signal quality metrics
    signal_to_noise_ratio = Column(Float)
    artifact_probability = Column(Float)
    
    # Relationships
    trial = relationship("Trial", back_populates="features")

class PreprocessingLog(Base):
    """Preprocessing log table"""
    __tablename__ = 'preprocessing_log'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trial_id = Column(String)
    timestamp = Column(DateTime, default=datetime.now)
    step = Column(String)
    parameters = Column(Text)
    success = Column(Integer)  # 1 for success, 0 for failure

def create_database(db_path='data/sql_database/bci_eeg.db'):
    """Create SQLite database with schema"""
    print("\n🗄️  Creating SQL database...")
    
    # Create engine
    engine = create_engine(f'sqlite:///{db_path}', echo=False)
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    print(f"✓ Database created: {db_path}")
    print(f"✓ Tables created: {', '.join(Base.metadata.tables.keys())}")
    
    return engine

def populate_subjects_table(engine):
    """Populate subjects table"""
    print("\n👥 Populating subjects table...")
    
    subject_info = pd.read_csv('data/raw/subject_info.csv')
    subject_info.to_sql('subjects', engine, if_exists='replace', index=False)
    
    print(f"✓ Inserted {len(subject_info)} subjects")

def populate_channels_table(engine):
    """Populate channels table"""
    print("\n📡 Populating channels table...")
    
    channel_info = pd.read_csv('data/raw/channel_info.csv')
    
    # Add dummy coordinates (in production, use actual 10-20 coordinates)
    channel_info['x_coord'] = np.random.uniform(-1, 1, len(channel_info))
    channel_info['y_coord'] = np.random.uniform(-1, 1, len(channel_info))
    channel_info['z_coord'] = np.random.uniform(0, 1, len(channel_info))
    
    channel_info.to_sql('channels', engine, if_exists='replace', index=False)
    
    print(f"✓ Inserted {len(channel_info)} channels")

def populate_trials_table(engine):
    """Populate trials table with preprocessing info"""
    print("\n🔬 Populating trials table...")
    
    trial_metadata = pd.read_csv('data/raw/trial_metadata.csv')
    
    # Add preprocessing status
    try:
        preprocessing_log = pd.read_csv('results/quality_reports/preprocessing_log.csv')
        trial_metadata = trial_metadata.merge(
            preprocessing_log[['trial_id', 'bad_channels', 'bad_channel_names']], 
            on='trial_id', 
            how='left'
        )
        trial_metadata['preprocessing_status'] = 'completed'
        trial_metadata['bad_channels_count'] = trial_metadata['bad_channels'].fillna(0)
        trial_metadata['bad_channels_list'] = trial_metadata['bad_channel_names'].fillna('')
    except:
        trial_metadata['preprocessing_status'] = 'pending'
        trial_metadata['bad_channels_count'] = 0
        trial_metadata['bad_channels_list'] = ''
    
    # Select relevant columns
    trials_df = trial_metadata[[
        'trial_id', 'subject_id', 'condition', 'trial_number', 
        'sampling_rate', 'n_channels', 'n_timepoints', 'duration_sec',
        'preprocessing_status', 'bad_channels_count', 'bad_channels_list'
    ]]
    
    trials_df.to_sql('trials', engine, if_exists='replace', index=False)
    
    print(f"✓ Inserted {len(trials_df)} trials")

def compute_and_populate_eeg_data(engine, n_trials=20):
    """Compute and populate EEG data statistics"""
    print("\n📊 Computing EEG data statistics...")
    
    trial_metadata = pd.read_csv('data/raw/trial_metadata.csv')
    channel_info = pd.read_csv('data/raw/channel_info.csv')
    
    eeg_data_records = []
    
    for idx, trial_id in enumerate(trial_metadata['trial_id'].tolist()[:n_trials]):
        # Load preprocessed data
        try:
            data = np.load(f'data/processed/preprocessed_eeg_trial_{idx:04d}.npy')
        except:
            # If preprocessed doesn't exist, use raw
            data = np.load(f'data/raw/eeg_trial_{idx:04d}.npy')
        
        # Compute statistics for each channel
        for ch_idx in range(data.shape[0]):
            channel_data = data[ch_idx, :]
            
            record = {
                'trial_id': trial_id,
                'channel_id': ch_idx,
                'mean_amplitude': float(np.mean(channel_data)),
                'std_amplitude': float(np.std(channel_data)),
                'min_amplitude': float(np.min(channel_data)),
                'max_amplitude': float(np.max(channel_data)),
                'rms_amplitude': float(np.sqrt(np.mean(channel_data**2)))
            }
            eeg_data_records.append(record)
        
        if (idx + 1) % 5 == 0:
            print(f"   Processed {idx + 1}/{n_trials} trials...")
    
    eeg_data_df = pd.DataFrame(eeg_data_records)
    eeg_data_df.to_sql('eeg_data', engine, if_exists='replace', index=False)
    
    print(f"✓ Inserted {len(eeg_data_records)} EEG data records")

def compute_spectral_features(eeg_data, sampling_rate=250):
    """Compute power spectral density features"""
    from scipy import signal
    
    # Compute PSD using Welch's method
    freqs, psd = signal.welch(eeg_data, fs=sampling_rate, nperseg=sampling_rate)
    
    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    band_powers = {}
    for band_name, (low, high) in bands.items():
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        band_powers[band_name] = np.trapz(psd[idx_band], freqs[idx_band])
    
    return band_powers

def populate_trial_features(engine, n_trials=20):
    """Compute and populate trial features"""
    print("\n🎯 Computing trial features...")
    
    trial_metadata = pd.read_csv('data/raw/trial_metadata.csv')
    channel_info = pd.read_csv('data/raw/channel_info.csv')
    
    features_records = []
    
    for idx, trial_id in enumerate(trial_metadata['trial_id'].tolist()[:n_trials]):
        try:
            data = np.load(f'data/processed/preprocessed_eeg_trial_{idx:04d}.npy')
        except:
            data = np.load(f'data/raw/eeg_trial_{idx:04d}.npy')
        
        # Compute average spectral features across all channels
        all_band_powers = {band: [] for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']}
        
        for ch_idx in range(data.shape[0]):
            band_powers = compute_spectral_features(data[ch_idx, :])
            for band, power in band_powers.items():
                all_band_powers[band].append(power)
        
        # Average across channels
        avg_powers = {band: np.mean(powers) for band, powers in all_band_powers.items()}
        
        # Motor cortex specific features (C3, C4)
        c3_idx = channel_info[channel_info['channel_name'] == 'C3'].index
        c4_idx = channel_info[channel_info['channel_name'] == 'C4'].index
        
        c3_alpha = 0.0
        c4_alpha = 0.0
        if len(c3_idx) > 0 and c3_idx[0] < data.shape[0]:
            c3_alpha = compute_spectral_features(data[c3_idx[0], :])['alpha']
        if len(c4_idx) > 0 and c4_idx[0] < data.shape[0]:
            c4_alpha = compute_spectral_features(data[c4_idx[0], :])['alpha']
        
        motor_asymmetry = 0.0
        if (c3_alpha + c4_alpha) > 0:
            motor_asymmetry = (c3_alpha - c4_alpha) / (c3_alpha + c4_alpha)
        
        # Signal quality metrics
        snr = np.mean(np.abs(data)) / np.std(data)
        artifact_prob = np.sum(np.abs(data) > 100) / data.size  # Simple artifact detection
        
        record = {
            'trial_id': trial_id,
            'delta_power': float(avg_powers['delta']),
            'theta_power': float(avg_powers['theta']),
            'alpha_power': float(avg_powers['alpha']),
            'beta_power': float(avg_powers['beta']),
            'gamma_power': float(avg_powers['gamma']),
            'c3_alpha_power': float(c3_alpha),
            'c4_alpha_power': float(c4_alpha),
            'motor_asymmetry': float(motor_asymmetry),
            'signal_to_noise_ratio': float(snr),
            'artifact_probability': float(artifact_prob)
        }
        features_records.append(record)
    
    features_df = pd.DataFrame(features_records)
    features_df.to_sql('trial_features', engine, if_exists='replace', index=False)
    
    print(f"✓ Inserted {len(features_records)} feature records")

def create_database_views(engine):
    """Create useful SQL views for analysis"""
    print("\n👁️  Creating database views...")
    
    with engine.connect() as conn:
        # View 1: Trial summary with subject info
        conn.execute("""
            CREATE VIEW IF NOT EXISTS trial_summary AS
            SELECT 
                t.trial_id,
                t.subject_id,
                s.age,
                s.sex,
                s.handedness,
                t.condition,
                t.trial_number,
                t.preprocessing_status,
                t.bad_channels_count,
                f.alpha_power,
                f.motor_asymmetry
            FROM trials t
            JOIN subjects s ON t.subject_id = s.subject_id
            LEFT JOIN trial_features f ON t.trial_id = f.trial_id
        """)
        
        # View 2: Condition averages
        conn.execute("""
            CREATE VIEW IF NOT EXISTS condition_averages AS
            SELECT 
                t.condition,
                COUNT(*) as n_trials,
                AVG(f.alpha_power) as avg_alpha_power,
                AVG(f.beta_power) as avg_beta_power,
                AVG(f.motor_asymmetry) as avg_motor_asymmetry
            FROM trials t
            LEFT JOIN trial_features f ON t.trial_id = f.trial_id
            GROUP BY t.condition
        """)
        
        conn.commit()
    
    print("✓ Created database views")

def main():
    """Main execution function"""
    print("="*70)
    print("STEP 3: SQL DATABASE CREATION")
    print("="*70)
    
    # Create database
    engine = create_database()
    
    # Populate tables
    populate_subjects_table(engine)
    populate_channels_table(engine)
    populate_trials_table(engine)
    compute_and_populate_eeg_data(engine, n_trials=20)
    populate_trial_features(engine, n_trials=20)
    
    # Create views
    create_database_views(engine)
    
    # Generate database report
    print("\n" + "="*70)
    print("📊 DATABASE SUMMARY")
    print("="*70)
    
    with engine.connect() as conn:
        subjects_count = pd.read_sql("SELECT COUNT(*) as count FROM subjects", conn).iloc[0]['count']
        trials_count = pd.read_sql("SELECT COUNT(*) as count FROM trials", conn).iloc[0]['count']
        channels_count = pd.read_sql("SELECT COUNT(*) as count FROM channels", conn).iloc[0]['count']
        eeg_data_count = pd.read_sql("SELECT COUNT(*) as count FROM eeg_data", conn).iloc[0]['count']
        features_count = pd.read_sql("SELECT COUNT(*) as count FROM trial_features", conn).iloc[0]['count']
        
        print(f"   Subjects: {subjects_count}")
        print(f"   Trials: {trials_count}")
        print(f"   Channels: {channels_count}")
        print(f"   EEG Data Records: {eeg_data_count}")
        print(f"   Trial Features: {features_count}")
    
    print("\n" + "="*70)
    print("✅ SQL DATABASE CREATION COMPLETE")
    print("="*70)
    print("\n📁 Database file:")
    print("   - data/sql_database/bci_eeg.db")
    print("\n📊 Tables created:")
    print("   - subjects, channels, trials, eeg_data, trial_features, preprocessing_log")
    print("\n👁️  Views created:")
    print("   - trial_summary, condition_averages")

if __name__ == "__main__":
    main()
