"""
Railway Track Fault Detection - Data Preprocessing & Exploratory Analysis
Step 1: Environment Setup and Data Loading
"""

# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import os
from scipy import stats
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('default')
sns.set_palette("husl")

print("=" * 70)
print("RAILWAY TRACK FAULT DETECTION - DATA PREPROCESSING")
print("=" * 70)

# =============================================================================
# STEP 1: LOAD AND INSPECT DATASETS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 1: LOADING DATASETS")
print("=" * 70)

# 1.1 Load RT_PLC_RSFPD.csv (Main Dataset)
print("\n--- Loading RT_PLC_RSFPD.csv ---")
try:
    plc_df = pd.read_csv('datasets/RT_PLC_RSFPD.csv')
    print(f"✓ Successfully loaded PLC data")
    print(f"  Shape: {plc_df.shape}")
    print(f"  Columns: {len(plc_df.columns)}")
except Exception as e:
    print(f"✗ Error loading PLC data: {e}")
    plc_df = None

# 1.2 Load vibration_raw.csv (High-frequency Data)
print("\n--- Loading vibration_raw.csv ---")
try:
    # Load first 50000 rows for initial analysis (full dataset is very large)
    vibration_df = pd.read_csv('datasets/vibration_raw.csv', nrows=50000)
    print(f"✓ Successfully loaded vibration data (first 50000 rows)")
    print(f"  Shape: {vibration_df.shape}")
    print(f"  Columns: {len(vibration_df.columns)}")
except Exception as e:
    print(f"✗ Error loading vibration data: {e}")
    vibration_df = None

# 1.3 Load Sensor Vibration Files (Multiple CSVs)
print("\n--- Loading Sensor Vibration Files ---")
try:
    sensor_files = glob.glob('datasets/sensor_vibration/*.csv')
    # Filter out Excel files
    sensor_files = [f for f in sensor_files if not f.endswith(('.xlsx', '.xls'))]
    print(f"✓ Found {len(sensor_files)} CSV sensor files")
    
    # Load first 10 files for initial inspection
    sample_sensor_dfs = []
    for i, file in enumerate(sensor_files[:10]):
        try:
            df = pd.read_csv(file, sep=';')
            df['source_file'] = os.path.basename(file)
            sample_sensor_dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {os.path.basename(file)}: {e}")
    
    if sample_sensor_dfs:
        sensor_df = pd.concat(sample_sensor_dfs, ignore_index=True)
        print(f"✓ Loaded {len(sample_sensor_dfs)} sensor files")
        print(f"  Combined shape: {sensor_df.shape}")
    else:
        sensor_df = None
        print("✗ No sensor files could be loaded")
        
except Exception as e:
    print(f"✗ Error loading sensor data: {e}")
    sensor_df = None

# =============================================================================
# STEP 2: DATA INSPECTION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: DATA INSPECTION")
print("=" * 70)

def inspect_dataset(df, name):
    """Inspect a dataset and print key information"""
    print(f"\n--- {name} ---")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Duplicated Rows: {df.duplicated().sum()}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")

if plc_df is not None:
    inspect_dataset(plc_df, "RT_PLC_RSFPD Dataset")

if vibration_df is not None:
    inspect_dataset(vibration_df, "Vibration Raw Dataset")

if sensor_df is not None:
    inspect_dataset(sensor_df, "Sensor Vibration Dataset")

# =============================================================================
# STEP 3: DATA CLEANING
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: DATA CLEANING")
print("=" * 70)

def clean_plc_data(df):
    """Clean and preprocess PLC data"""
    print("\n--- Cleaning PLC Data ---")
    df_clean = df.copy()
    
    # Convert timestamp
    df_clean['Timestamp'] = pd.to_datetime(df_clean['Timestamp'])
    
    # Check missing values before cleaning
    missing_before = df_clean.isnull().sum().sum()
    print(f"Missing values before cleaning: {missing_before}")
    
    # Handle missing values for numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
    
    # Handle missing values for categorical columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()
            if len(mode_val) > 0:
                df_clean[col] = df_clean[col].fillna(mode_val[0])
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    missing_after = df_clean.isnull().sum().sum()
    print(f"Missing values after cleaning: {missing_after}")
    print(f"Shape after cleaning: {df_clean.shape}")
    
    return df_clean

def clean_vibration_data(df):
    """Clean raw vibration data"""
    print("\n--- Cleaning Vibration Data ---")
    df_clean = df.copy()
    
    # Convert timestamp
    df_clean['TIMESTAMP'] = pd.to_datetime(df_clean['TIMESTAMP'])
    
    # Check for missing values
    missing_before = df_clean.isnull().sum().sum()
    print(f"Missing values before cleaning: {missing_before}")
    
    # Key sensor columns
    sensor_cols = ['ACCEL_X', 'ACCEL_Y', 'ACCEL_Z']
    
    # Remove rows with all NaN in critical sensor columns
    df_clean = df_clean.dropna(subset=sensor_cols, how='all')
    
    # Interpolate missing values
    for col in sensor_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].interpolate(method='linear')
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Remove extreme outliers using IQR method
    for col in sensor_cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            print(f"  {col}: Removed {outliers} extreme outliers")
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    print(f"Shape after cleaning: {df_clean.shape}")
    return df_clean

def clean_sensor_data(df):
    """Clean sensor vibration data"""
    print("\n--- Cleaning Sensor Data ---")
    df_clean = df.copy()
    
    # Rename columns based on typical sensor structure
    if 'source_file' in df_clean.columns:
        data_cols = [col for col in df_clean.columns if col != 'source_file']
        if len(data_cols) == 4:
            df_clean.columns = ['Value1', 'Accel_X', 'Accel_Y', 'Accel_Z', 'source_file']
        else:
            # Generic naming
            new_cols = [f'Column_{i}' for i in range(len(data_cols))] + ['source_file']
            df_clean.columns = new_cols
    
    # Convert to numeric
    for col in df_clean.columns:
        if col != 'source_file':
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Remove rows with all NaN
    df_clean = df_clean.dropna(how='all')
    
    # Extract timestamp from filename
    if 'source_file' in df_clean.columns:
        df_clean['datetime'] = df_clean['source_file'].str.extract(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})')
        df_clean['datetime'] = pd.to_datetime(df_clean['datetime'], 
                                               format='%Y_%m_%d_%H_%M_%S', 
                                               errors='coerce')
    
    print(f"Shape after cleaning: {df_clean.shape}")
    return df_clean

# Apply cleaning
plc_clean = clean_plc_data(plc_df) if plc_df is not None else None
vibration_clean = clean_vibration_data(vibration_df) if vibration_df is not None else None
sensor_clean = clean_sensor_data(sensor_df) if sensor_df is not None else None

# =============================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)

# Create output directory for plots
os.makedirs('analysis_output', exist_ok=True)

def analyze_failures(df):
    """Analyze failure types and distributions"""
    print("\n--- Failure Analysis ---")
    
    if 'Failure_Type' not in df.columns:
        print("Failure_Type column not found")
        return
    
    # Failure type distribution
    failure_counts = df['Failure_Type'].value_counts()
    print(f"\nFailure Type Distribution:\n{failure_counts}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Failure type bar chart
    axes[0, 0].bar(failure_counts.index, failure_counts.values)
    axes[0, 0].set_title('Distribution of Failure Types')
    axes[0, 0].set_xlabel('Failure Type')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # RUL distribution
    if 'RUL_Predicted_days' in df.columns:
        axes[0, 1].hist(df['RUL_Predicted_days'], bins=50, alpha=0.7, color='orange')
        axes[0, 1].set_title('Distribution of Remaining Useful Life (RUL)')
        axes[0, 1].set_xlabel('RUL (days)')
        axes[0, 1].set_ylabel('Frequency')
    
    # Failure probability distribution
    if 'Predicted_Failure_Prob' in df.columns:
        axes[1, 0].hist(df['Predicted_Failure_Prob'], bins=50, alpha=0.7, color='red')
        axes[1, 0].set_title('Predicted Failure Probability Distribution')
        axes[1, 0].set_xlabel('Failure Probability')
        axes[1, 0].set_ylabel('Frequency')
    
    # Anomaly score distribution
    if 'Edge_Anomaly_Score' in df.columns:
        axes[1, 1].hist(df['Edge_Anomaly_Score'], bins=50, alpha=0.7, color='purple')
        axes[1, 1].set_title('Edge Anomaly Score Distribution')
        axes[1, 1].set_xlabel('Anomaly Score')
        axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('analysis_output/failure_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: analysis_output/failure_analysis.png")
    plt.close()
    
    # RUL statistics by failure type
    if 'RUL_Predicted_days' in df.columns:
        print(f"\nRUL Statistics by Failure Type:")
        print(df.groupby('Failure_Type')['RUL_Predicted_days'].describe())

def analyze_sensor_correlations(df):
    """Analyze correlations between sensors"""
    print("\n--- Sensor Correlation Analysis ---")
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation analysis")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
    plt.title('Sensor Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig('analysis_output/correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: analysis_output/correlation_matrix.png")
    plt.close()
    
    # Find highly correlated features with RUL
    if 'RUL_Predicted_days' in corr_matrix.columns:
        rul_correlations = corr_matrix['RUL_Predicted_days'].abs().sort_values(ascending=False)
        print(f"\nTop 10 correlations with RUL:")
        print(rul_correlations.head(10))

def analyze_vibration_patterns(df):
    """Analyze vibration patterns"""
    print("\n--- Vibration Pattern Analysis ---")
    
    # Check for required columns
    accel_cols = ['ACCEL_X', 'ACCEL_Y', 'ACCEL_Z']
    available_cols = [col for col in accel_cols if col in df.columns]
    
    if not available_cols:
        print("No acceleration columns found")
        return
    
    # Calculate vibration magnitude
    df['vibration_magnitude'] = np.sqrt(sum(df[col]**2 for col in available_cols))
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Vibration over time (first 1000 points)
    sample_size = min(1000, len(df))
    axes[0, 0].plot(range(sample_size), df['vibration_magnitude'][:sample_size])
    axes[0, 0].set_title('Vibration Magnitude Over Time (Sample)')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Magnitude')
    
    # Vibration distribution
    axes[0, 1].hist(df['vibration_magnitude'], bins=50, alpha=0.7, color='green')
    axes[0, 1].set_title('Vibration Magnitude Distribution')
    axes[0, 1].set_xlabel('Magnitude')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3D scatter (if we have all 3 axes)
    if len(available_cols) >= 2:
        sample_df = df.sample(min(1000, len(df)))
        axes[1, 0].scatter(sample_df[available_cols[0]], sample_df[available_cols[1]], 
                          alpha=0.5, s=1)
        axes[1, 0].set_title(f'{available_cols[0]} vs {available_cols[1]}')
        axes[1, 0].set_xlabel(available_cols[0])
        axes[1, 0].set_ylabel(available_cols[1])
    
    # FFT analysis
    if len(df) >= 1024:
        sample_signal = df[available_cols[0]].iloc[:1024].values
        fft_vals = np.abs(fft(sample_signal))
        freqs = np.fft.fftfreq(len(sample_signal))
        axes[1, 1].plot(freqs[:len(freqs)//2], fft_vals[:len(fft_vals)//2])
        axes[1, 1].set_title('Frequency Domain (FFT)')
        axes[1, 1].set_xlabel('Frequency')
        axes[1, 1].set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig('analysis_output/vibration_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: analysis_output/vibration_analysis.png")
    plt.close()
    
    print(f"\nVibration statistics:")
    print(df['vibration_magnitude'].describe())

def analyze_time_series(df):
    """Analyze time-series trends"""
    print("\n--- Time-Series Analysis ---")
    
    if 'Timestamp' not in df.columns:
        print("Timestamp column not found")
        return
    
    # Set timestamp as index
    df_ts = df.set_index('Timestamp')
    
    # Key sensors to analyze
    sensors_to_plot = ['Vibration_m_s2', 'Temperature_C', 'Voltage_V', 'Current_A']
    available_sensors = [col for col in sensors_to_plot if col in df_ts.columns]
    
    if not available_sensors:
        print("No standard sensor columns found for time-series analysis")
        return
    
    # Create subplots
    n_sensors = len(available_sensors)
    fig, axes = plt.subplots(n_sensors, 1, figsize=(14, 3*n_sensors))
    
    if n_sensors == 1:
        axes = [axes]
    
    for i, sensor in enumerate(available_sensors):
        # Plot raw data (sample)
        sample_data = df_ts[sensor].dropna()
        if len(sample_data) > 1000:
            sample_data = sample_data.iloc[:1000]
        
        axes[i].plot(sample_data.index, sample_data.values)
        axes[i].set_title(f'{sensor} Over Time')
        axes[i].set_ylabel(sensor)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('analysis_output/time_series_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: analysis_output/time_series_analysis.png")
    plt.close()
    
    # Trend analysis
    print("\nTrend Analysis (linear regression on each sensor):")
    for sensor in available_sensors:
        clean_data = df_ts[sensor].dropna()
        if len(clean_data) > 10:
            x = np.arange(len(clean_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, clean_data.values)
            trend = "increasing" if slope > 0 else "decreasing"
            print(f"  {sensor}: {trend} trend (slope={slope:.6f}, R²={r_value**2:.4f})")

# Run EDA functions
if plc_clean is not None:
    analyze_failures(plc_clean)
    analyze_sensor_correlations(plc_clean)
    analyze_time_series(plc_clean)

if vibration_clean is not None:
    analyze_vibration_patterns(vibration_clean)

# =============================================================================
# STEP 5: FEATURE ENGINEERING
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: FEATURE ENGINEERING")
print("=" * 70)

def engineer_features(df):
    """Create engineered features"""
    print("\n--- Engineering Features ---")
    df_feat = df.copy()
    
    features_created = []
    
    # 1. Statistical features for vibration
    if 'Vibration_m_s2' in df_feat.columns:
        window = 10
        df_feat['vib_roll_mean'] = df_feat['Vibration_m_s2'].rolling(window=window, min_periods=1).mean()
        df_feat['vib_roll_std'] = df_feat['Vibration_m_s2'].rolling(window=window, min_periods=1).std()
        df_feat['vib_roll_max'] = df_feat['Vibration_m_s2'].rolling(window=window, min_periods=1).max()
        features_created.extend(['vib_roll_mean', 'vib_roll_std', 'vib_roll_max'])
    
    # 2. Temperature differential
    if 'Temperature_C' in df_feat.columns and 'Ambient_Temp_C' in df_feat.columns:
        df_feat['temp_differential'] = df_feat['Temperature_C'] - df_feat['Ambient_Temp_C']
        features_created.append('temp_differential')
    
    # 3. Power features
    if 'Voltage_V' in df_feat.columns and 'Current_A' in df_feat.columns:
        df_feat['power_W'] = df_feat['Voltage_V'] * df_feat['Current_A']
        df_feat['resistance_Ohm'] = df_feat['Voltage_V'] / (df_feat['Current_A'] + 1e-6)
        features_created.extend(['power_W', 'resistance_Ohm'])
    
    # 4. Health score combination
    health_cols = ['Edge_Anomaly_Score', 'Cloud_Health_Index', 'Predicted_Failure_Prob']
    available_health_cols = [col for col in health_cols if col in df_feat.columns]
    if available_health_cols:
        df_feat['composite_health_score'] = df_feat[available_health_cols].mean(axis=1)
        features_created.append('composite_health_score')
    
    # 5. Time-based features
    if 'Timestamp' in df_feat.columns:
        df_feat['hour'] = df_feat['Timestamp'].dt.hour
        df_feat['day_of_week'] = df_feat['Timestamp'].dt.dayofweek
        df_feat['month'] = df_feat['Timestamp'].dt.month
        df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
        features_created.extend(['hour', 'day_of_week', 'month', 'is_weekend'])
    
    print(f"✓ Created {len(features_created)} new features:")
    for feat in features_created:
        print(f"  - {feat}")
    print(f"New shape: {df_feat.shape}")
    
    return df_feat

# Apply feature engineering
plc_features = engineer_features(plc_clean) if plc_clean is not None else None

# =============================================================================
# STEP 6: SAVE PROCESSED DATA
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: SAVING PROCESSED DATA")
print("=" * 70)

def save_processed_data():
    """Save cleaned and processed datasets"""
    os.makedirs('processed_data', exist_ok=True)
    
    saved_files = []
    
    if plc_features is not None:
        plc_features.to_csv('processed_data/plc_processed.csv', index=False)
        saved_files.append('processed_data/plc_processed.csv')
        print(f"✓ Saved: processed_data/plc_processed.csv ({plc_features.shape})")
    
    if vibration_clean is not None:
        vibration_clean.to_csv('processed_data/vibration_processed.csv', index=False)
        saved_files.append('processed_data/vibration_processed.csv')
        print(f"✓ Saved: processed_data/vibration_processed.csv ({vibration_clean.shape})")
    
    if sensor_clean is not None:
        sensor_clean.to_csv('processed_data/sensor_processed.csv', index=False)
        saved_files.append('processed_data/sensor_processed.csv')
        print(f"✓ Saved: processed_data/sensor_processed.csv ({sensor_clean.shape})")
    
    return saved_files

saved_files = save_processed_data()

# =============================================================================
# STEP 7: SUMMARY REPORT
# =============================================================================

print("\n" + "=" * 70)
print("PREPROCESSING SUMMARY REPORT")
print("=" * 70)

print("\n📊 DATASETS PROCESSED:")
if plc_df is not None:
    print(f"   ✓ RT_PLC_RSFPD.csv - {plc_df.shape[0]} records, {plc_df.shape[1]} features")
if vibration_df is not None:
    print(f"   ✓ vibration_raw.csv - {vibration_df.shape[0]} records (sample), {vibration_df.shape[1]} features")
if sensor_df is not None:
    print(f"   ✓ sensor_vibration/ - {len(sensor_files)} files, {sensor_df.shape[0]} combined records")

print("\n🔧 CLEANING OPERATIONS:")
print("   ✓ Timestamp standardization")
print("   ✓ Missing value imputation (median for numeric, mode for categorical)")
print("   ✓ Duplicate removal")
print("   ✓ Outlier detection and removal (IQR method)")

print("\n📈 EXPLORATORY ANALYSIS:")
print("   ✓ Failure type distribution analysis")
print("   ✓ Sensor correlation matrix")
print("   ✓ Vibration pattern analysis")
print("   ✓ Time-series trend analysis")

print("\n🔮 FEATURE ENGINEERING:")
print("   ✓ Rolling statistics (mean, std, max)")
print("   ✓ Temperature differentials")
print("   ✓ Power and resistance calculations")
print("   ✓ Composite health scores")
print("   ✓ Time-based features (hour, day, month, weekend)")

print("\n💾 OUTPUT FILES:")
for file in saved_files:
    print(f"   ✓ {file}")

print("\n📊 VISUALIZATIONS CREATED:")
viz_files = [
    'analysis_output/failure_analysis.png',
    'analysis_output/correlation_matrix.png',
    'analysis_output/vibration_analysis.png',
    'analysis_output/time_series_analysis.png'
]
for viz in viz_files:
    if os.path.exists(viz):
        print(f"   ✓ {viz}")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("1. Review the generated visualizations in 'analysis_output/' folder")
print("2. Examine the processed data in 'processed_data/' folder")
print("3. Proceed to model development for fault detection")
print("=" * 70)

print("\n✅ Data Preprocessing Complete!")
