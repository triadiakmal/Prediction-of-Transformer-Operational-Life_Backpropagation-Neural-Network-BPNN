import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import os
import random
from datetime import datetime

# --- KONFIGURASI GLOBAL ---
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# --- KONFIGURASI SKRIP ---
FILE_PATH = 'D:/PREDIKSI SISA UMUR TRAFO/dataset_bpnn_transformator.xlsx'
SHEET_NAMES = ['Transformator 1', 'Transformator 2', 'Transformator 3', 'Transformator 4']
FEATURES = ['Rata-Rata Beban (MW)', 'Suhu Hotspot (℃)', 'Laju Penuaan Thermal (p.u)', 'TDCG (ppm)',
             'Kadar Air (ppm)', 'BDV (kV/2.5 mm)', 'Kadar Asam (mgKOH/g)', 'Usia Pakai (Tahun)']
TARGET = 'Sisa Umur'
TARGET_YEAR = 2030 
K_FOLDS = 10
EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 0.03

# Inisialisasi
all_results = {}
future_predictions = []

# --- FUNGSI UNTUK MEMBUAT MODEL ---
def create_model(input_shape):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mse'])
    return model

# --- PROSES PELATIHAN DAN EVALUASI PER TRANSFORMATOR ---
for sheet_name in SHEET_NAMES:
    print(f"\n{'='*50}")
    print(f"Memproses: {sheet_name}")
    print(f"{'='*50}")

    if not os.path.exists(FILE_PATH):
        print(f"Error: File tidak ditemukan di '{FILE_PATH}'")
        continue

    df = pd.read_excel(FILE_PATH, sheet_name=sheet_name)
    df['Waktu'] = pd.to_datetime(df['Waktu'])

    X = df[FEATURES].values
    y = df[TARGET].values.reshape(-1, 1)

    x_scaler = StandardScaler()
    y_scaler = MinMaxScaler()

    X_scaled = x_scaler.fit_transform(X)

    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = {'mse': [], 'r2': []}
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
        print(f"\nFold {fold+1}/{K_FOLDS}")
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        y_train_scaled = y_scaler.fit_transform(y_train)
        y_val_scaled = y_scaler.transform(y_val)

        model = create_model(X_train.shape[1])
        callbacks = [EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)]

        model.fit(X_train, y_train_scaled, validation_data=(X_val, y_val_scaled),
                  epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0)

        y_pred_scaled = model.predict(X_val)
        y_pred_original = y_scaler.inverse_transform(y_pred_scaled)

        mse = mean_squared_error(y_val, y_pred_original)
        r2 = r2_score(y_val, y_pred_original)
        fold_results['mse'].append(mse)
        fold_results['r2'].append(r2)
        fold_metrics.append((fold+1, mse, r2))
        print(f"MSE: {mse:.4f}, R²: {r2:.4f}")

    avg_mse = np.mean(fold_results['mse'])
    avg_r2 = np.mean(fold_results['r2'])

    print(f"\nHasil Rata-rata K-Fold untuk {sheet_name}:\n"
          f"MSE: {avg_mse:.4f} | R²: {avg_r2:.4f}")

    print("\nMelatih model final dengan seluruh data...")
    X_scaled_full = x_scaler.fit_transform(X)
    y_scaled_full = y_scaler.fit_transform(y)

    final_model = create_model(X_scaled_full.shape[1])
    final_model.fit(X_scaled_full, y_scaled_full, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    callbacks=[EarlyStopping(monitor='loss', patience=25), ReduceLROnPlateau(monitor='loss', patience=10)],
                    verbose=0)

    df['Prediksi'] = y_scaler.inverse_transform(final_model.predict(X_scaled_full))
    print("Model final selesai dilatih.")

    last_year_in_data = df['Waktu'].dt.year.max()
    last_year_df = df[df['Waktu'].dt.year == last_year_in_data]

    future_input = {}
    for feature in FEATURES:
        if feature != 'Usia Pakai (Tahun)':
            future_input[feature] = [last_year_df[feature].mean()]

    last_age = df['Usia Pakai (Tahun)'].iloc[-1]
    years_to_add = TARGET_YEAR - last_year_in_data
    future_input['Usia Pakai (Tahun)'] = [last_age + years_to_add]

    future_input_df = pd.DataFrame(future_input)[FEATURES]
    future_input_scaled = x_scaler.transform(future_input_df)
    predicted_rul_scaled = final_model.predict(future_input_scaled)
    predicted_rul_2030 = y_scaler.inverse_transform(predicted_rul_scaled)[0][0]

    future_predictions.append({
        'Transformator': sheet_name,
        'Prediksi Sisa Umur Tahun 2030': predicted_rul_2030
    })

    # Menyimpan hasil untuk laporan akhir
    all_results[sheet_name] = {
        'data': df.copy(),
        'metrics': {'MSE': avg_mse, 'R2': avg_r2},
        'fold_metrics': fold_metrics,
        'model': final_model,
        'predicted_rul_2030': predicted_rul_2030
    }

# --- PENYIMPANAN HASIL, VISUALISASI, DAN LAPORAN ---
print("\nMenyimpan hasil, model, dan semua grafik...")
output_dir = 'output'
per_transformator_dir = os.path.join(output_dir, 'per_transformator')
os.makedirs(per_transformator_dir, exist_ok=True)

for sheet_name, result in all_results.items():
    df_result = result['data']

    # Menyimpan data prediksi historis ke Excel
    df_output = df_result[['Waktu'] + FEATURES + [TARGET, 'Prediksi']]
    output_path = os.path.join(per_transformator_dir, f'prediksi_{sheet_name}_2021-2024.xlsx')
    with pd.ExcelWriter(output_path) as writer:
        df_output.to_excel(writer, index=False, sheet_name='Prediksi')
        pd.DataFrame(result['fold_metrics'], columns=['Fold', 'MSE', 'R2']).to_excel(writer, index=False, sheet_name='Evaluasi K-Fold')

    # Menyimpan model
    result['model'].save(os.path.join(output_dir, f'model_{sheet_name}.h5'))

    # Menmbuat dan menyimpan grafik perbandingan (2021-2024)
    plt.figure(figsize=(15, 7))
    plt.plot(df_result['Waktu'], df_result[TARGET], 'b-', label='Sisa Umur Aktual', linewidth=2)
    plt.plot(df_result['Waktu'], df_result['Prediksi'], 'r--', label='Sisa Umur Prediksi', linewidth=2, alpha=0.8)
    plt.title(f'Perbandingan Sisa Umur Aktual vs Prediksi - {sheet_name}', fontsize=16)
    plt.xlabel('Waktu', fontsize=12)
    plt.ylabel('Sisa Umur (tahun)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'grafik_perbandingan_{sheet_name}.png'), dpi=300)
    plt.close()

    # Menmbuat dan menyimpan GRAFIK PROYEKSI BARU ke tahun 2030
    predicted_rul_2030 = result['predicted_rul_2030']
    target_date_2030 = datetime(TARGET_YEAR, 12, 31)

    plt.figure(figsize=(15, 7))
    plt.plot(df_result['Waktu'], df_result[TARGET], 'b-', label='Sisa Umur Aktual (Historis)')
    plt.plot(df_result['Waktu'], df_result['Prediksi'], 'r-', label='Prediksi Model (Historis)')

    forecast_x = [df_result['Waktu'].iloc[-1], target_date_2030]
    forecast_y = [df_result['Prediksi'].iloc[-1], predicted_rul_2030]
    plt.plot(forecast_x, forecast_y, 'g--', label=f'Proyeksi ke {TARGET_YEAR}')

    plt.plot(target_date_2030, predicted_rul_2030, 'g*', markersize=15,
             label=f'Prediksi {TARGET_YEAR}: {predicted_rul_2030:.2f} Tahun')

    plt.title(f'Proyeksi Sisa Umur Transformator - {sheet_name}', fontsize=16)
    plt.xlabel('Waktu', fontsize=12)
    plt.ylabel('Sisa Umur (tahun)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'grafik_proyeksi_{sheet_name}.png'), dpi=300)
    plt.close()

# --- LAPORAN AKHIR DI KONSOL DAN FILE ---
report_df = pd.DataFrame([{
    'Transformator': name,
    'MSE Rata-rata (K-Fold)': res['metrics']['MSE'],
    'R² Rata-rata (K-Fold)': res['metrics']['R2'],
} for name, res in all_results.items()])
report_df.to_excel(os.path.join(output_dir, 'laporan_evaluasi_akhir.xlsx'), index=False)

future_report_df = pd.DataFrame(future_predictions)
future_report_df.to_excel(os.path.join(output_dir, 'laporan_prediksi_2030.xlsx'), index=False)

print("\n\n" + "="*95)
print("REKAPITULASI HASIL AKHIR EVALUASI MODEL")
print("="*95)
print(f"{'Transformator':<25} | {'MSE (K-Fold)':<20} | {'R² (K-Fold)':<20}")
print("-"*95)
for index, row in report_df.iterrows():
    print(f"{row['Transformator']:<25} | {row['MSE Rata-rata (K-Fold)']:<20.4f} | {row['R² Rata-rata (K-Fold)']:<20.4f}")
print("-"*95)

print("\n\n" + "="*95)
print(f"PREDIKSI SISA UMUR TRANSFORMATOR UNTUK TAHUN {TARGET_YEAR}")
print("="*95)
print(f"{'Transformator':<25} | {'Prediksi Sisa Umur {TARGET_YEAR} (Tahun)':<35}")
print("-"*95)
for index, row in future_report_df.iterrows():
    print(f"{row['Transformator']:<25} | {row['Prediksi Sisa Umur Tahun 2030']:>35.2f}")
print("-"*95)

# --- SELESAI ---
print(f"\nProses selesai! Semua hasil disimpan di folder '{output_dir}'")
print(f"Waktu selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")