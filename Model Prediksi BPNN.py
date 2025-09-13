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
from pandas.tseries.offsets import DateOffset
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Konfigurasi global seed
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Konfigurasi & Inisialisasi
FILE_PATH = 'D:/PREDIKSI SISA UMUR TRAFO/dataset_bpnn_transformator.xlsx'
SHEET_NAMES = ['Transformator 1', 'Transformator 2', 'Transformator 3', 'Transformator 4']
FEATURES = ['Rata-Rata Beban (MW)', 'Kapasitas Beban (%)', 'Suhu Lingkungan (℃)', 'Suhu Hotspot (℃)',
            'Laju Penuaan Thermal (p.u)', 'TDCG (ppm)', 'Kadar Air (ppm)', 'BDV (kV/2.5 mm)',
            'Kadar Asam (mgKOH/g)', 'Usia Pakai (Tahun)']
TARGET = 'Sisa Umur'
K_FOLDS = 10
EPOCHS = 1000
BATCH_SIZE = 8
LEARNING_RATE = 0.03
all_results = {}

# Fungsi untuk membuat model BPNN
def create_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='mse',
                  metrics=['mse'])
    return model

# Proses setiap transformator
for sheet_name in SHEET_NAMES:
    print(f"\n{'='*50}")
    print(f"Memproses: {sheet_name}")
    print(f"{'='*50}")

    if not os.path.exists(FILE_PATH):
        print(f"Error: File tidak ditemukan di '{FILE_PATH}'")
        continue

    df = pd.read_excel(FILE_PATH, sheet_name=sheet_name)
    df['Waktu'] = pd.to_datetime(df['Waktu'])
    df = df.sort_values(by='Waktu').reset_index(drop=True)

    X = df[FEATURES].values
    y = df[TARGET].values.reshape(-1, 1)

    x_scaler = StandardScaler()
    y_scaler = MinMaxScaler()

    # K-FOLD CROSS-VALIDATION
    print("Memulai K-Fold Cross-Validation...")
    X_scaled_kfold = x_scaler.fit_transform(X)
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = {'mse': [], 'r2': []}
    fold_metrics_detail = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled_kfold)):
        print(f"Fold {fold+1}/{K_FOLDS}...")
        X_train, X_val = X_scaled_kfold[train_idx], X_scaled_kfold[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_val_scaled = y_scaler.transform(y_val)
        
        model = create_model(X_train.shape[1])
        early_stop_kfold = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
        reduce_lr_kfold = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
        
        model.fit(X_train, y_train_scaled, validation_data=(X_val, y_val_scaled), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop_kfold, reduce_lr_kfold], verbose=0)
        
        y_pred_original = y_scaler.inverse_transform(model.predict(X_val)).clip(0)
        mse, r2 = mean_squared_error(y_val, y_pred_original), r2_score(y_val, y_pred_original)
        
        fold_results['mse'].append(mse)
        fold_results['r2'].append(r2)
        fold_metrics_detail.append({'Fold': fold + 1, 'MSE': mse, 'R2': r2})
        
    avg_mse_kfold = np.mean(fold_results['mse'])
    avg_r2_kfold = np.mean(fold_results['r2'])
    print(f"\nHasil Rata-rata K-Fold:\nMSE: {avg_mse_kfold:.4f} | R²: {avg_r2_kfold:.4f}")

    # PELATIHAN MODEL FINAL
    print("\nMelatih model final dengan seluruh data...")
    X_scaled_full = x_scaler.fit_transform(X)
    y_scaled_full = y_scaler.fit_transform(y)
    final_model = create_model(X_scaled_full.shape[1])

    early_stop_final = EarlyStopping(monitor='loss', patience=25, restore_best_weights=True)
    reduce_lr_final = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_lr=1e-6)

    final_model.fit(X_scaled_full, y_scaled_full,
                      epochs=EPOCHS,
                      batch_size=BATCH_SIZE,
                      callbacks=[early_stop_final, reduce_lr_final],
                      verbose=0)

    # PREDIKSI HISTORIS & MASA DEPAN
    pred_hist_nn_scaled = final_model.predict(X_scaled_full)
    pred_hist_nn_original = y_scaler.inverse_transform(pred_hist_nn_scaled)
    df['Prediksi_Aktual_Model'] = pred_hist_nn_original.clip(0)

    print("Membuat tren historis yang mulus (polinomial)...")
    time_numeric_hist = np.arange(len(df))
    coeffs = np.polyfit(time_numeric_hist, df['Prediksi_Aktual_Model'], 2)
    polynomial = np.poly1d(coeffs)
    df['Prediksi_Tren_Mulus'] = polynomial(time_numeric_hist).clip(0)

    print("Membuat proyeksi masa depan dengan kurva menurun yang realistis...")
    last_date = df['Waktu'].max()
    future_dates = pd.to_datetime(pd.date_range(start=last_date + DateOffset(months=1), end='2030-12-31', freq='MS'))
    df_future = pd.DataFrame({'Waktu': future_dates})
    
    last_y = df['Prediksi_Tren_Mulus'].iloc[-1]
    current_slope = polynomial.deriv()(time_numeric_hist[-1])
    slope_decay_factor = 0.998
    future_predictions = []
    current_y = last_y
    for _ in range(len(df_future)):
        current_y += current_slope
        future_predictions.append(current_y)
        current_slope *= slope_decay_factor
        
    df_future['Prediksi_Masa_Depan'] = np.array(future_predictions).clip(0)

    mse_final = mean_squared_error(df[TARGET], df['Prediksi_Aktual_Model'])
    r2_final = r2_score(df[TARGET], df['Prediksi_Aktual_Model'])
    print(f"Metrik Evaluasi Model Final -> MSE: {mse_final:.4f}, R²: {r2_final:.4f}")

    all_results[sheet_name] = {
        'data_hist': df.copy(),
        'data_future': df_future.copy(),
        'metrics_final': {'MSE_Final': mse_final, 'R2_Final': r2_final},
        'metrics_kfold_avg': {'MSE_KFold_Avg': avg_mse_kfold, 'R2_KFold_Avg': avg_r2_kfold},
        'metrics_kfold_detail': fold_metrics_detail,
        'model': final_model
    }
    print("Model final dan semua prediksi selesai dibuat.")


# PENYIMPANAN HASIL DAN VISUALISASI
print("\nMenyimpan hasil prediksi, model, dan grafik final...")
output_dir = 'output_final_paling_akurat'
os.makedirs(output_dir, exist_ok=True)

# --- GRAFIK 1: Prediksi Sisa Umur ---
for sheet_name, result in all_results.items():
    df_hist = result['data_hist']
    df_future = result['data_future']
    metrics_final_df = pd.DataFrame([result['metrics_final']])
    metrics_kfold_df = pd.DataFrame(result['metrics_kfold_detail'])
    model_to_save = result['model']

    model_to_save.save(os.path.join(output_dir, f'model_{sheet_name}.h5'))

    output_path = os.path.join(output_dir, f'hasil_prediksi_{sheet_name}.xlsx')
    with pd.ExcelWriter(output_path) as writer:
        df_hist[['Waktu', TARGET, 'Prediksi_Tren_Mulus']].to_excel(writer, index=False, sheet_name='Historis_dan_Tren_Prediksi')
        df_future.to_excel(writer, index=False, sheet_name='Proyeksi_Masa_Depan')
        metrics_final_df.to_excel(writer, index=False, sheet_name='Metrik_Evaluasi_Final')
        metrics_kfold_df.to_excel(writer, index=False, sheet_name='Evaluasi_K-Fold_Detail')

    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(df_hist['Waktu'], df_hist[TARGET], 'b-', label='Sisa Umur Aktual', linewidth=2, zorder=3)
    ax.plot(df_hist['Waktu'], df_hist['Prediksi_Tren_Mulus'], 'r--', label='Sisa Umur Prediksi', linewidth=2, zorder=3)
    ax.plot(df_hist['Waktu'], df_hist[TARGET], 'bo', markersize=7, zorder=3)
    ax.plot(df_hist['Waktu'], df_hist['Prediksi_Tren_Mulus'], 'r^', markersize=7, zorder=3)

    y_min = min(df_hist[TARGET].min(), df_hist['Prediksi_Tren_Mulus'].min())
    y_max = max(df_hist[TARGET].max(), df_hist['Prediksi_Tren_Mulus'].max())
    padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - padding, y_max + padding)
    bottom_y = ax.get_ylim()[0]

    ax.vlines(df_hist['Waktu'], ymin=bottom_y, ymax=df_hist[TARGET], colors='b', linestyles='solid', alpha=0.4, zorder=1)
    ax.vlines(df_hist['Waktu'], ymin=bottom_y, ymax=df_hist['Prediksi_Tren_Mulus'], colors='r', linestyles='dashed', alpha=0.4, zorder=1)
    start_date = pd.Timestamp('2021-01-01')
    end_date = pd.Timestamp('2024-12-31')
    ax.set_xlim(start_date, end_date)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) 
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_title(f'Grafik Perbandingan Aktual vs Prediksi - {sheet_name} (2021-2024)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Waktu', fontsize=12)
    ax.set_ylabel('Sisa Umur (Tahun)', fontsize=12)
    ax.legend(fontsize=12)

    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'1_grafik_historis_final_{sheet_name}.png'), dpi=300)
    plt.close(fig)

    # --- GRAFIK 2: Proyeksi Sisa Umur ---
    last_hist_point = pd.DataFrame({'Waktu': [df_hist['Waktu'].iloc[-1]], 'Prediksi_Masa_Depan': [df_hist['Prediksi_Tren_Mulus'].iloc[-1]]})
    plot_future_data = pd.concat([last_hist_point, df_future[['Waktu', 'Prediksi_Masa_Depan']]])
    
    plt.figure(figsize=(15, 7))
    plt.plot(plot_future_data['Waktu'], plot_future_data['Prediksi_Masa_Depan'], 'r--', label='Prediksi Sisa Umur', linewidth=2.5)
    
    start_date_future = plot_future_data['Waktu'].min()
    end_date_future = pd.Timestamp('2030-12-31')
    plt.xlim(start_date_future, end_date_future)

    visible_data = plot_future_data[(plot_future_data['Waktu'] >= start_date_future) & (plot_future_data['Waktu'] <= end_date_future)]
    y_min_future = visible_data['Prediksi_Masa_Depan'].min()
    y_max_future = visible_data['Prediksi_Masa_Depan'].max()
    plt.ylim(y_min_future - 0.2, y_max_future + 0.2)
    
    plt.title(f'Grafik Prediksi Sisa Umur - {sheet_name} (Hingga 2030)', fontsize=16)
    plt.xlabel('Waktu', fontsize=12)
    plt.ylabel('Sisa Umur (tahun)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'2_grafik_proyeksi_final_{sheet_name}.png'), dpi=300)
    plt.close()

# Laporan Rekapitulasi
report_data = []
for name, res in all_results.items():
    res_dict = {'Transformator': name}
    res_dict.update(res['metrics_kfold_avg'])
    res_dict.update(res['metrics_final'])
    report_data.append(res_dict)
report_df = pd.DataFrame(report_data)
report_df.to_excel(os.path.join(output_dir, 'laporan_rekapitulasi_evaluasi.xlsx'), index=False)
print("\n\n" + "="*85)
print("REKAPITULASI HASIL AKHIR EVALUASI MODEL")
print("="*85)
print(f"{'Transformator':<25} | {'MSE (K-Fold)':<15} | {'R² (K-Fold)':<15} | {'MSE (Final)':<12} | {'R² (Final)':<12}")
print("-"*85)
for index, row in report_df.iterrows():
    print(f"{row['Transformator']:<25} | {row['MSE_KFold_Avg']:>14.4f} | {row['R2_KFold_Avg']:>14.4f} | {row['MSE_Final']:>11.4f} | {row['R2_Final']:>11.4f}")
print("-"*85)
print(f"\nProses selesai! Semua hasil disimpan di folder '{output_dir}'")
print(f"Waktu selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")