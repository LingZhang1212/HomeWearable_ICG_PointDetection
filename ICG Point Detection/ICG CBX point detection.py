import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pywt
from PyEMD import EEMD
import pandas as pd
import os

from InitializeHRVparams import InitializeHRVparams
from ConvertRawDataToRRIntervals import ConvertRawDataToRRIntervals

# ==== 读取 Excel ECG/ICG 数据 ====
def load_ecg_icg_from_excel(filepath):
    df = pd.read_excel(filepath, header=None)
    ecg = df.iloc[:, 0].values.astype(float)
    icg = df.iloc[:, 1].values.astype(float)
    return ecg, icg

# ==== 自适应软阈值小波去噪 ====
def adaptive_soft_threshold(coeffs, sigma=None):
    if sigma is None:
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(coeffs[-1])))
    return [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]

def wavelet_denoise(signal, wavelet_name='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet=wavelet_name, level=level)
    coeffs_thresh = adaptive_soft_threshold(coeffs)
    return pywt.waverec(coeffs_thresh, wavelet=wavelet_name)

# ==== EEMD 去噪 ====
def eemd_denoise(signal, max_imfs=10):
    eemd = EEMD()
    imfs = eemd.eemd(signal)
    return np.sum(imfs[1:min(max_imfs, len(imfs))], axis=0)

# ==== LMS 滤波 ====
def lms_filter(signal, desired, mu=0.01, order=5):
    N = len(signal)
    w = np.zeros(order)
    y = np.zeros(N)
    for n in range(order, N):
        x = signal[n - order:n][::-1]
        y[n] = np.dot(w, x)
        e = desired[n] - y[n]
        w += 2 * mu * e * x
    return y

# ==== 三阶导数函数 ====
def third_derivative(signal):
    return np.gradient(np.gradient(np.gradient(signal)))

def detect_c_point_from_r(signal, r_idx, fs=1000):
    start = r_idx + int(0.08 * fs)
    end = r_idx + int(0.15 * fs)
    if end > len(signal): end = len(signal)
    region = signal[start:end]
    if len(region) == 0:
        return None
    peak_rel = np.argmax(region)
    return start + peak_rel

def detect_b_point_from_r(signal, r_idx, fs=1000):
    start = r_idx + int(0.01 * fs)
    end = r_idx + int(0.08 * fs)
    if end > len(signal): end = len(signal)
    region = third_derivative(signal[start:end])
    if len(region) == 0:
        return None
    b_rel = np.argmin(region)
    return start + b_rel

def detect_x_point_from_r(signal, r_idx, fs=1000):
    start = r_idx + int(0.20 * fs)
    end = r_idx + int(0.35 * fs)
    if end > len(signal): end = len(signal)
    region = signal[start:end]
    if len(region) == 0:
        return None
    x_rel = np.argmin(region)
    return start + x_rel

def extract_bcx_points_from_beats(beats_denoised):
    b_list, c_list, x_list = [], [], []
    for beat in beats_denoised:
        N = len(beat)
        try:
            c_start, c_end = int(0.6*N), int(0.8*N)
            c_idx = np.argmax(beat[c_start:c_end]) + c_start

            # 提前B点窗口起始 & 限制查找区间不得靠近C
            b_start = int(0.05 * N)
            b_end = c_idx - int(0.05 * N)
            b_region = third_derivative(beat[b_start:b_end])
            b_idx = np.argmin(b_region) + b_start

            x_start = c_idx + int(0.05 * N)
            x_end = int(0.95 * N)
            x_idx = np.argmin(beat[x_start:x_end]) + x_start
        except:
            b_idx, c_idx, x_idx = None, None, None
        b_list.append(b_idx)
        c_list.append(c_idx)
        x_list.append(x_idx)
    return np.array(b_list), np.array(c_list), np.array(x_list)





# ==== 主处理流程 ====
def process_with_ecg_toolbox(ecg, clean_icg, fs=1000):
    HRVparams = InitializeHRVparams('Excel_ECG_ICG')
    HRVparams['Fs'] = fs

    b, a = butter(4, [0.5 / (fs / 2), 40 / (fs / 2)], btype='band')
    filtered_icg = filtfilt(b, a, clean_icg)

    print("Running ConvertRawDataToRRIntervals to get R peaks...")
    _, rr, R_pk, _, _ = ConvertRawDataToRRIntervals(ecg, HRVparams, subjectID="real_data")

    RR_intervals = np.diff(R_pk)
    median_RR = int(np.ceil(np.median(RR_intervals)))
    llim_beat = int(0.15 * fs)
    ulim_beat = median_RR - llim_beat
    beat_len = llim_beat + ulim_beat

    beat_segments_clean = []
    beat_segments_denoised = []

    for r in R_pk:
        start = r - llim_beat
        end = r + ulim_beat
        if start < 0 or end > len(clean_icg):
            continue
        icg_seg = filtered_icg[start:end]
        db4_out = wavelet_denoise(icg_seg, wavelet_name='db4')
        sym8_out = wavelet_denoise(db4_out, wavelet_name='sym8')
        eemd_out = eemd_denoise(sym8_out)
        lms_out = lms_filter(eemd_out, icg_seg)

        beat_segments_clean.append(icg_seg)
        beat_segments_denoised.append(lms_out)

    denoised_icg_full = np.zeros_like(clean_icg)
    counts = np.zeros_like(clean_icg)
    valid_R_peaks = [r for r in R_pk if r - llim_beat >= 0 and r + ulim_beat <= len(clean_icg)]

    for idx, r in enumerate(valid_R_peaks):
        start = r - llim_beat
        end = r + ulim_beat
        denoised_icg_full[start:end] += beat_segments_denoised[idx]
        counts[start:end] += 1

    counts[counts == 0] = 1
    denoised_icg_full /= counts

    return np.array(beat_segments_clean), np.array(beat_segments_denoised), beat_len, filtered_icg, denoised_icg_full, valid_R_peaks

# ==== 主程序入口 ====
if __name__ == "__main__":
    filepath = r"C:\Users\LingZhang\Desktop\ECG ICG\ECG_ICG\ICG Point Detection\RawData_Subject_1_task_BL_converted.xlsx"
    output_dir = r"C:\Users\LingZhang\Desktop\ECG ICG\ECG_ICG\ICG Point Detection"
    os.makedirs(output_dir, exist_ok=True)

    ecg, icg = load_ecg_icg_from_excel(filepath)
    beats_clean, beats_denoised, beat_len, filtered_icg, denoised_icg_full, valid_R_peaks = process_with_ecg_toolbox(ecg, icg, fs=1000)

    avg_denoised = np.mean(beats_denoised, axis=0)
    b_points_rel, c_points_rel, x_points_rel = extract_bcx_points_from_beats(beats_denoised)  # 相对索引
    avg_denoised = np.mean(beats_denoised, axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(avg_denoised, label="Avg Denoised ICG", linewidth=2)
    plt.axvline(np.mean(b_points_rel), color='r', linestyle='--', label='B (mean)')
    plt.axvline(np.mean(c_points_rel), color='g', linestyle='--', label='C (mean)')
    plt.axvline(np.mean(x_points_rel), color='b', linestyle='--', label='X (mean)')

    plt.title("Avg ICG Beat with BCX Feature Points")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_icg_with_bcx.png"), dpi=300)
    plt.show()

    # ==== 连续信号图 ====
    plt.figure(figsize=(16, 6))
    plt.plot(icg, label='Raw ICG', alpha=0.4)
    plt.plot(filtered_icg, label='Filtered ICG (bandpass)', alpha=0.6)
    plt.plot(denoised_icg_full, label='Denoised ICG (full signal)', linewidth=1.5)
    plt.title("Full ICG Signal Before and After Denoising")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "full_denoised_icg.png"), dpi=300)
    plt.show()
