import csv
import os
import time
import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags


def safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else np.nan


def safe_var(x):
    return float(np.var(x, ddof=0)) if len(x) > 0 else np.nan


def safe_std(x):
    return float(np.std(x, ddof=0)) if len(x) > 0 else np.nan


# ========================================================
# CONFIG
# ========================================================

TX_PORT = "COM4"
RX_PORT = "COM3"
BAUD = 115200

UPS = 8
SKIP_SYMBOLS = 100
TX_TOTAL_SYMBOLS = 100_000
RX_CAPTURE_POINTS = 100_000
RX_EXPECTED_N = RX_CAPTURE_POINTS

# número de símbolos úteis dentro da captura RX
N_SYMBOLS_ANALYSIS = RX_CAPTURE_POINTS // UPS

VREF = 3.3
ADC_MAX = 4095.0
V_MID = VREF / 2.0

STARTUP_WAIT = 3.0
TIMEOUT_CMD = None
TIMEOUT_DUMP = None
PRINT_EVERY = 0.5

SHOW_MAX_SAMPLES_X = 20000
PEAK_REL_THRESHOLD = 0.60

USE_MATCHED_FILTER = True
REMOVE_DC = True

QAM = 64
SQRT_QAM = 8
SIGMA = 15.0
N_TARGET = 10.0

ALPHA = 0.4
NTAPS = 10 * UPS + 1
GROUP_DELAY = (NTAPS - 1) // 2

FIXED_SEED = 123456789

SAVE_FIGURES = True
FIG_PREFIX = "ensaio_100k"
RESULTS_CSV = "ensaios_resultados_100k.csv"
SUMMARY_TXT_PREFIX = "stats_ensaio_100k"

# ========================================================
# PASTA NO AMBIENTE DE TRABALHO
# ========================================================

DESKTOP_DIR = os.path.join(os.path.expanduser("~"), "Desktop")
OUTPUT_DIR = os.path.join(DESKTOP_DIR, "ensaios")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(OUTPUT_DIR, RESULTS_CSV)
SUMMARY_TXT_PREFIX = os.path.join(OUTPUT_DIR, SUMMARY_TXT_PREFIX)
FIG_PREFIX = os.path.join(OUTPUT_DIR, FIG_PREFIX)


# ========================================================
# AUX
# ========================================================

def parse_int_pair(line):
    try:
        a, b = line.split(",")
        return int(a), int(b)
    except Exception:
        return None


def parse_keyvals(lines):
    out = {}
    for line in lines:
        parts = line.replace(",", " ").split()
        for part in parts:
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def parse_tx_rate_line(line):
    if not line.startswith("TX_RATE"):
        return None

    out = {}
    parts = line.strip().split()

    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                out[k.strip()] = v.strip()

    return out



def read_lines(ser, duration=0.3):
    t0 = time.time()
    lines = []
    while time.time() - t0 < duration:
        if ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                lines.append(line)
    return lines


def wait_for_text(ser, target, timeout=None):
    t0 = time.time()
    seen = []
    while True:
        if ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                seen.append(line)
                print(f"[{ser.port}] {line}")
                if target in line:
                    return True, seen

        if timeout is not None and (time.time() - t0 > timeout):
            return False, seen


def capture_dump(ser, name, expected_n=None, timeout=None):
    data = []
    started = False
    t0 = time.time()
    t_last = time.time()

    while True:
        if ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            if line == "BEGIN":
                started = True
                print(f"{name} BEGIN")
                continue

            if line == "END":
                print(f"{name} END")
                break

            if started:
                p = parse_int_pair(line)
                if p is not None:
                    data.append(p)

        if time.time() - t_last > PRINT_EVERY:
            if expected_n is None:
                print(f"{name} dump: {len(data)}")
            else:
                print(f"{name} dump: {len(data)}/{expected_n}")
            t_last = time.time()

        if timeout is not None and (time.time() - t0 > timeout):
            raise TimeoutError(f"Timeout à espera do dump de {name}.")

    return data


def make_rrc(alpha, ups, ntaps):
    T = float(ups)
    eps = 1e-12
    h = np.zeros(ntaps, dtype=float)

    for i in range(ntaps):
        t = i - (ntaps - 1) / 2.0

        if abs(t) < eps:
            h[i] = 1.0 + alpha * (4.0 / np.pi - 1.0)
        elif abs(abs(t) - T / (4.0 * alpha)) < eps:
            h[i] = (alpha / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha)) +
                (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
            )
        else:
            num = (
                np.sin(np.pi * t * (1.0 - alpha) / T) +
                4.0 * alpha * t / T * np.cos(np.pi * t * (1.0 + alpha) / T)
            )
            den = np.pi * t / T * (1.0 - (4.0 * alpha * t / T) ** 2.0)
            h[i] = num / den

    energy = np.sqrt(np.sum(h * h))
    if energy > 1e-15:
        h /= energy
    return h


def apply_fir_same(x, h):
    return np.convolve(np.asarray(x, dtype=float), np.asarray(h, dtype=float), mode="same")


def adc_to_volts(x):
    return np.asarray(x, dtype=float) * VREF / ADC_MAX


def remove_dc_volts(xv):
    return xv - V_MID if REMOVE_DC else xv.copy()


def affine_fit(x_ref, x):
    x_ref = np.asarray(x_ref, dtype=float)
    x = np.asarray(x, dtype=float)

    L = min(len(x_ref), len(x))
    x_ref = x_ref[:L]
    x = x[:L]

    if L < 10:
        return x.copy(), 1.0, 0.0

    x_mean = np.mean(x)
    y_mean = np.mean(x_ref)

    x0 = x - x_mean
    y0 = x_ref - y_mean

    den = np.dot(x0, x0)
    if den < 1e-12:
        return x.copy(), 1.0, 0.0

    a = np.dot(x0, y0) / den
    b = y_mean - a * x_mean

    return a * x + b, float(a), float(b)


def best_sampling_offset_energy(rxI_frame, rxQ_frame, ups):
    best_off = 0
    best_metric = -1.0
    metrics = []

    for off in range(ups):
        xI = rxI_frame[off::ups]
        xQ = rxQ_frame[off::ups]

        L = min(len(xI), len(xQ))
        if L < 10:
            metrics.append(0.0)
            continue

        mag2 = xI[:L] ** 2 + xQ[:L] ** 2
        metric = np.mean(mag2)
        metrics.append(metric)

        if metric > best_metric:
            best_metric = metric
            best_off = off

    return int(best_off), np.array(metrics, dtype=float)


def corr_metric_against_tx_symbols_complex(txI, txQ, rxI_frame, rxQ_frame, ups):
    offs = []
    metrics = []

    txI = np.asarray(txI, dtype=float)
    txQ = np.asarray(txQ, dtype=float)

    for off in range(ups):
        rxI = np.asarray(rxI_frame[off::ups][:len(txI)], dtype=float)
        rxQ = np.asarray(rxQ_frame[off::ups][:len(txQ)], dtype=float)

        L = min(len(txI), len(txQ), len(rxI), len(rxQ))
        if L < 20:
            offs.append(off)
            metrics.append(0.0)
            continue

        tx = txI[:L] + 1j * txQ[:L]
        rx = rxI[:L] + 1j * rxQ[:L]

        tx0 = tx - np.mean(tx)
        rx0 = rx - np.mean(rx)

        den = np.linalg.norm(tx0) * np.linalg.norm(rx0)
        metric = abs(np.vdot(tx0, rx0)) / den if den > 1e-12 else 0.0

        offs.append(off)
        metrics.append(metric)

    offs = np.array(offs, dtype=int)
    metrics = np.array(metrics, dtype=float)
    best_off = int(offs[np.argmax(metrics)]) if len(metrics) else 0
    return best_off, offs, metrics


def build_constellation_mb():
    levels = np.array([i - (SQRT_QAM - 1) / 2.0 for i in range(SQRT_QAM)], dtype=float)

    valuesI = np.zeros(QAM, dtype=float)
    valuesQ = np.zeros(QAM, dtype=float)

    k = 0
    for q in range(SQRT_QAM):
        for i in range(SQRT_QAM):
            valuesI[k] = levels[i]
            valuesQ[k] = levels[q]
            k += 1

    r2 = valuesI ** 2 + valuesQ ** 2
    probs = np.exp(-r2 / SIGMA)
    probs /= np.sum(probs)

    n_base = np.sum(probs * r2)
    scale = np.sqrt(N_TARGET / n_base)

    valuesI *= scale
    valuesQ *= scale

    cdf = np.cumsum(probs)
    cdf[-1] = 1.0
    return valuesI, valuesQ, probs, cdf


def lcg_sequence(n, seed=FIXED_SEED):
    state = np.uint32(seed)
    out = np.empty(n, dtype=np.uint32)
    for i in range(n):
        state = np.uint32((1664525 * int(state) + 1013904223) & 0xFFFFFFFF)
        out[i] = state
    return out


def sample_symbols_mb(n_symbols, seed=FIXED_SEED):
    valuesI, valuesQ, _, cdf = build_constellation_mb()
    states = lcg_sequence(n_symbols, seed=seed)
    u24 = ((states >> 8) & 0x00FFFFFF).astype(np.float64)
    u = u24 / 16777216.0
    idx = np.searchsorted(cdf, u, side="left")
    symI = valuesI[idx]
    symQ = valuesQ[idx]
    return symI, symQ


def synthesize_tx_waveform_from_symbols(symI, symQ, ups, rrc):
    n_symbols = len(symI)
    n_points = n_symbols * ups

    upI = np.zeros(n_points, dtype=float)
    upQ = np.zeros(n_points, dtype=float)
    upI[::ups] = symI
    upQ[::ups] = symQ

    seqI = np.convolve(upI, rrc, mode="full")[:n_points]
    seqQ = np.convolve(upQ, rrc, mode="full")[:n_points]
    return seqI, seqQ


def acquire_once(tx, rx):
    print("A reiniciar TX e captura RX...")

    rx.write(b'r')
    rx.flush()
    time.sleep(0.15)

    rx.write(b'c')
    rx.flush()
    time.sleep(0.15)

    tx.write(b'r')
    tx.flush()

    ok, _ = wait_for_text(rx, "RX_CAPTURE_DONE", TIMEOUT_CMD)
    if not ok:
        raise TimeoutError("RX não terminou a captura.")

    print("A pedir dump ao RX...")
    rx.write(b'd')
    rx.flush()
    rx_d = capture_dump(rx, 'RX', expected_n=RX_EXPECTED_N, timeout=TIMEOUT_DUMP)

    if len(rx_d) == 0:
        raise RuntimeError("Não chegaram dados do RX.")

    return np.array(rx_d, dtype=float)


def analyse_once(rx_d):
    rrc = make_rrc(ALPHA, UPS, NTAPS)

    adcI_raw_v = adc_to_volts(rx_d[:, 0])
    adcQ_raw_v = adc_to_volts(rx_d[:, 1])

    rx_raw_I_bb = remove_dc_volts(adcI_raw_v)
    rx_raw_Q_bb = remove_dc_volts(adcQ_raw_v)

    if USE_MATCHED_FILTER:
        rx_proc_I = apply_fir_same(rx_raw_I_bb, rrc)
        rx_proc_Q = apply_fir_same(rx_raw_Q_bb, rrc)
        proc_label = "baseband + matched filter"
    else:
        rx_proc_I = rx_raw_I_bb.copy()
        rx_proc_Q = rx_raw_Q_bb.copy()
        proc_label = "baseband sem matched filter"

    if len(rx_proc_I) > GROUP_DELAY:
        rx_proc_I = rx_proc_I[GROUP_DELAY:]
        rx_proc_Q = rx_proc_Q[GROUP_DELAY:]

    n_needed_syms = SKIP_SYMBOLS + N_SYMBOLS_ANALYSIS + 32
    tx_symI_all, tx_symQ_all = sample_symbols_mb(n_needed_syms, seed=FIXED_SEED)

    tx_symbols_I = tx_symI_all[SKIP_SYMBOLS:SKIP_SYMBOLS + N_SYMBOLS_ANALYSIS]
    tx_symbols_Q = tx_symQ_all[SKIP_SYMBOLS:SKIP_SYMBOLS + N_SYMBOLS_ANALYSIS]

    tx_seq_I, tx_seq_Q = synthesize_tx_waveform_from_symbols(
        tx_symI_all[:SKIP_SYMBOLS + N_SYMBOLS_ANALYSIS + 16],
        tx_symQ_all[:SKIP_SYMBOLS + N_SYMBOLS_ANALYSIS + 16],
        UPS,
        rrc
    )

    tx_capture_ref_I = tx_seq_I[SKIP_SYMBOLS * UPS:SKIP_SYMBOLS * UPS + RX_CAPTURE_POINTS]
    tx_capture_ref_Q = tx_seq_Q[SKIP_SYMBOLS * UPS:SKIP_SYMBOLS * UPS + RX_CAPTURE_POINTS]

    if USE_MATCHED_FILTER:
        tx_capture_ref_I = apply_fir_same(tx_capture_ref_I, rrc)
        tx_capture_ref_Q = apply_fir_same(tx_capture_ref_Q, rrc)

    if len(tx_capture_ref_I) > GROUP_DELAY:
        tx_capture_ref_I = tx_capture_ref_I[GROUP_DELAY:]
        tx_capture_ref_Q = tx_capture_ref_Q[GROUP_DELAY:]

    Lsamp = min(len(tx_capture_ref_I), len(rx_proc_I), len(tx_capture_ref_Q), len(rx_proc_Q))
    tx_capture_ref_I = tx_capture_ref_I[:Lsamp]
    tx_capture_ref_Q = tx_capture_ref_Q[:Lsamp]
    rx_proc_I = rx_proc_I[:Lsamp]
    rx_proc_Q = rx_proc_Q[:Lsamp]

    corr = correlate(rx_proc_I - np.mean(rx_proc_I), tx_capture_ref_I - np.mean(tx_capture_ref_I), mode="full")
    lags = correlation_lags(len(rx_proc_I), len(tx_capture_ref_I), mode="full")
    idx_peak = int(np.argmax(np.abs(corr)))
    lag = int(lags[idx_peak])
    corr_peak_abs = float(np.abs(corr[idx_peak]))
    den_global = np.linalg.norm(rx_proc_I - np.mean(rx_proc_I)) * np.linalg.norm(tx_capture_ref_I - np.mean(tx_capture_ref_I))
    corr_peak_norm = float(corr_peak_abs / den_global) if den_global > 1e-12 else 0.0

    if lag >= 0:
        rxI_frame = rx_proc_I[lag:]
        rxQ_frame = rx_proc_Q[lag:]
        refI = tx_capture_ref_I[:len(rxI_frame)]
        refQ = tx_capture_ref_Q[:len(rxQ_frame)]
    else:
        refI = tx_capture_ref_I[-lag:]
        refQ = tx_capture_ref_Q[-lag:]
        rxI_frame = rx_proc_I[:len(refI)]
        rxQ_frame = rx_proc_Q[:len(refQ)]

    L = min(len(refI), len(refQ), len(rxI_frame), len(rxQ_frame))
    refI = refI[:L]
    refQ = refQ[:L]
    rxI_frame = rxI_frame[:L]
    rxQ_frame = rxQ_frame[:L]

    best_off_energy, off_energy = best_sampling_offset_energy(rxI_frame, rxQ_frame, UPS)
    best_off_corr, offs_corr, corr_vs_offset = corr_metric_against_tx_symbols_complex(
        tx_symbols_I, tx_symbols_Q, rxI_frame, rxQ_frame, UPS
    )
    best_off = best_off_corr

    rxI_sym = rxI_frame[best_off::UPS][:N_SYMBOLS_ANALYSIS]
    rxQ_sym = rxQ_frame[best_off::UPS][:N_SYMBOLS_ANALYSIS]

    Lsym = min(len(tx_symbols_I), len(tx_symbols_Q), len(rxI_sym), len(rxQ_sym))
    txI_sym_aligned = tx_symbols_I[:Lsym]
    txQ_sym_aligned = tx_symbols_Q[:Lsym]
    rxI_sym_aligned = rxI_sym[:Lsym]
    rxQ_sym_aligned = rxQ_sym[:Lsym]

    rxI_sym_fit, fit_aI, fit_bI = affine_fit(txI_sym_aligned, rxI_sym_aligned)
    rxQ_sym_fit, fit_aQ, fit_bQ = affine_fit(txQ_sym_aligned, rxQ_sym_aligned)

    errI = rxI_frame[:len(refI)] - refI
    errQ = rxQ_frame[:len(refQ)] - refQ
    errI_sym = rxI_sym_fit - txI_sym_aligned
    errQ_sym = rxQ_sym_fit - txQ_sym_aligned
    noise_sym_mag = np.sqrt(errI_sym ** 2 + errQ_sym ** 2)

    corr_time_I_abs = np.array([])
    corr_time_Q_abs = np.array([])
    peaks_I = np.array([], dtype=int)
    peaks_Q = np.array([], dtype=int)
    x_time = np.arange(len(rx_proc_I))

    evm_rms = float(np.sqrt(np.mean(errI_sym ** 2 + errQ_sym ** 2))) if len(errI_sym) else np.nan
    tx_rms = float(np.sqrt(np.mean(txI_sym_aligned ** 2 + txQ_sym_aligned ** 2))) if len(txI_sym_aligned) else np.nan
    snr_num = np.mean(txI_sym_aligned ** 2 + txQ_sym_aligned ** 2) if len(txI_sym_aligned) else np.nan
    snr_den = np.mean(errI_sym ** 2 + errQ_sym ** 2) if len(errI_sym) else np.nan
    evm_pct = float(100.0 * evm_rms / tx_rms) if tx_rms and tx_rms > 1e-12 else np.nan
    snr_est_db = float(10.0 * np.log10(snr_num / snr_den)) if np.isfinite(snr_num) and np.isfinite(snr_den) and snr_den > 1e-15 else np.nan

    return {
        "proc_label": proc_label,
        "lags": lags,
        "corr": corr,
        "lag": lag,
        "corr_peak_abs": corr_peak_abs,
        "corr_peak_norm": corr_peak_norm,
        "lag_mod_ups": lag % UPS if UPS > 0 else 0,
        "alinhamento_txt": f"Lag residual estimado = {lag} amostras",
        "L": L,
        "offs_corr": offs_corr,
        "corr_vs_offset": corr_vs_offset,
        "best_off": best_off,
        "best_off_energy": best_off_energy,
        "best_metric_complex": float(corr_vs_offset[best_off]) if len(corr_vs_offset) > best_off else 0.0,
        "off_energy": off_energy,
        "fit_aI": fit_aI,
        "fit_bI": fit_bI,
        "fit_aQ": fit_aQ,
        "fit_bQ": fit_bQ,
        "frame_start": lag,
        "refI": refI,
        "refQ": refQ,
        "rxI": rxI_frame,
        "rxQ": rxQ_frame,
        "tx_symbols_I": tx_symbols_I,
        "tx_symbols_Q": tx_symbols_Q,
        "txI_sym_aligned": txI_sym_aligned,
        "txQ_sym_aligned": txQ_sym_aligned,
        "rxI_sym_aligned": rxI_sym_aligned,
        "rxQ_sym_aligned": rxQ_sym_aligned,
        "rxI_sym_fit": rxI_sym_fit,
        "rxQ_sym_fit": rxQ_sym_fit,
        "Ls": Lsym,
        "errI": errI,
        "errQ": errQ,
        "errI_sym": errI_sym,
        "errQ_sym": errQ_sym,
        "noise_var_I_sym": safe_var(errI_sym),
        "noise_var_Q_sym": safe_var(errQ_sym),
        "noise_var_mag_sym": safe_var(noise_sym_mag),
        "noise_std_I_sym": safe_std(errI_sym),
        "noise_std_Q_sym": safe_std(errQ_sym),
        "noise_std_mag_sym": safe_std(noise_sym_mag),
        "noise_mean_I_sym": safe_mean(errI_sym),
        "noise_mean_Q_sym": safe_mean(errQ_sym),
        "evm_rms": evm_rms,
        "evm_pct": evm_pct,
        "snr_est_db": snr_est_db,
        "corr_time_I_abs": corr_time_I_abs,
        "corr_time_Q_abs": corr_time_Q_abs,
        "x_time": x_time,
        "peaks_I": peaks_I,
        "peaks_Q": peaks_Q,
    }


def print_result(res):
    print("\n=========================")
    print("RESULTADO SINGLE RUN")
    print("=========================")
    print(f"Modo de processamento: {res['proc_label']}")
    print(f"Lag residual encontrado: {res['lag']} amostras")
    print(f"Lag módulo UPS: {res['lag_mod_ups']}")
    print(f"Pico da correlação (valor absoluto): {res['corr_peak_abs']:.6f}")
    print(f"Pico da correlação normalizado: {res['corr_peak_norm']:.6f}")
    print(res['alinhamento_txt'])
    print(f"Melhor offset por correlação complexa: {res['best_off']}")
    print(f"Melhor offset por energia: {res['best_off_energy']}")
    print(f"Métrica complexa no offset ótimo: {res['best_metric_complex']:.6f}")
    print(f"Ajuste linear I: a={res['fit_aI']:.6f}, b={res['fit_bI']:.6f}")
    print(f"Ajuste linear Q: a={res['fit_aQ']:.6f}, b={res['fit_bQ']:.6f}")
    print(f"Variância ruído I: {res['noise_var_I_sym']:.8f} V²")
    print(f"Variância ruído Q: {res['noise_var_Q_sym']:.8f} V²")
    print(f"Variância ruído magnitude: {res['noise_var_mag_sym']:.8f} V²")
    print(f"Desvio-padrão ruído final magnitude: {res['noise_std_mag_sym']:.8f} V")
    print(f"EVM RMS: {res['evm_rms']:.8f} V")
    print(f"EVM (%): {res['evm_pct']:.4f}")
    print(f"SNR estimado: {res['snr_est_db']:.4f} dB")


def plot_run(res):
    Ls = res["Ls"]
    ns = np.arange(Ls)

    show_sym = min(Ls, 400)

    plt.figure(figsize=(8, 4), dpi=120)
    plt.plot(res["lags"], res["corr"], label="Corr")
    plt.axvline(res["lag"], linestyle="--", label=f"Lag={res['lag']}")
    plt.title("Correlação global vs lag")
    plt.xlabel("Lag")
    plt.ylabel("Corr")
    plt.legend()
    plt.grid()

    plt.figure(figsize=(8, 4), dpi=120)
    plt.plot(res["offs_corr"], res["corr_vs_offset"], 'o-')
    plt.axvline(res["best_off"], linestyle="--", label=f"Off={res['best_off']}")
    plt.title("Correlação complexa vs offset simbólico")
    plt.xlabel("Offset")
    plt.ylabel("Métrica de correlação complexa")
    plt.xticks(np.arange(UPS))
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.grid()

    plt.figure(figsize=(10, 4), dpi=120)
    plt.plot(ns[:show_sym], res["txI_sym_aligned"][:show_sym], 'o-', label="TX I real")
    plt.plot(ns[:show_sym], res["rxI_sym_fit"][:show_sym], 'o-', label="RX I ajustado")
    plt.title("Símbolos I finais")
    plt.xlabel("Símbolo")
    plt.ylabel("V")
    plt.legend()
    plt.grid()

    plt.figure(figsize=(10, 4), dpi=120)
    plt.plot(ns[:show_sym], res["txQ_sym_aligned"][:show_sym], 'o-', label="TX Q real")
    plt.plot(ns[:show_sym], res["rxQ_sym_fit"][:show_sym], 'o-', label="RX Q ajustado")
    plt.title("Símbolos Q finais")
    plt.xlabel("Símbolo")
    plt.ylabel("V")
    plt.legend()
    plt.grid()

    plt.figure(figsize=(6, 6), dpi=120)
    plt.scatter(res["tx_symbols_I"], res["tx_symbols_Q"], s=3, alpha=0.6)
    plt.title("Constelação TX reconstruída")
    plt.xlabel("I TX")
    plt.ylabel("Q TX")
    plt.grid()
    plt.axis("equal")

    plt.figure(figsize=(6, 6), dpi=120)
    plt.scatter(res["rxI_sym_fit"], res["rxQ_sym_fit"], s=3, alpha=0.6)
    plt.title("Constelação RX final recebida")
    plt.xlabel("I RX")
    plt.ylabel("Q RX")
    plt.grid()
    plt.axis("equal")

    plt.figure(figsize=(6, 6), dpi=120)
    plt.hist2d(res["rxI_sym_fit"], res["rxQ_sym_fit"], bins=100)
    plt.title("Densidade da constelação RX")
    plt.xlabel("I RX ajustado")
    plt.ylabel("Q RX ajustado")
    plt.grid()
    plt.axis("equal")
    plt.colorbar(label="Contagens")


def save_all_figures(prefix=FIG_PREFIX):
    fig_nums = plt.get_fignums()
    paths = []
    for i, num in enumerate(fig_nums, start=1):
        fig = plt.figure(num)
        path = f"{prefix}_{i:02d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        paths.append(path)
    return paths


def append_results_log(res, tx_info=None, rx_info=None, csv_path=RESULTS_CSV):
    tx_info = tx_info or {}
    rx_info = rx_info or {}

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tx_port": TX_PORT,
        "rx_port": RX_PORT,
        "baud": BAUD,
        "ups_cfg": UPS,
        "skip_symbols_cfg": SKIP_SYMBOLS,
        "tx_total_symbols_cfg": TX_TOTAL_SYMBOLS,
        "rx_capture_points_cfg": RX_CAPTURE_POINTS,
        "n_symbols_analysis_cfg": N_SYMBOLS_ANALYSIS,
        "tx_symbol_rate": tx_info.get("SYMBOL_RATE", ""),
        "tx_sample_rate": tx_info.get("SAMPLE_RATE", ""),
        "tx_sample_period_us": tx_info.get("SAMPLE_PERIOD_US", ""),
        "tx_fixed_seed": tx_info.get("FIXED_SEED", ""),
        "tx_reported_n_symbols": tx_info.get("N_SYMBOLS", ""),
        "rx_skip_symbols_reported": rx_info.get("RX_SKIP_SYMBOLS", ""),
        "rx_skip_points_reported": rx_info.get("RX_SKIP_POINTS", ""),
        "rx_capture_points_reported": rx_info.get("RX_CAPTURE_POINTS", ""),
        "lag": res["lag"],
        "lag_mod_ups": res["lag_mod_ups"],
        "corr_peak_abs": res["corr_peak_abs"],
        "corr_peak_norm": res["corr_peak_norm"],
        "best_off": res["best_off"],
        "best_off_energy": res["best_off_energy"],
        "best_metric_complex": res["best_metric_complex"],
        "fit_aI": res["fit_aI"],
        "fit_bI": res["fit_bI"],
        "fit_aQ": res["fit_aQ"],
        "fit_bQ": res["fit_bQ"],
        "noise_mean_I_sym": res["noise_mean_I_sym"],
        "noise_mean_Q_sym": res["noise_mean_Q_sym"],
        "noise_var_I_sym": res["noise_var_I_sym"],
        "noise_var_Q_sym": res["noise_var_Q_sym"],
        "noise_var_mag_sym": res["noise_var_mag_sym"],
        "noise_std_I_sym": res["noise_std_I_sym"],
        "noise_std_Q_sym": res["noise_std_Q_sym"],
        "noise_std_mag_sym": res["noise_std_mag_sym"],
        "evm_rms": res["evm_rms"],
        "evm_pct": res["evm_pct"],
        "snr_est_db": res["snr_est_db"],
        "proc_label": res["proc_label"],
        "L": res["L"],
        "Ls": res["Ls"],
        "acq_time_s": res.get("acq_time_s", ""),
        "real_sample_rate": res.get("real_sample_rate", ""),
        "real_symbol_rate": res.get("real_symbol_rate", ""),
    }

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return csv_path


def save_summary_txt(res, tx_info=None, rx_info=None, prefix=SUMMARY_TXT_PREFIX):
    tx_info = tx_info or {}
    rx_info = rx_info or {}
    path = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("RESULTADO SINGLE RUN\n")
        f.write("===================\n")
        f.write(f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"TX_PORT={TX_PORT}\n")
        f.write(f"RX_PORT={RX_PORT}\n")
        f.write(f"BAUD={BAUD}\n")
        f.write(f"UPS={UPS}\n")
        f.write(f"SKIP_SYMBOLS={SKIP_SYMBOLS}\n")
        f.write(f"TX_TOTAL_SYMBOLS={TX_TOTAL_SYMBOLS}\n")
        f.write(f"RX_CAPTURE_POINTS={RX_CAPTURE_POINTS}\n")
        f.write(f"N_SYMBOLS_ANALYSIS={N_SYMBOLS_ANALYSIS}\n")
        f.write(f"acq_time_s={res.get('acq_time_s', np.nan):.10f}\n")
        f.write(f"real_sample_rate={res.get('real_sample_rate', np.nan):.10f}\n")
        f.write(f"real_symbol_rate={res.get('real_symbol_rate', np.nan):.10f}\n")
        for k, v in tx_info.items():
            f.write(f"TX_{k}={v}\n")
        for k, v in rx_info.items():
            f.write(f"RX_{k}={v}\n")
        f.write(f"lag={res['lag']}\n")
        f.write(f"lag_mod_ups={res['lag_mod_ups']}\n")
        f.write(f"corr_peak_abs={res['corr_peak_abs']:.10f}\n")
        f.write(f"corr_peak_norm={res['corr_peak_norm']:.10f}\n")
        f.write(f"best_off={res['best_off']}\n")
        f.write(f"best_off_energy={res['best_off_energy']}\n")
        f.write(f"best_metric_complex={res['best_metric_complex']:.10f}\n")
        f.write(f"fit_aI={res['fit_aI']:.10f}\n")
        f.write(f"fit_bI={res['fit_bI']:.10f}\n")
        f.write(f"fit_aQ={res['fit_aQ']:.10f}\n")
        f.write(f"fit_bQ={res['fit_bQ']:.10f}\n")
        f.write(f"noise_mean_I_sym={res['noise_mean_I_sym']:.10f}\n")
        f.write(f"noise_mean_Q_sym={res['noise_mean_Q_sym']:.10f}\n")
        f.write(f"noise_var_I_sym={res['noise_var_I_sym']:.10f}\n")
        f.write(f"noise_var_Q_sym={res['noise_var_Q_sym']:.10f}\n")
        f.write(f"noise_var_mag_sym={res['noise_var_mag_sym']:.10f}\n")
        f.write(f"noise_std_I_sym={res['noise_std_I_sym']:.10f}\n")
        f.write(f"noise_std_Q_sym={res['noise_std_Q_sym']:.10f}\n")
        f.write(f"noise_std_mag_sym={res['noise_std_mag_sym']:.10f}\n")
        f.write(f"evm_rms={res['evm_rms']:.10f}\n")
        f.write(f"evm_pct={res['evm_pct']:.10f}\n")
        f.write(f"snr_est_db={res['snr_est_db']:.10f}\n")
    return path


def main():
    print("A guardar ficheiros em:", OUTPUT_DIR)
    print("A abrir portas...")
    tx = serial.Serial(TX_PORT, BAUD, timeout=0.1)
    rx = serial.Serial(RX_PORT, BAUD, timeout=0.1)

    time.sleep(STARTUP_WAIT)

    tx.reset_input_buffer()
    rx.reset_input_buffer()
    tx.reset_output_buffer()
    rx.reset_output_buffer()

    print("Mensagens iniciais TX:", read_lines(tx, 0.5))
    print("Mensagens iniciais RX:", read_lines(rx, 0.5))

    tx_info = {}
    rx_info = {}

    try:
        t0_acq = time.time()
        rx_d = acquire_once(tx, rx)
        t1_acq = time.time()

        acq_time_s = t1_acq - t0_acq
        real_sample_rate = RX_CAPTURE_POINTS / acq_time_s
        real_symbol_rate = real_sample_rate / UPS

        res = analyse_once(rx_d)
        res["acq_time_s"] = float(acq_time_s)
        res["real_sample_rate"] = float(real_sample_rate)
        res["real_symbol_rate"] = float(real_symbol_rate)

        print("\n==== TAXA REAL MEDIDA ====")
        print(f"Tempo de aquisição: {acq_time_s:.6f} s")
        print(f"Sample rate real: {real_sample_rate:.2f} Hz")
        print(f"Symbol rate real: {real_symbol_rate:.2f} sym/s")

        print_result(res)



        

        tx.write(b'i')
        tx.flush()
        time.sleep(0.2)
        tx_info_lines = read_lines(tx, 0.5)
        
        tx_info = parse_keyvals(tx_info_lines)

        tx_rate_auto = {}
        for line in tx_info_lines:
            parsed_rate = parse_tx_rate_line(line)
            if parsed_rate is not None:
                tx_rate_auto = parsed_rate

        if tx_rate_auto:
            for k, v in tx_rate_auto.items():
                tx_info[f"AUTO_{k}"] = v

        print("\nInfo TX:", tx_info_lines)
        print("TX_RATE automático:", tx_rate_auto)    

    finally:
        tx.close()
        rx.close()

    csv_path = append_results_log(res, tx_info=tx_info, rx_info=rx_info)
    txt_path = save_summary_txt(res, tx_info=tx_info, rx_info=rx_info)
    print(f"\nLog CSV atualizado: {csv_path}")
    print(f"Resumo TXT gravado: {txt_path}")

    plot_run(res)

    if SAVE_FIGURES:
        saved = save_all_figures(FIG_PREFIX)
        print("\nFiguras gravadas:")
        for p in saved:
            print("  ", p)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
