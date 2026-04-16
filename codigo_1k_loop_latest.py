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

TX_PORT = "COM7"
RX_PORT = "COM5"
BAUD = 115200

UPS = 8
N_SYMBOLS = 1000
FRAME_N = N_SYMBOLS * UPS
RX_EXPECTED_N = 30000

VREF = 3.3
ADC_MAX = 4095.0
V_MID = VREF / 2.0

STARTUP_WAIT = 3.0
TIMEOUT_CMD = 20.0
TIMEOUT_DUMP = 60.0
PRINT_EVERY = 0.5

SHOW_MAX_SAMPLES_X = 30000
PEAK_MIN_DISTANCE = FRAME_N // 2
PEAK_REL_THRESHOLD = 0.60

USE_MATCHED_FILTER = True
REMOVE_DC = True

ALPHA = 0.4
NTAPS = 10 * UPS + 1
GROUP_DELAY = (NTAPS - 1) // 2

SAVE_FIGURES = True
FIG_PREFIX = "ensaio_offset_joint_estimation"

ZOOM_SAMPLES_1 = (900, 1040)
ZOOM_SAMPLES_2 = (1400, 1560)
ZOOM_SAMPLES_3 = (2200, 2360)
ZOOM_SYMBOLS = (40, 90)


# ========================================================
# AUX
# ========================================================

def parse_int_pair(line):
    try:
        a, b = line.split(",")
        return int(a), int(b)
    except Exception:
        return None


def parse_float_pair(line):
    try:
        a, b = line.split(",")
        return float(a), float(b)
    except Exception:
        return None


def read_lines(ser, duration=0.3):
    t0 = time.time()
    lines = []
    while time.time() - t0 < duration:
        if ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                lines.append(line)
    return lines


def wait_for_text(ser, target, timeout):
    t0 = time.time()
    seen = []
    while time.time() - t0 < timeout:
        if ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                seen.append(line)
                print(f"[{ser.port}] {line}")
                if target in line:
                    return True, seen
    return False, seen


def capture_dump(ser, name, expected_n=None):
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

        if time.time() - t0 > TIMEOUT_DUMP:
            raise TimeoutError(f"Timeout à espera do dump de {name}.")

    return data


def capture_symbol_dump(ser, expected_n=None):
    data = []
    started = False
    t0 = time.time()
    t_last = time.time()

    while True:
        if ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            if line == "SYM_BEGIN":
                started = True
                print("TX SYMBOLS BEGIN")
                continue

            if line == "SYM_END":
                print("TX SYMBOLS END")
                break

            if started:
                p = parse_float_pair(line)
                if p is not None:
                    data.append(p)

        if time.time() - t_last > PRINT_EVERY:
            if started:
                if expected_n is None:
                    print(f"TX symbols dump: {len(data)}")
                else:
                    print(f"TX symbols dump: {len(data)}/{expected_n}")
            t_last = time.time()

        if time.time() - t0 > TIMEOUT_DUMP:
            raise TimeoutError("Timeout à espera do dump de símbolos do TX.")

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


def normalize_to_0_vref(*arrays):
    valid = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        if a.size > 0:
            valid.append(a)

    if len(valid) == 0:
        return [np.asarray(arr, dtype=float).copy() for arr in arrays]

    global_min = min(float(np.min(a)) for a in valid)
    global_max = max(float(np.max(a)) for a in valid)

    if global_max - global_min < 1e-12:
        mid = np.full_like(valid[0], VREF / 2.0, dtype=float)
        out = []
        for arr in arrays:
            a = np.asarray(arr, dtype=float)
            out.append(np.full_like(a, VREF / 2.0, dtype=float))
        return out

    out = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        out.append((a - global_min) * VREF / (global_max - global_min))
    return out


def remove_dc_volts(xv):
    return xv - V_MID if REMOVE_DC else xv.copy()


def sliding_normalized_correlation_complex(rxI, rxQ, txI, txQ):
    tx = np.asarray(txI, dtype=float) + 1j * np.asarray(txQ, dtype=float)
    rx = np.asarray(rxI, dtype=float) + 1j * np.asarray(rxQ, dtype=float)

    M = len(tx)
    N = len(rx)
    if N < M:
        raise ValueError("rx tem de ter comprimento >= tx")

    tx0 = tx - np.mean(tx)
    tx_energy = np.linalg.norm(tx0)
    out = np.zeros(N - M + 1, dtype=float)

    if tx_energy < 1e-12:
        return out

    for k in range(N - M + 1):
        w = rx[k:k + M]
        w0 = w - np.mean(w)
        den = np.linalg.norm(w0) * tx_energy
        out[k] = abs(np.vdot(tx0, w0)) / den if den > 1e-12 else 0.0

    return out


def find_peaks_simple(y, min_distance, threshold):
    y = np.asarray(y)
    if len(y) < 3:
        return np.array([], dtype=int)

    candidates = []
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] >= threshold:
            candidates.append(i)

    candidates = sorted(candidates, key=lambda i: y[i], reverse=True)

    chosen = []
    for idx in candidates:
        if all(abs(idx - c) >= min_distance for c in chosen):
            chosen.append(idx)

    chosen.sort()
    return np.array(chosen, dtype=int)


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


def complex_global_corr_and_lag(rxI, rxQ, refI, refQ):
    rx = np.asarray(rxI, dtype=float) + 1j * np.asarray(rxQ, dtype=float)
    ref = np.asarray(refI, dtype=float) + 1j * np.asarray(refQ, dtype=float)

    rx0 = rx - np.mean(rx)
    ref0 = ref - np.mean(ref)

    corr = correlate(rx0, ref0, mode="full")
    lags = correlation_lags(len(rx), len(ref), mode="full")
    idx_peak = int(np.argmax(np.abs(corr)))
    lag = int(lags[idx_peak])

    corr_peak_abs = float(np.abs(corr[idx_peak]))
    den = np.linalg.norm(rx0) * np.linalg.norm(ref0)
    corr_peak_norm = float(corr_peak_abs / den) if den > 1e-12 else 0.0

    return corr, lags, lag, corr_peak_abs, corr_peak_norm


def apply_lag_alignment(refI, refQ, rxI, rxQ, lag):
    refI2 = np.asarray(refI, dtype=float).copy()
    refQ2 = np.asarray(refQ, dtype=float).copy()
    rxI2 = np.asarray(rxI, dtype=float).copy()
    rxQ2 = np.asarray(rxQ, dtype=float).copy()

    if lag > 0:
        rxI2 = rxI2[lag:]
        rxQ2 = rxQ2[lag:]
        refI2 = refI2[:len(rxI2)]
        refQ2 = refQ2[:len(rxQ2)]
        txt = f"RX atrasado {lag} amostras -> cortado início do RX"
    elif lag < 0:
        refI2 = refI2[-lag:]
        refQ2 = refQ2[-lag:]
        rxI2 = rxI2[:len(refI2)]
        rxQ2 = rxQ2[:len(refQ2)]
        txt = f"TX(ref) atrasado {-lag} amostras -> cortado início da referência"
    else:
        txt = "Lag = 0 -> não foi necessário cortar início"

    L = min(len(refI2), len(refQ2), len(rxI2), len(rxQ2))
    refI2 = refI2[:L]
    refQ2 = refQ2[:L]
    rxI2 = rxI2[:L]
    rxQ2 = rxQ2[:L]

    return refI2, refQ2, rxI2, rxQ2, txt


def estimate_offset_and_symbol_start_complex(tx_sym_I, tx_sym_Q, rx_long_I, rx_long_Q, ups):
    tx = np.asarray(tx_sym_I, dtype=float) + 1j * np.asarray(tx_sym_Q, dtype=float)

    best_off = 0
    best_sym_start = 0
    best_metric = -1.0
    offs = []
    metrics = []
    sym_starts = []

    for off in range(ups):
        rx_dec = np.asarray(rx_long_I[off::ups], dtype=float) + 1j * np.asarray(rx_long_Q[off::ups], dtype=float)
        if len(rx_dec) < len(tx):
            offs.append(off)
            metrics.append(0.0)
            sym_starts.append(0)
            continue

        corr = sliding_normalized_correlation_complex(rx_dec.real, rx_dec.imag, tx.real, tx.imag)
        idx = int(np.argmax(corr))
        metric = float(corr[idx])

        offs.append(off)
        metrics.append(metric)
        sym_starts.append(idx)

        if metric > best_metric:
            best_metric = metric
            best_off = off
            best_sym_start = idx

    metrics = np.array(metrics, dtype=float)
    if len(metrics) > 0 and np.max(metrics) > 1e-12:
        metrics = metrics / np.max(metrics)

    best_sample_start = int(best_off + best_sym_start * ups)
    return int(best_off), int(best_sym_start), best_sample_start, np.array(offs, dtype=int), np.array(sym_starts, dtype=int), metrics


def acquire_once(tx, rx):
    print("A reiniciar frame TX e captura RX...")

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
    rx_d = capture_dump(rx, 'RX', expected_n=RX_EXPECTED_N)

    print("A pedir dump ao TX...")
    tx.write(b'd')
    tx.flush()
    tx_d = capture_dump(tx, 'TX', expected_n=FRAME_N)

    print("A pedir símbolos do TX...")
    tx.write(b's')
    tx.flush()
    tx_s = capture_symbol_dump(tx, expected_n=N_SYMBOLS)

    if len(tx_d) == 0 or len(rx_d) == 0 or len(tx_s) == 0:
        raise RuntimeError("Não chegaram dados suficientes.")

    return np.array(tx_d, dtype=float), np.array(rx_d, dtype=float), np.array(tx_s, dtype=float)


def analyse_once(tx_d, rx_d, tx_s):
    rrc = make_rrc(ALPHA, UPS, NTAPS)

    dacI_raw_v = adc_to_volts(tx_d[:, 0])
    dacQ_raw_v = adc_to_volts(tx_d[:, 1])
    adcI_raw_v = adc_to_volts(rx_d[:, 0])
    adcQ_raw_v = adc_to_volts(rx_d[:, 1])

    tx_frame_I_raw = dacI_raw_v[:FRAME_N]
    tx_frame_Q_raw = dacQ_raw_v[:FRAME_N]
    rx_long_I_raw = adcI_raw_v
    rx_long_Q_raw = adcQ_raw_v

    tx_frame_I_bb = remove_dc_volts(tx_frame_I_raw)
    tx_frame_Q_bb = remove_dc_volts(tx_frame_Q_raw)
    rx_long_I_bb = remove_dc_volts(rx_long_I_raw)
    rx_long_Q_bb = remove_dc_volts(rx_long_Q_raw)

    tx_symbols_I = np.asarray(tx_s[:, 0], dtype=float)
    tx_symbols_Q = np.asarray(tx_s[:, 1], dtype=float)

    if USE_MATCHED_FILTER:
        tx_frame_I_ref = apply_fir_same(tx_frame_I_bb, rrc)
        tx_frame_Q_ref = apply_fir_same(tx_frame_Q_bb, rrc)
        rx_long_I_proc = apply_fir_same(rx_long_I_bb, rrc)
        rx_long_Q_proc = apply_fir_same(rx_long_Q_bb, rrc)
        proc_label = "baseband + matched filter"
    else:
        tx_frame_I_ref = tx_frame_I_bb.copy()
        tx_frame_Q_ref = tx_frame_Q_bb.copy()
        rx_long_I_proc = rx_long_I_bb.copy()
        rx_long_Q_proc = rx_long_Q_bb.copy()
        proc_label = "baseband sem matched filter"

    best_off, best_sym_start, best_sample_start, offs_corr, sym_starts, corr_vs_offset = estimate_offset_and_symbol_start_complex(
        tx_symbols_I, tx_symbols_Q, rx_long_I_proc, rx_long_Q_proc, UPS
    )

    corr_time_c = sliding_normalized_correlation_complex(
        rx_long_I_proc, rx_long_Q_proc, tx_frame_I_ref, tx_frame_Q_ref
    )
    peaks_c = find_peaks_simple(
        corr_time_c,
        PEAK_MIN_DISTANCE,
        PEAK_REL_THRESHOLD * (np.max(corr_time_c) if len(corr_time_c) else 0.0)
    )

    rxI_with_offset = rx_long_I_proc[best_off:]
    rxQ_with_offset = rx_long_Q_proc[best_off:]
    frame_start_after_offset = best_sym_start * UPS

    rxI_unaligned = rxI_with_offset[frame_start_after_offset:frame_start_after_offset + len(tx_frame_I_ref)]
    rxQ_unaligned = rxQ_with_offset[frame_start_after_offset:frame_start_after_offset + len(tx_frame_Q_ref)]

    refI_unaligned = tx_frame_I_ref[:len(rxI_unaligned)]
    refQ_unaligned = tx_frame_Q_ref[:len(rxQ_unaligned)]

    Ld = min(len(refI_unaligned), len(refQ_unaligned), len(rxI_unaligned), len(rxQ_unaligned))
    refI_unaligned = refI_unaligned[:Ld]
    refQ_unaligned = refQ_unaligned[:Ld]
    rxI_unaligned = rxI_unaligned[:Ld]
    rxQ_unaligned = rxQ_unaligned[:Ld]

    corr_global_c, lags, lag, corr_peak_abs, corr_peak_norm = complex_global_corr_and_lag(
        rxI_unaligned, rxQ_unaligned, refI_unaligned, refQ_unaligned
    )

    refI, refQ, rxI, rxQ, alinhamento_txt = apply_lag_alignment(
        refI_unaligned, refQ_unaligned, rxI_unaligned, rxQ_unaligned, lag
    )

    rxI_sym = rxI[::UPS][:N_SYMBOLS]
    rxQ_sym = rxQ[::UPS][:N_SYMBOLS]

    Ls = min(len(tx_symbols_I), len(tx_symbols_Q), len(rxI_sym), len(rxQ_sym))
    txI_sym_aligned = tx_symbols_I[:Ls]
    txQ_sym_aligned = tx_symbols_Q[:Ls]
    rxI_sym_aligned = rxI_sym[:Ls]
    rxQ_sym_aligned = rxQ_sym[:Ls]

    refI_unaligned, rxI_unaligned = normalize_to_0_vref(refI_unaligned, rxI_unaligned)
    refQ_unaligned, rxQ_unaligned = normalize_to_0_vref(refQ_unaligned, rxQ_unaligned)
    refI, rxI = normalize_to_0_vref(refI, rxI)
    refQ, rxQ = normalize_to_0_vref(refQ, rxQ)
    txI_sym_aligned, rxI_sym_aligned = normalize_to_0_vref(txI_sym_aligned, rxI_sym_aligned)
    txQ_sym_aligned, rxQ_sym_aligned = normalize_to_0_vref(txQ_sym_aligned, rxQ_sym_aligned)

    rxI_sym_fit, fit_aI, fit_bI = affine_fit(txI_sym_aligned, rxI_sym_aligned)
    rxQ_sym_fit, fit_aQ, fit_bQ = affine_fit(txQ_sym_aligned, rxQ_sym_aligned)

    errI = rxI - refI
    errQ = rxQ - refQ
    errI_sym = rxI_sym_fit - txI_sym_aligned
    errQ_sym = rxQ_sym_fit - txQ_sym_aligned
    noise_sym_mag = np.sqrt(errI_sym ** 2 + errQ_sym ** 2)

    return {
        "proc_label": proc_label,
        "lags": lags,
        "corr": np.abs(corr_global_c),
        "lag": lag,
        "corr_peak_abs": corr_peak_abs,
        "corr_peak_norm": corr_peak_norm,
        "lag_mod_ups": lag % UPS if UPS > 0 else 0,
        "alinhamento_txt": alinhamento_txt,
        "frame_start": int(best_sample_start),
        "frame_start_after_offset": int(frame_start_after_offset),
        "best_sym_start": int(best_sym_start),
        "L": len(refI),
        "offs_corr": offs_corr,
        "sym_starts": sym_starts,
        "corr_vs_offset": corr_vs_offset,
        "best_off": best_off,
        "best_metric_complex": float(corr_vs_offset[best_off]) if len(corr_vs_offset) > best_off else 0.0,
        "fit_aI": fit_aI,
        "fit_bI": fit_bI,
        "fit_aQ": fit_aQ,
        "fit_bQ": fit_bQ,
        "refI_unaligned": refI_unaligned,
        "refQ_unaligned": refQ_unaligned,
        "rxI_unaligned": rxI_unaligned,
        "rxQ_unaligned": rxQ_unaligned,
        "refI": refI,
        "refQ": refQ,
        "rxI": rxI,
        "rxQ": rxQ,
        "tx_symbols_I": tx_symbols_I,
        "tx_symbols_Q": tx_symbols_Q,
        "txI_sym_aligned": txI_sym_aligned,
        "txQ_sym_aligned": txQ_sym_aligned,
        "rxI_sym_aligned": rxI_sym_aligned,
        "rxQ_sym_aligned": rxQ_sym_aligned,
        "rxI_sym_fit": rxI_sym_fit,
        "rxQ_sym_fit": rxQ_sym_fit,
        "Ls": Ls,
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
        "corr_time_c_abs": corr_time_c,
        "x_time": np.arange(len(corr_time_c)),
        "peaks_c": peaks_c,
    }


def print_result(res):
    print("\n=========================")
    print("RESULTADO SINGLE RUN")
    print("=========================")
    print(f"Modo de processamento: {res['proc_label']}")
    print(f"Offset escolhido: {res['best_off']}")
    print(f"Início simbólico escolhido: {res['best_sym_start']}")
    print(f"Frame start total no RX: {res['frame_start']}")
    print(f"Lag fino depois do offset: {res['lag']} amostras")
    print(f"Lag módulo UPS: {res['lag_mod_ups']}")
    print(f"Pico da correlação (valor absoluto): {res['corr_peak_abs']:.6f}")
    print(f"Pico da correlação normalizado: {res['corr_peak_norm']:.6f}")
    print(res['alinhamento_txt'])
    print(f"Métrica complexa no offset ótimo: {res['best_metric_complex']:.6f}")
    print(f"Ajuste linear I: a={res['fit_aI']:.6f}, b={res['fit_bI']:.6f}")
    print(f"Ajuste linear Q: a={res['fit_aQ']:.6f}, b={res['fit_bQ']:.6f}")
    print(f"Desvio-padrão ruído final magnitude: {res['noise_std_mag_sym']:.8f} V")


def plot_run(res):
    L = res["L"]
    Ls = res["Ls"]
    ns = np.arange(Ls)
    n = np.arange(L)

    plt.figure(figsize=(8, 4), dpi=120)
    plt.plot(res["lags"], res["corr"])
    plt.axvline(res["lag"], linestyle="--", label=f"Lag={res['lag']}")
    plt.xlabel("Lag", fontsize=18)
    plt.ylabel("Corr", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend()
    plt.grid()

    plt.figure(figsize=(12, 4), dpi=120)
    x_shift = res["x_time"] - res["frame_start"]
    plt.plot(x_shift, res["corr_time_c_abs"])
    y0, y1 = plt.ylim()
    y_arrow = y0 + 0.12 * (y1 - y0)
    y_text = y0 + 0.28 * (y1 - y0)
    plt.annotate(
        "capture start",
        xy=(0, y_arrow),
        xytext=(0, y_text),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", lw=1.5),
        fontsize=16,
        ha="center",
        va="bottom",
    )
    plt.xlabel("Sample", fontsize=18)
    plt.ylabel("Correlation", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(min(x_shift), max(x_shift))
    plt.grid()


    plt.figure(figsize=(8, 4), dpi=120)
    plt.plot(res["offs_corr"], res["corr_vs_offset"], 'o-')
    y0, y1 = plt.ylim()
    y_arrow = res["corr_vs_offset"][res["best_off"]]
    y_text = y0 + 0.28 * (y1 - y0)
    plt.annotate(
        "best",
        xy=(res["best_off"], y_arrow),
        xytext=(res["best_off"], y_text),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", lw=1.5),
        fontsize=16,
        ha="center",
        va="bottom",
    )
    plt.xlabel("Offset", fontsize=18)
    plt.ylabel("Correlation", fontsize=18)
    plt.xticks(np.arange(UPS), fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(-0.25, UPS - 0.75)
    plt.ylim(0.0, 1.05)
    plt.grid()

    plt.figure(figsize=(6, 6), dpi=120)
    plt.scatter(res["tx_symbols_I"], res["tx_symbols_Q"], s=3, alpha=0.6)
    plt.xlabel("I TX", fontsize=18)
    plt.ylabel("Q TX", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    plt.axis("equal")

    plt.figure(figsize=(6, 6), dpi=120)
    plt.scatter(res["rxI_sym_fit"], res["rxQ_sym_fit"], s=3, alpha=0.6)
    plt.xlabel("I RX", fontsize=18)
    plt.ylabel("Q RX", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    plt.axis("equal")

    x0, x1 = ZOOM_SAMPLES_1
    x = np.arange(len(res["refI_unaligned"]))
    if len(x) > 0 and x1 > x0:
        plt.figure(figsize=(10, 4), dpi=120)
        plt.plot(x, res["refI_unaligned"], 'o-', label="TX I")
        plt.plot(x, res["rxI_unaligned"], 'o-', label="RX I")
        plt.xlabel("I Samples", fontsize=18)
        plt.ylabel("Voltage (V)", fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(x0, min(x1, len(x) - 1))
        plt.ylim(0, VREF)
        plt.legend()
        plt.grid()

    x0, x1 = ZOOM_SAMPLES_1
    x = np.arange(len(res["refQ_unaligned"]))
    if len(x) > 0 and x1 > x0:
        plt.figure(figsize=(10, 4), dpi=120)
        plt.plot(x, res["refQ_unaligned"], 'o-', label="TX Q")
        plt.plot(x, res["rxQ_unaligned"], 'o-', label="RX Q")
        plt.xlabel("Q Samples", fontsize=18)
        plt.ylabel("Voltage (V)", fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(x0, min(x1, len(x) - 1))
        plt.ylim(0, VREF)
        plt.legend()
        plt.grid()

    x0, x1 = ZOOM_SAMPLES_1
    y_ref = np.asarray(res["refI"])
    y_rx = np.asarray(res["rxI"])
    x_ref = np.arange(len(y_ref)) + int(res["best_off"])
    x_rx = np.arange(len(y_rx))
    if (len(x_ref) > 0 or len(x_rx) > 0) and x1 > x0:
        plt.figure(figsize=(10, 4), dpi=120)
        plt.plot(x_ref, y_ref, 'o-', label="TX I")
        plt.plot(x_rx, y_rx, 'o-', label="RX I")
        plt.xlabel("I Samples", fontsize=18)
        plt.ylabel("Voltage (V)", fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(x0, x1)
        plt.ylim(0, VREF)
        plt.legend()
        plt.grid()

    x0, x1 = ZOOM_SAMPLES_1
    y_ref = np.asarray(res["refQ"])
    y_rx = np.asarray(res["rxQ"])
    x_ref = np.arange(len(y_ref)) + int(res["best_off"])
    x_rx = np.arange(len(y_rx))
    if (len(x_ref) > 0 or len(x_rx) > 0) and x1 > x0:
        plt.figure(figsize=(10, 4), dpi=120)
        plt.plot(x_ref, y_ref, 'o-', label="TX Q")
        plt.plot(x_rx, y_rx, 'o-', label="RX Q")
        plt.xlabel("Q Samples", fontsize=18)
        plt.ylabel("Voltage (V)", fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(x0, x1)
        plt.ylim(0, VREF)
        plt.legend()
        plt.grid()

    x0, x1 = ZOOM_SAMPLES_1
    x = np.arange(len(res["refI"]))
    if len(x) > 0 and x1 > x0:
        plt.figure(figsize=(10, 4), dpi=120)
        plt.plot(x, res["refI"], 'o-', label="TX I")
        plt.plot(x, res["rxI"], 'o-', label="RX I")
        plt.xlabel("I Samples", fontsize=18)
        plt.ylabel("Voltage (V)", fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(x0, min(x1, len(x) - 1))
        plt.ylim(0, VREF)
        plt.legend()
        plt.grid()

    x0, x1 = ZOOM_SAMPLES_1
    x = np.arange(len(res["refQ"]))
    if len(x) > 0 and x1 > x0:
        plt.figure(figsize=(10, 4), dpi=120)
        plt.plot(x, res["refQ"], 'o-', label="TX Q")
        plt.plot(x, res["rxQ"], 'o-', label="RX Q")
        plt.xlabel("Q Samples", fontsize=18)
        plt.ylabel("Voltage (V)", fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(x0, min(x1, len(x) - 1))
        plt.ylim(0, VREF)
        plt.legend()
        plt.grid()

    plt.figure(figsize=(10, 4), dpi=120)
    plt.plot(n, res["errI"], label="I")
    plt.plot(n, res["errQ"], label="Q")
    plt.xlabel("Sample", fontsize=18)
    plt.ylabel("Error (V)", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0, 8000)
    plt.grid()
    plt.legend()


def save_all_figures(prefix=FIG_PREFIX):
    fig_nums = plt.get_fignums()
    paths = []
    for i, num in enumerate(fig_nums, start=1):
        fig = plt.figure(num)
        path = f"{prefix}_{i:02d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        paths.append(path)
    return paths


def main():
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

    try:
        tx_d, rx_d, tx_s = acquire_once(tx, rx)
        res = analyse_once(tx_d, rx_d, tx_s)
        print_result(res)

        tx.write(b'i')
        tx.flush()
        time.sleep(0.2)
        print("\nInfo TX:", read_lines(tx, 0.5))

    finally:
        tx.close()
        rx.close()

    plot_run(res)

    if SAVE_FIGURES:
        saved = save_all_figures(FIG_PREFIX)
        print("\nFiguras gravadas:")
        for p in saved:
            print("  ", p)

    for num in plt.get_fignums():
        plt.figure(num)
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
