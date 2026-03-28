
import time
import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

tx_port = "COM7"
rx_port = "COM5"
baud = 115200

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

z0 = 400
z1 = 520
zs0 = 40
zs1 = 70

SHOW_MAX_SAMPLES_X = 30000
PEAK_MIN_DISTANCE = FRAME_N // 2
PEAK_REL_THRESHOLD = 0.60

USE_MATCHED_FILTER = True
REMOVE_DC = True

alpha = 0.4
Ntaps = 10 * UPS + 1
GROUP_DELAY = (Ntaps - 1) // 2

SWEEP = [
    {
        "rate_label": "config_atual",
        "rate_value": 1.0,
        "rate_unit": "configuração",
        "repeats": 3,
        "rate_cmd_tx": None,
        "rate_cmd_rx": None,
    }
]

SAVE_FIGURES = True
FIG_PREFIX = "ensaio_mf_symbols_fixed_v4"


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


def safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else np.nan


def safe_var(x):
    return float(np.var(x, ddof=0)) if len(x) > 0 else np.nan


def safe_std(x):
    return float(np.std(x, ddof=0)) if len(x) > 0 else np.nan


def set_rate_if_supported(tx, rx, cfg):
    if cfg.get("rate_cmd_rx") is not None:
        print(f"A enviar comando de taxa ao RX: {cfg['rate_cmd_rx']}")
        rx.write(cfg["rate_cmd_rx"])
        rx.flush()
        time.sleep(0.3)
        print("RX:", read_lines(rx, 0.5))

    if cfg.get("rate_cmd_tx") is not None:
        print(f"A enviar comando de taxa ao TX: {cfg['rate_cmd_tx']}")
        tx.write(cfg["rate_cmd_tx"])
        tx.flush()
        time.sleep(0.3)
        print("TX:", read_lines(tx, 0.5))


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


def sliding_normalized_correlation(rx, tx):
    tx = np.asarray(tx, dtype=float)
    rx = np.asarray(rx, dtype=float)

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
        out[k] = np.dot(w0, tx0) / den if den > 1e-12 else 0.0

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

        mag2 = xI[:L]**2 + xQ[:L]**2
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
    rrc = make_rrc(alpha, UPS, Ntaps)

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

    if len(tx_frame_I_ref) > GROUP_DELAY and len(rx_long_I_proc) > GROUP_DELAY:
        tx_frame_I_ref = tx_frame_I_ref[GROUP_DELAY:]
        tx_frame_Q_ref = tx_frame_Q_ref[GROUP_DELAY:]
        rx_long_I_proc = rx_long_I_proc[GROUP_DELAY:]
        rx_long_Q_proc = rx_long_Q_proc[GROUP_DELAY:]

    corr_time_I = sliding_normalized_correlation(rx_long_I_proc, tx_frame_I_ref)
    corr_time_I_abs = np.abs(corr_time_I)
    frame_start = int(np.argmax(corr_time_I_abs))

    rxI_frame = rx_long_I_proc[frame_start:frame_start + len(tx_frame_I_ref)]
    rxQ_frame = rx_long_Q_proc[frame_start:frame_start + len(tx_frame_Q_ref)]
    refI = tx_frame_I_ref[:len(rxI_frame)]
    refQ = tx_frame_Q_ref[:len(rxQ_frame)]

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

    rxI_sym = rxI_frame[best_off::UPS][:N_SYMBOLS]
    rxQ_sym = rxQ_frame[best_off::UPS][:N_SYMBOLS]

    Ls = min(len(tx_symbols_I), len(tx_symbols_Q), len(rxI_sym), len(rxQ_sym))
    txI_sym_aligned = tx_symbols_I[:Ls]
    txQ_sym_aligned = tx_symbols_Q[:Ls]
    rxI_sym_aligned = rxI_sym[:Ls]
    rxQ_sym_aligned = rxQ_sym[:Ls]

    rxI_sym_fit, fit_aI, fit_bI = affine_fit(txI_sym_aligned, rxI_sym_aligned)
    rxQ_sym_fit, fit_aQ, fit_bQ = affine_fit(txQ_sym_aligned, rxQ_sym_aligned)

    errI = rxI_frame - refI
    errQ = rxQ_frame - refQ
    errI_sym = rxI_sym_fit - txI_sym_aligned
    errQ_sym = rxQ_sym_fit - txQ_sym_aligned
    noise_sym_mag = np.sqrt(errI_sym**2 + errQ_sym**2)

    corr = correlate(rxI_frame - np.mean(rxI_frame), refI - np.mean(refI), mode="full")
    lags = correlation_lags(len(rxI_frame), len(refI), mode="full")
    idx_peak = np.argmax(np.abs(corr))
    lag = int(lags[idx_peak])
    corr_peak_abs = float(np.abs(corr[idx_peak]))
    den_global = np.linalg.norm(rxI_frame - np.mean(rxI_frame)) * np.linalg.norm(refI - np.mean(refI))
    corr_peak_norm = float(corr_peak_abs / den_global) if den_global > 1e-12 else 0.0

    corr_time_Q = sliding_normalized_correlation(rx_long_Q_proc, tx_frame_Q_ref)
    corr_time_Q_abs = np.abs(corr_time_Q)

    x_time = np.arange(len(corr_time_I_abs))
    thr_I = PEAK_REL_THRESHOLD * (np.max(corr_time_I_abs) if len(corr_time_I_abs) else 0.0)
    thr_Q = PEAK_REL_THRESHOLD * (np.max(corr_time_Q_abs) if len(corr_time_Q_abs) else 0.0)
    peaks_I = find_peaks_simple(corr_time_I_abs, PEAK_MIN_DISTANCE, thr_I)
    peaks_Q = find_peaks_simple(corr_time_Q_abs, PEAK_MIN_DISTANCE, thr_Q)
    ideal_pts = np.unique(np.round(np.column_stack([tx_symbols_I, tx_symbols_Q]), 6), axis=0)

    return {
        "proc_label": proc_label,
        "lags": lags,
        "corr": corr,
        "lag": lag,
        "corr_peak_abs": corr_peak_abs,
        "corr_peak_norm": corr_peak_norm,
        "lag_mod_ups": lag % UPS if UPS > 0 else 0,
        "alinhamento_txt": f"Frame RX encontrado em {frame_start}",
        "L": L,
        "offs_I": offs_corr,
        "mets_I": corr_vs_offset,
        "offs_Q": offs_corr,
        "mets_Q": corr_vs_offset,
        "mets_sum": corr_vs_offset,
        "best_off": best_off,
        "best_off_energy": best_off_energy,
        "best_metric_I": float(corr_vs_offset[best_off]) if len(corr_vs_offset) > best_off else 0.0,
        "best_metric_Q": float(corr_vs_offset[best_off]) if len(corr_vs_offset) > best_off else 0.0,
        "best_metric_sum": float(corr_vs_offset[best_off]) if len(corr_vs_offset) > best_off else 0.0,
        "off_energy": off_energy,
        "fit_aI": fit_aI,
        "fit_bI": fit_bI,
        "fit_aQ": fit_aQ,
        "fit_bQ": fit_bQ,
        "std_txI": float(np.std(txI_sym_aligned)) if len(txI_sym_aligned) else np.nan,
        "std_txQ": float(np.std(txQ_sym_aligned)) if len(txQ_sym_aligned) else np.nan,
        "std_rxI": float(np.std(rxI_sym_aligned)) if len(rxI_sym_aligned) else np.nan,
        "std_rxQ": float(np.std(rxQ_sym_aligned)) if len(rxQ_sym_aligned) else np.nan,
        "frame_start": frame_start,
        "tx_frame_I_raw": tx_frame_I_raw,
        "rx_long_I_raw": rx_long_I_raw,
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
        "ideal_pts": ideal_pts,
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
        "corr_time_I_abs": corr_time_I_abs,
        "corr_time_Q_abs": corr_time_Q_abs,
        "x_time": x_time,
        "peaks_I": peaks_I,
        "peaks_Q": peaks_Q,
    }


def print_single_result(res, title="RESULTADO"):
    print("\n=========================")
    print(title)
    print("=========================")
    print(f"Modo de processamento: {res['proc_label']}")
    print(f"Frame start: {res['frame_start']}")
    print(f"Lag ótimo encontrado: {res['lag']} amostras")
    print(f"Lag módulo UPS: {res['lag_mod_ups']}")
    print(f"Pico da correlação (valor absoluto): {res['corr_peak_abs']:.6f}")
    print(f"Pico da correlação normalizado: {res['corr_peak_norm']:.6f}")
    print(res['alinhamento_txt'])
    print(f"Melhor offset por correlação complexa: {res['best_off']}")
    print(f"Melhor offset por energia: {res['best_off_energy']}")
    print(f"Métrica complexa no offset ótimo: {res['best_metric_sum']:.6f}")
    print(f"Ajuste linear I: a={res['fit_aI']:.6f}, b={res['fit_bI']:.6f}")
    print(f"Ajuste linear Q: a={res['fit_aQ']:.6f}, b={res['fit_bQ']:.6f}")
    print(f"std txI={res['std_txI']:.6f}, std rxI={res['std_rxI']:.6f}")
    print(f"std txQ={res['std_txQ']:.6f}, std rxQ={res['std_rxQ']:.6f}")
    print(f"Desvio-padrão ruído final magnitude: {res['noise_std_mag_sym']:.8f} V")


def plot_single_run(res, rate_label="config_atual"):
    Ls = res["Ls"]
    ns = np.arange(Ls)

    x_time = res["x_time"]
    plot_len = min(SHOW_MAX_SAMPLES_X, len(x_time))
    x_time_plot = x_time[:plot_len]
    corr_time_I_plot = res["corr_time_I_abs"][:plot_len]
    corr_time_Q_plot = res["corr_time_Q_abs"][:plot_len]
    peaks_I_plot = res["peaks_I"][res["peaks_I"] < plot_len]
    peaks_Q_plot = res["peaks_Q"][res["peaks_Q"] < plot_len]

    plt.figure(figsize=(8, 4), dpi=120)
    plt.plot(res["lags"], res["corr"], label="Corr")
    plt.axvline(res["lag"], linestyle="--", label=f"Lag={res['lag']}")
    plt.title(f"Correlação vs lag ({rate_label})")
    plt.xlabel("Lag")
    plt.ylabel("Corr")
    plt.legend()
    plt.grid()

    plt.figure(figsize=(12, 4), dpi=120)
    plt.plot(x_time_plot, corr_time_I_plot, label="|Corr| I")
    for p in peaks_I_plot:
        plt.axvline(p, linestyle="--", alpha=0.5)
    plt.axvline(res["frame_start"], linestyle="-", label="Frame start")
    plt.title(f"Correlação no RX - I ({rate_label})")
    plt.xlabel("Amostra")
    plt.ylabel("Correlação")
    plt.xlim(0, plot_len)
    plt.legend()
    plt.grid()

    plt.figure(figsize=(12, 4), dpi=120)
    plt.plot(x_time_plot, corr_time_Q_plot, label="|Corr| Q")
    for p in peaks_Q_plot:
        plt.axvline(p, linestyle="--", alpha=0.5)
    plt.axvline(res["frame_start"], linestyle="-", label="Frame start")
    plt.title(f"Correlação no RX - Q ({rate_label})")
    plt.xlabel("Amostra")
    plt.ylabel("Correlação")
    plt.xlim(0, plot_len)
    plt.legend()
    plt.grid()

    plt.figure(figsize=(8, 4), dpi=120)
    plt.plot(res["offs_I"], res["mets_sum"], 'o-')
    plt.axvline(res["best_off"], linestyle="--", label=f"Off={res['best_off']}")
    plt.title(f"Correlação vs offset ({rate_label})")
    plt.xlabel("Offset")
    plt.ylabel("Correlação")
    plt.xticks(np.arange(UPS))
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.grid()

    plt.figure(figsize=(8, 4), dpi=120)
    plt.plot(ns[zs0:min(zs1, Ls)], res["txI_sym_aligned"][zs0:min(zs1, Ls)], 'o-', label="TX I real")
    plt.plot(ns[zs0:min(zs1, Ls)], res["rxI_sym_fit"][zs0:min(zs1, Ls)], 'o-', label="RX I ajustado")
    plt.title(f"Símbolos I finais ({rate_label})")
    plt.xlabel("Símbolo")
    plt.ylabel("V")
    plt.legend()
    plt.grid()

    plt.figure(figsize=(8, 4), dpi=120)
    plt.plot(ns[zs0:min(zs1, Ls)], res["txQ_sym_aligned"][zs0:min(zs1, Ls)], 'o-', label="TX Q real")
    plt.plot(ns[zs0:min(zs1, Ls)], res["rxQ_sym_fit"][zs0:min(zs1, Ls)], 'o-', label="RX Q ajustado")
    plt.title(f"Símbolos Q finais ({rate_label})")
    plt.xlabel("Símbolo")
    plt.ylabel("V")
    plt.legend()
    plt.grid()

    plt.figure(figsize=(6, 6), dpi=120)
    plt.scatter(res["tx_symbols_I"], res["tx_symbols_Q"], s=3, alpha=0.6)
    plt.title(f"Constelação TX final enviada ({rate_label})")
    plt.xlabel("I TX")
    plt.ylabel("Q TX")
    plt.grid()
    plt.axis("equal")

    plt.figure(figsize=(6, 6), dpi=120)
    plt.scatter(res["rxI_sym_fit"], res["rxQ_sym_fit"], s=3, alpha=0.6)
    plt.title(f"Constelação RX final recebida ({rate_label})")
    plt.xlabel("I RX")
    plt.ylabel("Q RX")
    plt.grid()
    plt.axis("equal")

    plt.figure(figsize=(6, 6), dpi=120)
    plt.hist2d(res["rxI_sym_fit"], res["rxQ_sym_fit"], bins=100)
    plt.title(f"Densidade da constelação RX ({rate_label})")
    plt.xlabel("I RX")
    plt.ylabel("Q RX")
    plt.grid()
    plt.axis("equal")
    plt.colorbar(label="Contagens")


def plot_summary(summary_rows, x_label):
    x = np.array([row["rate_value"] for row in summary_rows], dtype=float)
    labels = [row["rate_label"] for row in summary_rows]
    noise_std = np.array([row["noise_std_mag_sym_mean"] for row in summary_rows], dtype=float)
    lag_mean = np.array([row["lag_mean"] for row in summary_rows], dtype=float)

    plt.figure(figsize=(8, 4), dpi=120)
    plt.plot(x, noise_std, 'o-')
    plt.title("Ruído final em função da taxa/velocidade")
    plt.xlabel(x_label)
    plt.ylabel("Desvio-padrão do ruído final (V)")
    for xi, yi, lab in zip(x, noise_std, labels):
        plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(0, 6), ha='center')
    plt.grid()

    plt.figure(figsize=(8, 4), dpi=120)
    plt.plot(x, lag_mean, 'o-')
    plt.title("Lag ótimo em função da taxa")
    plt.xlabel(x_label)
    plt.ylabel("Lag ótimo médio (amostras)")
    for xi, yi, lab in zip(x, lag_mean, labels):
        plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(0, 6), ha='center')
    plt.grid()


def save_all_figures(prefix=FIG_PREFIX):
    fig_nums = plt.get_fignums()
    paths = []
    for i, num in enumerate(fig_nums, start=1):
        fig = plt.figure(num)
        path = f"{prefix}_{i:02d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        paths.append(path)
    return paths


print("A abrir portas...")
tx = serial.Serial(tx_port, baud, timeout=0.1)
rx = serial.Serial(rx_port, baud, timeout=0.1)

time.sleep(STARTUP_WAIT)

tx.reset_input_buffer()
rx.reset_input_buffer()
tx.reset_output_buffer()
rx.reset_output_buffer()

print("Mensagens iniciais TX:", read_lines(tx, 0.5))
print("Mensagens iniciais RX:", read_lines(rx, 0.5))

all_run_results = []
summary_rows = []
rate_unit = SWEEP[0].get("rate_unit", "taxa") if len(SWEEP) > 0 else "taxa"

try:
    for cfg in SWEEP:
        print("\n==================================================")
        print(f"CONFIGURAÇÃO: {cfg['rate_label']}  (valor={cfg['rate_value']})")
        print("==================================================")

        set_rate_if_supported(tx, rx, cfg)

        rep_results = []
        nrep = int(cfg.get("repeats", 1))

        for rep in range(nrep):
            print("\n----------------------------------------")
            print(f"Repetição {rep + 1}/{nrep}")
            print("----------------------------------------")

            tx_d, rx_d, tx_s = acquire_once(tx, rx)
            res = analyse_once(tx_d, rx_d, tx_s)
            print_single_result(res, title=f"RESULTADO {cfg['rate_label']} - repetição {rep + 1}")
            rep_results.append(res)
            all_run_results.append((cfg, rep, res))

        row = {
            "rate_label": cfg["rate_label"],
            "rate_value": float(cfg["rate_value"]),
            "lag_mean": safe_mean([r["lag"] for r in rep_results]),
            "lag_std": safe_std([r["lag"] for r in rep_results]),
            "noise_std_mag_sym_mean": safe_mean([r["noise_std_mag_sym"] for r in rep_results]),
            "noise_var_mag_sym_mean": safe_mean([r["noise_var_mag_sym"] for r in rep_results]),
            "best_off_mean": safe_mean([r["best_off"] for r in rep_results]),
        }
        summary_rows.append(row)

        print("\nResumo desta configuração:")
        for k, v in row.items():
            print(f"  {k}: {v}")

    tx.write(b'i')
    tx.flush()
    time.sleep(0.2)
    print("\nInfo TX:", read_lines(tx, 0.5))

finally:
    tx.close()
    rx.close()

if len(all_run_results) > 0:
    cfg_last, rep_last, res_last = all_run_results[-1]
    plot_single_run(res_last, rate_label=f"{cfg_last['rate_label']} rep{rep_last+1}")

plot_summary(summary_rows, x_label=f"Taxa / velocidade ({rate_unit})")

if SAVE_FIGURES:
    saved = save_all_figures(FIG_PREFIX)
    print("\nFiguras gravadas:")
    for p in saved:
        print("  ", p)

print("\n==================================================")
print("RESUMO FINAL DO ESTUDO")
print("==================================================")
for row in summary_rows:
    print(
        f"{row['rate_label']}: "
        f"lag_mean={row['lag_mean']:.3f}, "
        f"noise_std_mag_sym_mean={row['noise_std_mag_sym_mean']:.8f} V, "
        f"noise_var_mag_sym_mean={row['noise_var_mag_sym_mean']:.8f} V^2"
    )

plt.tight_layout()
plt.show()
