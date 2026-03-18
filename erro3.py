import time
import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

# =========================
# CONFIG
# =========================

tx_port = "COM4"
rx_port = "COM3"
baud = 115200

N = 2000
VREF = 3.3
ADC_MAX = 4095.0
UPS = 8

STARTUP_WAIT = 5.0
TIMEOUT_TOTAL = 30.0
PRINT_EVERY = 0.5

# zoom em amostras
z0 = 400
z1 = 520

# zoom em símbolos
zs0 = 40
zs1 = 70

# =========================
# FUNÇÃO PARSE
# =========================

def parse(line):
    try:
        a, b = line.split(",")
        return int(a), int(b)
    except:
        return None

# =========================
# MÉTRICA PARA OFFSET FIXO
# =========================

def best_symbol_offset_by_corr(tx, rx, ups):
    offsets = []
    metrics = []

    best_offset = 0
    best_metric = -1.0

    for off in range(ups):
        tx_sym = tx[off::ups]
        rx_sym = rx[off::ups]

        L = min(len(tx_sym), len(rx_sym))
        tx_sym = tx_sym[:L]
        rx_sym = rx_sym[:L]

        if L < 5:
            metric = 0.0
        else:
            tx0 = tx_sym - np.mean(tx_sym)
            rx0 = rx_sym - np.mean(rx_sym)

            den = np.linalg.norm(tx0) * np.linalg.norm(rx0)
            if den < 1e-12:
                metric = 0.0
            else:
                c = np.correlate(rx0, tx0, mode="full")
                metric = np.max(np.abs(c)) / den

        offsets.append(off)
        metrics.append(metric)

        if metric > best_metric:
            best_metric = metric
            best_offset = off

    return best_offset, best_metric, np.array(offsets), np.array(metrics)


# =========================
# SERIAL
# =========================

print("A abrir portas...")
tx = serial.Serial(tx_port, baud, timeout=0.1)
rx = serial.Serial(rx_port, baud, timeout=0.1)

time.sleep(STARTUP_WAIT)

tx.reset_input_buffer()
rx.reset_input_buffer()
tx.reset_output_buffer()
rx.reset_output_buffer()

print("A enviar start...")
rx.write(b's')
tx.write(b's')

rx.flush()
tx.flush()

time.sleep(0.2)

# =========================
# AQUISIÇÃO
# =========================

tx_d = []
rx_d = []

t0 = time.time()
t_last = time.time()

print("A recolher...")

while True:

    if tx.in_waiting:
        line = tx.readline().decode(errors="ignore").strip()
        p = parse(line)
        if p is not None:
            tx_d.append(p)

    if rx.in_waiting:
        line = rx.readline().decode(errors="ignore").strip()
        p = parse(line)
        if p is not None:
            rx_d.append(p)

    if time.time() - t_last > PRINT_EVERY:
        print(f"TX: {len(tx_d)}/{N}   RX: {len(rx_d)}/{N}")
        t_last = time.time()

    if len(tx_d) >= N and len(rx_d) >= N:
        print("Aquisição completa.")
        break

    if time.time() - t0 > TIMEOUT_TOTAL:
        print("Timeout atingido.")
        break

tx.close()
rx.close()

print("TX recebidos:", len(tx_d))
print("RX recebidos:", len(rx_d))

# =========================
# VERIFICAÇÃO
# =========================

if len(tx_d) == 0 or len(rx_d) == 0:
    raise RuntimeError("Não chegaram dados suficientes.")

tx_d = np.array(tx_d)
rx_d = np.array(rx_d)

Nmin = min(len(tx_d), len(rx_d))

if Nmin < 10:
    raise RuntimeError(f"Poucos dados válidos: {Nmin}")

tx_d = tx_d[:Nmin]
rx_d = rx_d[:Nmin]

# =========================
# CONVERSÃO PARA VOLTS
# =========================

dacI = tx_d[:, 0] * VREF / ADC_MAX
dacQ = tx_d[:, 1] * VREF / ADC_MAX
adcI = rx_d[:, 0] * VREF / ADC_MAX
adcQ = rx_d[:, 1] * VREF / ADC_MAX

# guardar cópias antes do alinhamento
dacI_raw = dacI.copy()
dacQ_raw = dacQ.copy()
adcI_raw = adcI.copy()
adcQ_raw = adcQ.copy()


# =========================
# CORRELAÇÃO CRUZADA GLOBAL
# =========================

x = adcI - np.mean(adcI)
y = dacI - np.mean(dacI)

corr = correlate(x, y, mode="full")
lags = correlation_lags(len(adcI), len(dacI), mode="full")

idx_peak = np.argmax(np.abs(corr))
lag = lags[idx_peak]
corr_peak = corr[idx_peak]
corr_peak_abs = np.abs(corr_peak)

den_global = np.linalg.norm(x) * np.linalg.norm(y)
corr_peak_norm = corr_peak_abs / den_global if den_global > 1e-12 else 0.0

print("\n=========================")
print("RESULTADOS DA CORRELAÇÃO")
print("=========================")
print(f"Número de amostras TX analisadas: {len(dacI)}")
print(f"Número de amostras RX analisadas: {len(adcI)}")
print(f"Lag ótimo encontrado: {lag} amostras")
print(f"Lag ótimo em símbolos: {lag / UPS:.3f}")
print(f"Pico da correlação (valor absoluto): {corr_peak_abs:.6f}")
print(f"Pico da correlação normalizado: {corr_peak_norm:.6f}")

# =========================
# ALINHAMENTO PELO LAG GLOBAL
# =========================

if lag > 0:
    adcI = adcI[lag:]
    dacI = dacI[:len(adcI)]
    adcQ = adcQ[lag:]
    dacQ = dacQ[:len(adcQ)]
    alinhamento_txt = f"RX atrasado {lag} amostras -> cortado início do RX"

elif lag < 0:
    dacI = dacI[-lag:]
    adcI = adcI[:len(dacI)]
    dacQ = dacQ[-lag:]
    adcQ = adcQ[:len(dacQ)]
    alinhamento_txt = f"TX atrasado {-lag} amostras -> cortado início do TX"

else:
    alinhamento_txt = "Lag = 0 -> não foi necessário cortar início"

L = min(len(dacI), len(adcI), len(dacQ), len(adcQ))
dacI = dacI[:L]
adcI = adcI[:L]
dacQ = dacQ[:L]
adcQ = adcQ[:L]

print(alinhamento_txt)
print(f"Comprimento final após alinhamento global: {L} amostras")

# =========================
# MELHOR OFFSET FIXO 0..UPS-1
# =========================

best_off_I, best_metric_I, offs_I, mets_I = best_symbol_offset_by_corr(dacI, adcI, UPS)
best_off_Q, best_metric_Q, offs_Q, mets_Q = best_symbol_offset_by_corr(dacQ, adcQ, UPS)

best_off = best_off_I
best_metric = best_metric_I

print("\n=========================")
print("OFFSET FIXO ENTRE 0 E UPS-1")
print("=========================")
print(f"UPS = {UPS}")
print(f"Melhor offset no canal I: {best_off_I}")
print(f"Métrica no canal I: {best_metric_I:.6f}")
print(f"Melhor offset no canal Q: {best_off_Q}")
print(f"Métrica no canal Q: {best_metric_Q:.6f}")
print(f"Offset fixo escolhido para amostragem simbólica: {best_off}")


# =========================
# SINAIS JÁ COM OFFSET FIXO
# =========================

dacI_sym = dacI[best_off::UPS]
adcI_sym = adcI[best_off::UPS]
dacQ_sym = dacQ[best_off::UPS]
adcQ_sym = adcQ[best_off::UPS]

Ls = min(len(dacI_sym), len(adcI_sym), len(dacQ_sym), len(adcQ_sym))

dacI_sym = dacI_sym[:Ls]
adcI_sym = adcI_sym[:Ls]
dacQ_sym = dacQ_sym[:Ls]
adcQ_sym = adcQ_sym[:Ls]

ns = np.arange(Ls)

print(f"Número de símbolos após offset fixo: {Ls}")

# =========================
# ERRO
# =========================

errI = adcI - dacI
errQ = adcQ - dacQ

errI_sym = adcI_sym - dacI_sym
errQ_sym = adcQ_sym - dacQ_sym

print("\n=========================")
print("ERRO APÓS ALINHAMENTO")
print("=========================")
print(f"Erro médio I (amostras)     = {np.mean(errI):.6f} V")
print(f"Erro médio Q (amostras)     = {np.mean(errQ):.6f} V")
print(f"Erro abs médio I (amostras) = {np.mean(np.abs(errI)):.6f} V")
print(f"Erro abs médio Q (amostras) = {np.mean(np.abs(errQ)):.6f} V")
print(f"Erro médio I (símbolos)     = {np.mean(errI_sym):.6f} V")
print(f"Erro médio Q (símbolos)     = {np.mean(errQ_sym):.6f} V")
print(f"Erro abs médio I (símbolos) = {np.mean(np.abs(errI_sym)):.6f} V")
print(f"Erro abs médio Q (símbolos) = {np.mean(np.abs(errQ_sym)):.6f} V")

# =========================
# EIXOS
# =========================

n = np.arange(L)

z0 = max(0, z0)
z1 = min(L, z1)

zs0 = max(0, zs0)
zs1 = min(Ls, zs1)

# =========================
# PLOTS
# =========================

# 1) Correlação
plt.figure(figsize=(8, 4), dpi=120)
plt.plot(lags, corr, label="Corr")
plt.axvline(lag, linestyle="--", label=f"Lag={lag}")
plt.title("Correlação")
plt.xlabel("Lag")
plt.ylabel("Corr")
plt.legend()
plt.grid()

# 2) Zoom I alinhado
plt.figure(figsize=(8, 4), dpi=120)
plt.plot(n[z0:z1], dacI[z0:z1], 'o-', label="TX I")
plt.plot(n[z0:z1], adcI[z0:z1], 'o-', label="RX I")
plt.title("I alinhado")
plt.xlabel("Amostra")
plt.ylabel("V")
plt.ylim(0.0, 3.3)
plt.legend()
plt.grid()

# 3) Zoom Q alinhado
plt.figure(figsize=(8, 4), dpi=120)
plt.plot(n[z0:z1], dacQ[z0:z1], 'o-', label="TX Q")
plt.plot(n[z0:z1], adcQ[z0:z1], 'o-', label="RX Q")
plt.title("Q alinhado")
plt.xlabel("Amostra")
plt.ylabel("V")
plt.ylim(0.0, 3.3)
plt.legend()
plt.grid()

# 4) Melhor offset
plt.figure(figsize=(8, 4), dpi=120)
plt.plot(offs_I, mets_I, 'o-', label="Métrica")
plt.axvline(best_off, linestyle="--", label=f"Off={best_off}")
plt.title("Offset")
plt.xlabel("Offset")
plt.ylabel("Métrica")
plt.xticks(np.arange(UPS))
plt.legend()
plt.grid()

# 5) Símbolos I
plt.figure(figsize=(8, 4), dpi=120)
plt.plot(ns[zs0:zs1], dacI_sym[zs0:zs1], 'o-', label="TX I")
plt.plot(ns[zs0:zs1], adcI_sym[zs0:zs1], 'o-', label="RX I")
plt.title("Símbolos I")
plt.xlabel("Símbolo")
plt.ylabel("V")
plt.ylim(0.0, 3.3)
plt.legend()
plt.grid()

# 6) Símbolos Q
plt.figure(figsize=(8, 4), dpi=120)
plt.plot(ns[zs0:zs1], dacQ_sym[zs0:zs1], 'o-', label="TX Q")
plt.plot(ns[zs0:zs1], adcQ_sym[zs0:zs1], 'o-', label="RX Q")
plt.title("Símbolos Q")
plt.xlabel("Símbolo")
plt.ylabel("V")
plt.ylim(0.0, 3.3)
plt.legend()
plt.grid()

# 7) Erro total
plt.figure(figsize=(8, 4), dpi=120)
plt.plot(n, errI, label="Err I")
plt.plot(n, errQ, label="Err Q")
plt.title("Erro")
plt.xlabel("Amostra")
plt.ylabel("V")
plt.legend()
plt.grid()

# 8) Zoom do erro
plt.figure(figsize=(8, 4), dpi=120)
plt.plot(n[z0:z1], errI[z0:z1], 'o-', label="Err I")
plt.plot(n[z0:z1], errQ[z0:z1], 'o-', label="Err Q")
plt.title("Erro zoom")
plt.xlabel("Amostra")
plt.ylabel("V")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()