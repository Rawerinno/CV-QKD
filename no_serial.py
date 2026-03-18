import time
import serial
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

PORT = "COM3"
BAUD = 115200
N = 2000

VREF = 3.3
ADC_MAX = 4095.0

# =========================
# SERIAL
# =========================

print("A abrir porta...")
ser = serial.Serial(PORT, BAUD, timeout=1)

time.sleep(4)
ser.reset_input_buffer()
ser.reset_output_buffer()

print("A pedir dump ao RX...")
ser.write(b'd')
ser.flush()

# =========================
# RECEIVE
# =========================

data = []
started = False

print("A receber...")
t0 = time.time()

while True:
    line = ser.readline().decode(errors="ignore").strip()

    if not line:
        if time.time() - t0 > 40:
            print("Timeout.")
            break
        continue

    if line == "BEGIN":
        started = True
        data = []
        print("BEGIN recebido")
        continue

    if line == "END":
        print("END recebido")
        break

    if started:
        try:
            a, b = line.split(",")
            data.append([int(a), int(b)])

            if len(data) % 200 == 0:
                print(f"{len(data)}/{N}")
        except:
            pass

ser.close()

# =========================
# CHECK
# =========================

if len(data) == 0:
    raise RuntimeError("Não chegou nenhum dado válido.")

data = np.array(data)

I_adc = data[:, 0]
Q_adc = data[:, 1]

I_v = I_adc * VREF / ADC_MAX
Q_v = Q_adc * VREF / ADC_MAX

print("Total recebido:", len(data))

# =========================
# PLOTS
# =========================

n = np.arange(len(data))

plt.figure(figsize=(8, 4), dpi=120)
plt.plot(n, I_adc, label="RX I (ADC)")
plt.plot(n, Q_adc, label="RX Q (ADC)")
plt.xlabel("Amostra")
plt.ylabel("Código ADC")
plt.title("Sinais recebidos")
plt.legend()
plt.grid()

plt.figure(figsize=(8, 4), dpi=120)
plt.plot(n, I_v, label="RX I (V)")
plt.plot(n, Q_v, label="RX Q (V)")
plt.xlabel("Amostra")
plt.ylabel("Volts")
plt.title("Sinais recebidos em Volts")
plt.ylim(0.0, 3.3)
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()