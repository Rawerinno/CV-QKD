import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

PORT = "COM3"
BAUD = 115200

UPS = 8
N_SYM = 1000
TOTAL_SYM = 4000

ser = serial.Serial(PORT, BAUD, timeout=1)

dataI = []
dataQ = []

print("A receber...")

while len(dataI) < TOTAL_SYM * UPS:
    line = ser.readline().decode(errors="ignore").strip()
    try:
        a,b = line.split(",")
        dataI.append(int(a))
        dataQ.append(int(b))
    except:
        pass

ser.close()

dataI = np.array(dataI)

# remover DC
dataI = dataI - np.mean(dataI)

# downsample → símbolos
sym = dataI[::UPS]

# referência = primeiros 1000 símbolos
ref = sym[:N_SYM]

# correlação
corr = correlate(sym, ref, mode="full")

# eixo em símbolos
lags = np.arange(-len(ref)+1, len(sym))

# manter só parte positiva
valid = lags >= 0
lags = lags[valid]
corr = corr[valid]

# converter para símbolos
lags_sym = lags

plt.figure(figsize=(10,4))
plt.plot(lags_sym, corr)

plt.title("Correlação vs símbolo (detecção de frames)")
plt.xlabel("Símbolo (lag)")
plt.ylabel("Correlação")
plt.grid()

plt.show()