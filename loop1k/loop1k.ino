#include <Arduino.h>
#include <math.h>

#define DAC_I A12
#define DAC_Q A13
#define SYNC_PIN A7

const int QAM = 64;
const int SQRT_QAM = 8;

const int UPS = 8;
const int N_SYMBOLS = 1000;
const int N_POINTS = N_SYMBOLS * UPS;

const float alpha = 0.4f;
const int Ntaps = 10 * UPS + 1;

const float sigma = 15.0f;
const float n_target = 10.0f;

const float SYMBOL_RATE = 1000.0f;
const float SAMPLE_RATE = SYMBOL_RATE * UPS;
const unsigned long SAMPLE_PERIOD_US =
  (unsigned long)(1000000.0f / SAMPLE_RATE);

const unsigned long SYNC_PULSE_US = 9;

float valuesI[QAM];
float valuesQ[QAM];
float probs[QAM];
float cdf[QAM];
float rrc[Ntaps];

// frame simbólico fixo de 1000 símbolos
float symbolsI[N_SYMBOLS];
float symbolsQ[N_SYMBOLS];

// estado do filtro em tempo real
float delayI[Ntaps] = {0};
float delayQ[Ntaps] = {0};

// escala do DAC
float frameMaxAbs = 1.0f;

// controlo temporal
volatile int symbolIndex = 0;
volatile int upsamplePhase = 0;
unsigned long nextSampleTime = 0;

// símbolo atual
float currentSymI = 0.0f;
float currentSymQ = 0.0f;

// ======================================================
// RNG / sampling MB
// ======================================================

int sampleSymbolMB() {
  float u = (float)random(0, 1000000) / 1000000.0f;
  for (int i = 0; i < QAM; i++) {
    if (u <= cdf[i]) return i;
  }
  return QAM - 1;
}

// ======================================================
// CONSTELAÇÃO
// ======================================================

void initConstellationMB() {
  float levels[SQRT_QAM];

  for (int i = 0; i < SQRT_QAM; i++) {
    levels[i] = i - (SQRT_QAM - 1) / 2.0f;
  }

  int k = 0;
  for (int q = 0; q < SQRT_QAM; q++) {
    for (int i = 0; i < SQRT_QAM; i++) {
      valuesI[k] = levels[i];
      valuesQ[k] = levels[q];
      k++;
    }
  }

  float sum = 0.0f;
  for (int i = 0; i < QAM; i++) {
    float r2 = valuesI[i] * valuesI[i] + valuesQ[i] * valuesQ[i];
    probs[i] = expf(-r2 / sigma);
    sum += probs[i];
  }

  for (int i = 0; i < QAM; i++) probs[i] /= sum;

  float n_base = 0.0f;
  for (int i = 0; i < QAM; i++) {
    float r2 = valuesI[i] * valuesI[i] + valuesQ[i] * valuesQ[i];
    n_base += probs[i] * r2;
  }

  float scale = sqrtf(n_target / n_base);
  for (int i = 0; i < QAM; i++) {
    valuesI[i] *= scale;
    valuesQ[i] *= scale;
  }

  float acc = 0.0f;
  for (int i = 0; i < QAM; i++) {
    acc += probs[i];
    cdf[i] = acc;
  }
  cdf[QAM - 1] = 1.0f;
}

// ======================================================
// RRC
// ======================================================

void initRRC() {
  const float T = (float)UPS;
  const float eps = 1e-6f;

  for (int i = 0; i < Ntaps; i++) {
    float t = i - (Ntaps - 1) / 2.0f;

    if (fabsf(t) < eps) {
      rrc[i] = 1.0f + alpha * (4.0f / PI - 1.0f);
    }
    else if (fabsf(fabsf(t) - T / (4.0f * alpha)) < eps) {
      rrc[i] = (alpha / sqrtf(2.0f)) *
               ((1.0f + 2.0f / PI) * sinf(PI / (4.0f * alpha)) +
                (1.0f - 2.0f / PI) * cosf(PI / (4.0f * alpha)));
    }
    else {
      float num =
        sinf(PI * t * (1.0f - alpha) / T) +
        4.0f * alpha * t / T * cosf(PI * t * (1.0f + alpha) / T);

      float den =
        PI * t / T * (1.0f - powf(4.0f * alpha * t / T, 2.0f));

      rrc[i] = num / den;
    }
  }

  float energy = 0.0f;
  for (int i = 0; i < Ntaps; i++) energy += rrc[i] * rrc[i];
  energy = sqrtf(energy);
  for (int i = 0; i < Ntaps; i++) rrc[i] /= energy;
}

// ======================================================
// FILTRO / MAPEAMENTO
// ======================================================

void resetFilterState() {
  for (int i = 0; i < Ntaps; i++) {
    delayI[i] = 0.0f;
    delayQ[i] = 0.0f;
  }
}

float filterSample(float *delay, float x) {
  for (int i = Ntaps - 1; i > 0; i--) delay[i] = delay[i - 1];
  delay[0] = x;

  float y = 0.0f;
  for (int i = 0; i < Ntaps; i++) y += delay[i] * rrc[i];
  return y;
}

uint16_t mapDAC(float x, float maxAbs) {
  float y = (x + maxAbs) * (4095.0f / (2.0f * maxAbs));
  if (y < 0.0f) y = 0.0f;
  if (y > 4095.0f) y = 4095.0f;
  return (uint16_t)y;
}

// ======================================================
// FRAME
// ======================================================

void buildSymbolFrame() {
  for (int s = 0; s < N_SYMBOLS; s++) {
    int idx = sampleSymbolMB();
    symbolsI[s] = valuesI[idx];
    symbolsQ[s] = valuesQ[idx];
  }
}

float computeFrameMaxAbs() {
  float tmpDelayI[Ntaps] = {0};
  float tmpDelayQ[Ntaps] = {0};
  float maxAbs = 0.0f;

  for (int s = 0; s < N_SYMBOLS; s++) {
    for (int k = 0; k < UPS; k++) {
      float xI = (k == 0) ? symbolsI[s] : 0.0f;
      float xQ = (k == 0) ? symbolsQ[s] : 0.0f;

      float yI = filterSample(tmpDelayI, xI);
      float yQ = filterSample(tmpDelayQ, xQ);

      if (fabsf(yI) > maxAbs) maxAbs = fabsf(yI);
      if (fabsf(yQ) > maxAbs) maxAbs = fabsf(yQ);
    }
  }

  if (maxAbs < 1e-9f) maxAbs = 1.0f;
  return maxAbs;
}

void prepareFrame() {
  buildSymbolFrame();
  frameMaxAbs = computeFrameMaxAbs();

  resetFilterState();
  symbolIndex = 0;
  upsamplePhase = 0;
  currentSymI = symbolsI[0];
  currentSymQ = symbolsQ[0];
}

void restartFrame() {
  noInterrupts();
  resetFilterState();
  symbolIndex = 0;
  upsamplePhase = 0;
  currentSymI = symbolsI[0];
  currentSymQ = symbolsQ[0];
  nextSampleTime = micros();
  interrupts();
  Serial.println("TX_FRAME_RESTARTED");
}

void rebuildAndRestartFrame() {
  prepareFrame();
  noInterrupts();
  nextSampleTime = micros();
  interrupts();
  Serial.println("TX_FRAME_REBUILT");
}

// ======================================================
// DUMPS
// ======================================================

void dumpWaveform() {
  float tmpDelayI[Ntaps] = {0};
  float tmpDelayQ[Ntaps] = {0};

  Serial.println("BEGIN");

  for (int s = 0; s < N_SYMBOLS; s++) {
    for (int k = 0; k < UPS; k++) {
      float xI = (k == 0) ? symbolsI[s] : 0.0f;
      float xQ = (k == 0) ? symbolsQ[s] : 0.0f;

      float yI = filterSample(tmpDelayI, xI);
      float yQ = filterSample(tmpDelayQ, xQ);

      uint16_t dI = mapDAC(yI, frameMaxAbs);
      uint16_t dQ = mapDAC(yQ, frameMaxAbs);

      Serial.print(dI);
      Serial.print(',');
      Serial.println(dQ);
    }

    if ((s % 32) == 31) delay(1);
  }

  Serial.println("END");
}

void dumpSymbols() {
  Serial.println("SYM_BEGIN");
  for (int i = 0; i < N_SYMBOLS; i++) {
    Serial.print(symbolsI[i], 6);
    Serial.print(',');
    Serial.println(symbolsQ[i], 6);
    if ((i % 64) == 63) delay(1);
  }
  Serial.println("SYM_END");
}

// ======================================================
// SERIAL
// ======================================================

void handleSerial() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == 'r' || c == 'R') {
      restartFrame();
    }
    else if (c == 'n' || c == 'N') {
      rebuildAndRestartFrame();
    }
    else if (c == 'd' || c == 'D') {
      dumpWaveform();
    }
    else if (c == 's' || c == 'S') {
      dumpSymbols();
    }
    else if (c == 'i' || c == 'I') {
      Serial.print("UPS=");
      Serial.print(UPS);
      Serial.print(" N_SYMBOLS=");
      Serial.print(N_SYMBOLS);
      Serial.print(" N_POINTS=");
      Serial.print(N_POINTS);
      Serial.print(" SYMBOL_RATE=");
      Serial.print(SYMBOL_RATE, 3);
      Serial.print(" SAMPLE_RATE=");
      Serial.print(SAMPLE_RATE, 3);
      Serial.print(" frameMaxAbs=");
      Serial.println(frameMaxAbs, 6);
    }
  }
}

// ======================================================
// SETUP / LOOP
// ======================================================

void setup() {
  Serial.begin(115200);

  analogWriteResolution(12);
  pinMode(DAC_I, OUTPUT);
  pinMode(DAC_Q, OUTPUT);
  pinMode(SYNC_PIN, OUTPUT);
  digitalWrite(SYNC_PIN, LOW);

  randomSeed(1);

  initConstellationMB();
  initRRC();
  prepareFrame();

  delay(1000);
  nextSampleTime = micros();
  Serial.println("TX_READY");
}

void loop() {
  handleSerial();

  unsigned long now = micros();
  if ((long)(now - nextSampleTime) < 0) return;

  float xI = (upsamplePhase == 0) ? currentSymI : 0.0f;
  float xQ = (upsamplePhase == 0) ? currentSymQ : 0.0f;

  float yI = filterSample(delayI, xI);
  float yQ = filterSample(delayQ, xQ);

  analogWrite(DAC_I, mapDAC(yI, frameMaxAbs));
  analogWrite(DAC_Q, mapDAC(yQ, frameMaxAbs));

  digitalWrite(SYNC_PIN, HIGH);
  delayMicroseconds(SYNC_PULSE_US);
  digitalWrite(SYNC_PIN, LOW);

  upsamplePhase++;

  if (upsamplePhase >= UPS) {
    upsamplePhase = 0;
    symbolIndex++;

    if (symbolIndex >= N_SYMBOLS) {
      symbolIndex = 0;
    }

    currentSymI = symbolsI[symbolIndex];
    currentSymQ = symbolsQ[symbolIndex];
  }

  nextSampleTime += SAMPLE_PERIOD_US;
}