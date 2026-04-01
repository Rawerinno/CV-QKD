#include <Arduino.h>
#include <math.h>

#define DAC_I A12
#define DAC_Q A13
#define SYNC_PIN A7

const int QAM = 64;
const int SQRT_QAM = 8;

const int UPS = 8;
const uint32_t N_SYMBOLS = 100000UL;
const uint32_t N_POINTS = N_SYMBOLS * UPS;

const float alpha = 0.4f;
const int Ntaps = 10 * UPS + 1;

const float sigma = 15.0f;
const float n_target = 10.0f;

const float SYMBOL_RATE = 4100.0f;
const float SAMPLE_RATE = SYMBOL_RATE * UPS;
const unsigned long SAMPLE_PERIOD_US =
  (unsigned long)(1000000.0f / SAMPLE_RATE);

const unsigned long SYNC_PULSE_US = 10;

// seed fixa
const uint32_t FIXED_SEED = 123456789UL;

float valuesI[QAM];
float valuesQ[QAM];
float probs[QAM];
float cdf[QAM];
float rrc[Ntaps];

// estado do filtro em tempo real
float delayI[Ntaps] = {0};
float delayQ[Ntaps] = {0};

// símbolo atual
float currentSymI = 0.0f;
float currentSymQ = 0.0f;
float lastSymbolI = 0.0f;
float lastSymbolQ = 0.0f;

// controlo TX
volatile uint32_t symbolIndex = 0;
volatile int upsamplePhase = 0;
unsigned long nextSampleTime = 0;

// escala real do frame
float frameMaxAbs = 1.0f;

// RNG determinístico
uint32_t rngState = FIXED_SEED;

// ===============================
// MEDIÇÃO TX PURO
// ===============================
unsigned long txFrameStartUs = 0;
unsigned long txFrameEndUs = 0;
double txFrameTimeUs = 0.0;
double txRealSampleRate = 0.0;
double txRealSymbolRate = 0.0;

void rngReset() {
  rngState = FIXED_SEED;
}

uint32_t rngNextU32() {
  rngState = 1664525UL * rngState + 1013904223UL;
  return rngState;
}

float rngUniform01() {
  return (float)((rngNextU32() >> 8) & 0x00FFFFFFUL) / 16777216.0f;
}

int sampleSymbolMB() {
  float u = rngUniform01();
  for (int i = 0; i < QAM; i++) {
    if (u <= cdf[i]) return i;
  }
  return QAM - 1;
}

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

void resetFilterState(float *dI, float *dQ) {
  for (int i = 0; i < Ntaps; i++) {
    dI[i] = 0.0f;
    dQ[i] = 0.0f;
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

void generateNewSymbol() {
  int idx = sampleSymbolMB();
  currentSymI = valuesI[idx];
  currentSymQ = valuesQ[idx];
  lastSymbolI = currentSymI;
  lastSymbolQ = currentSymQ;
}

float computeFrameMaxAbs() {
  float tmpDelayI[Ntaps] = {0};
  float tmpDelayQ[Ntaps] = {0};

  rngReset();

  float maxAbs = 0.0f;

  for (uint32_t s = 0; s < N_SYMBOLS; s++) {
    int idx = sampleSymbolMB();
    float symI = valuesI[idx];
    float symQ = valuesQ[idx];

    for (int k = 0; k < UPS; k++) {
      float xI = (k == 0) ? symI : 0.0f;
      float xQ = (k == 0) ? symQ : 0.0f;

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
  frameMaxAbs = computeFrameMaxAbs();

  rngReset();
  resetFilterState(delayI, delayQ);

  symbolIndex = 0;
  upsamplePhase = 0;
  generateNewSymbol();
}

void restartFrame() {
  noInterrupts();
  rngReset();
  resetFilterState(delayI, delayQ);
  symbolIndex = 0;
  upsamplePhase = 0;
  generateNewSymbol();
  nextSampleTime = micros();

  txFrameStartUs = micros();
  txFrameEndUs = 0;
  txFrameTimeUs = 0.0;
  txRealSampleRate = 0.0;
  txRealSymbolRate = 0.0;

  interrupts();
  Serial.println("TX_FRAME_RESTARTED");
}

void rebuildAndRestartFrame() {
  prepareFrame();

  noInterrupts();
  nextSampleTime = micros();

  txFrameStartUs = micros();
  txFrameEndUs = 0;
  txFrameTimeUs = 0.0;
  txRealSampleRate = 0.0;
  txRealSymbolRate = 0.0;

  interrupts();
  Serial.println("TX_FRAME_REBUILT");
}

void dumpLastSymbol() {
  Serial.println("SYM_BEGIN");
  Serial.print(lastSymbolI, 6);
  Serial.print(',');
  Serial.println(lastSymbolQ, 6);
  Serial.println("SYM_END");
}

void printInfo() {
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
  Serial.print(" SAMPLE_PERIOD_US=");
  Serial.print(SAMPLE_PERIOD_US);
  Serial.print(" FIXED_SEED=");
  Serial.print(FIXED_SEED);
  Serial.print(" frameMaxAbs=");
  Serial.print(frameMaxAbs, 6);
  Serial.print(" symbolIndex=");
  Serial.print(symbolIndex);
  Serial.print(" upsamplePhase=");
  Serial.print(upsamplePhase);
  Serial.print(" FRAME_TIME_US=");
  Serial.print(txFrameTimeUs, 3);
  Serial.print(" REAL_SAMPLE_RATE=");
  Serial.print(txRealSampleRate, 3);
  Serial.print(" REAL_SYMBOL_RATE=");
  Serial.println(txRealSymbolRate, 3);
}

void handleSerial() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == 'r' || c == 'R') {
      restartFrame();
    }
    else if (c == 'n' || c == 'N') {
      rebuildAndRestartFrame();
    }
    else if (c == 's' || c == 'S') {
      dumpLastSymbol();
    }
    else if (c == 'i' || c == 'I') {
      printInfo();
    }
  }
}

void setup() {
  Serial.begin(115200);

  analogWriteResolution(12);
  pinMode(DAC_I, OUTPUT);
  pinMode(DAC_Q, OUTPUT);
  pinMode(SYNC_PIN, OUTPUT);
  digitalWrite(SYNC_PIN, LOW);

  initConstellationMB();
  initRRC();
  prepareFrame();

  delay(1000);
  nextSampleTime = micros();
  txFrameStartUs = micros();

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
      txFrameEndUs = micros();
      txFrameTimeUs = (double)(txFrameEndUs - txFrameStartUs);

      if (txFrameTimeUs > 0.0) {
        txRealSampleRate = (double)N_POINTS * 1000000.0 / txFrameTimeUs;
        txRealSymbolRate = (double)N_SYMBOLS * 1000000.0 / txFrameTimeUs;
      }

      // print automático para o Python
      Serial.print("TX_RATE ");
      Serial.print("REAL_SYMBOL_RATE=");
      Serial.print(txRealSymbolRate, 3);
      Serial.print(" REAL_SAMPLE_RATE=");
      Serial.print(txRealSampleRate, 3);
      Serial.print(" FRAME_TIME_US=");
      Serial.println(txFrameTimeUs, 3);

      rngReset();
      resetFilterState(delayI, delayQ);
      symbolIndex = 0;
      txFrameStartUs = micros();
    }

    generateNewSymbol();
  }

  nextSampleTime += SAMPLE_PERIOD_US;
}