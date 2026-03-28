#include <Arduino.h>
#include <math.h>

#define DAC_I A12
#define DAC_Q A13
#define SYNC_PIN A7

const int QAM = 64;
const int SQRT_QAM = 8;

const int UPS = 8;
const int N_SYMBOLS = 1000;
const int N_POINTS = N_SYMBOLS * UPS;   // 8000 amostras/frame

const float alpha = 0.4f;
const int Ntaps = 10 * UPS + 1;

const float sigma = 15.0f;
const float n_target = 10.0f;

const float SYMBOL_RATE = 4627.0f;
const float SAMPLE_RATE = SYMBOL_RATE * UPS;
const unsigned long SAMPLE_PERIOD_US =
  (unsigned long)(1000000.0f / SAMPLE_RATE);

const unsigned long SYNC_PULSE_US = 9;

float valuesI[QAM];
float valuesQ[QAM];
float probs[QAM];
float cdf[QAM];
float rrc[Ntaps];

float seqI[N_POINTS];
float seqQ[N_POINTS];
uint16_t dacSeqI[N_POINTS];
uint16_t dacSeqQ[N_POINTS];

// Símbolos originais da constelação (antes de upsampling e RRC)
float symbolsI[N_SYMBOLS];
float symbolsQ[N_SYMBOLS];

volatile int pointIndex = 0;
unsigned long nextSampleTime = 0;

int sampleSymbolMB() {
  float u = (float)random(0, 1000000) / 1000000.0f;
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

void buildSequence() {
  float delayI[Ntaps] = {0};
  float delayQ[Ntaps] = {0};

  for (int s = 0; s < N_SYMBOLS; s++) {
    int idx = sampleSymbolMB();
    symbolsI[s] = valuesI[idx];
    symbolsQ[s] = valuesQ[idx];
  }

  float maxAbs = 0.0f;

  for (int n = 0; n < N_POINTS; n++) {
    int upcount = n % UPS;
    int symIdx = n / UPS;

    float xI = (upcount == 0) ? symbolsI[symIdx] : 0.0f;
    float xQ = (upcount == 0) ? symbolsQ[symIdx] : 0.0f;

    float yI = filterSample(delayI, xI);
    float yQ = filterSample(delayQ, xQ);

    seqI[n] = yI;
    seqQ[n] = yQ;

    if (fabsf(yI) > maxAbs) maxAbs = fabsf(yI);
    if (fabsf(yQ) > maxAbs) maxAbs = fabsf(yQ);
  }

  if (maxAbs < 1e-9f) maxAbs = 1.0f;

  for (int n = 0; n < N_POINTS; n++) {
    dacSeqI[n] = mapDAC(seqI[n], maxAbs);
    dacSeqQ[n] = mapDAC(seqQ[n], maxAbs);
  }
}

void restartFrame() {
  noInterrupts();
  pointIndex = 0;
  nextSampleTime = micros();
  interrupts();
  Serial.println("TX_FRAME_RESTARTED");
}

void rebuildAndRestartFrame() {
  buildSequence();
  restartFrame();
  Serial.println("TX_FRAME_REBUILT");
}

void dumpWaveform() {
  Serial.println("BEGIN");
  for (int i = 0; i < N_POINTS; i++) {
    Serial.print(dacSeqI[i]);
    Serial.print(',');
    Serial.println(dacSeqQ[i]);
    if ((i % 64) == 63) delay(1);
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
      Serial.println(N_POINTS);
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

  randomSeed(analogRead(A0) + micros());

  initConstellationMB();
  initRRC();
  buildSequence();

  delay(1000);
  nextSampleTime = micros();
  Serial.println("TX_READY");
}

void loop() {
  handleSerial();

  unsigned long now = micros();
  if ((long)(now - nextSampleTime) < 0) return;

  analogWrite(DAC_I, dacSeqI[pointIndex]);
  analogWrite(DAC_Q, dacSeqQ[pointIndex]);

  digitalWrite(SYNC_PIN, HIGH);
  delayMicroseconds(SYNC_PULSE_US);
  digitalWrite(SYNC_PIN, LOW);

  pointIndex++;
  if (pointIndex >= N_POINTS) pointIndex = 0;

  nextSampleTime += SAMPLE_PERIOD_US;
}
