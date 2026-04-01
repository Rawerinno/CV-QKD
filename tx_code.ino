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

const uint32_t FIXED_SEED = 123456789UL;

float valuesI[QAM];
float valuesQ[QAM];
float probs[QAM];
float cdf[QAM];
float rrc[Ntaps];

float delayI[Ntaps] = {0};
float delayQ[Ntaps] = {0};

float currentSymI = 0.0f;
float currentSymQ = 0.0f;

volatile uint32_t symbolIndex = 0;
volatile int upsamplePhase = 0;
unsigned long nextSampleTime = 0;

float frameMaxAbs = 1.0f;

uint32_t rngState = FIXED_SEED;

void rngReset() { rngState = FIXED_SEED; }
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
  for (int i = 0; i < SQRT_QAM; i++)
    levels[i] = i - (SQRT_QAM - 1) / 2.0f;

  int k = 0;
  for (int q = 0; q < SQRT_QAM; q++)
    for (int i = 0; i < SQRT_QAM; i++) {
      valuesI[k] = levels[i];
      valuesQ[k] = levels[q];
      k++;
    }

  float sum = 0.0f;
  for (int i = 0; i < QAM; i++) {
    float r2 = valuesI[i]*valuesI[i] + valuesQ[i]*valuesQ[i];
    probs[i] = expf(-r2 / sigma);
    sum += probs[i];
  }

  for (int i = 0; i < QAM; i++) probs[i] /= sum;

  float n_base = 0.0f;
  for (int i = 0; i < QAM; i++) {
    float r2 = valuesI[i]*valuesI[i] + valuesQ[i]*valuesQ[i];
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
  cdf[QAM-1] = 1.0f;
}

void initRRC() {
  const float T = (float)UPS;
  for (int i = 0; i < Ntaps; i++) {
    float t = i - (Ntaps - 1)/2.0f;
    float num = sinf(PI*t*(1-alpha)/T) +
                4*alpha*t/T*cosf(PI*t*(1+alpha)/T);
    float den = PI*t/T*(1 - powf(4*alpha*t/T,2));
    rrc[i] = (fabsf(t)<1e-6)?1.0f:num/den;
  }
}

float filterSample(float *delay, float x) {
  for (int i = Ntaps-1; i>0; i--) delay[i]=delay[i-1];
  delay[0]=x;
  float y=0;
  for (int i=0;i<Ntaps;i++) y+=delay[i]*rrc[i];
  return y;
}

uint16_t mapDAC(float x, float maxAbs) {
  float y=(x+maxAbs)*(4095.0/(2*maxAbs));
  if(y<0)y=0;
  if(y>4095)y=4095;
  return (uint16_t)y;
}

void setup() {
  Serial.begin(115200);
  analogWriteResolution(12);
  pinMode(DAC_I,OUTPUT);
  pinMode(DAC_Q,OUTPUT);
  pinMode(SYNC_PIN,OUTPUT);

  initConstellationMB();
  initRRC();

  nextSampleTime = micros();
}

void loop() {
  unsigned long now = micros();
  if ((long)(now - nextSampleTime) < 0) return;

  float xI = (upsamplePhase==0)?valuesI[sampleSymbolMB()]:0.0f;
  float xQ = (upsamplePhase==0)?valuesQ[sampleSymbolMB()]:0.0f;

  float yI = filterSample(delayI,xI);
  float yQ = filterSample(delayQ,xQ);

  analogWrite(DAC_I,mapDAC(yI,1.0));
  analogWrite(DAC_Q,mapDAC(yQ,1.0));

  digitalWrite(SYNC_PIN,HIGH);
  delayMicroseconds(10);
  digitalWrite(SYNC_PIN,LOW);

  upsamplePhase++;
  if(upsamplePhase>=UPS) upsamplePhase=0;

  nextSampleTime += SAMPLE_PERIOD_US;
}
