#include <Arduino.h>

#define ADC_I A8
#define ADC_Q A9
#define SYNC_PIN A7

// ======================================================
// CONFIG
// ======================================================

const int UPS = 8;

const int TX_FRAME_SYMBOLS = 1000;
const int TX_FRAME_POINTS  = TX_FRAME_SYMBOLS * UPS;   // 8000

const int RX_CAPTURE_POINTS = 30000;

// ======================================================
// BUFFER RX
// ======================================================

uint16_t bufI[RX_CAPTURE_POINTS];
uint16_t bufQ[RX_CAPTURE_POINTS];

// ======================================================
// ESTADO
// ======================================================

volatile int count = 0;
volatile bool armed = false;
volatile bool captured = false;

// deteção de flanco do sync
int lastSyncState = LOW;

// medições simples
unsigned long captureStartUs = 0;
unsigned long captureEndUs = 0;
double captureTimeUs = 0.0;
double captureRealSampleRate = 0.0;

// ======================================================
// AUXILIARES
// ======================================================

void resetCapture() {
  noInterrupts();
  count = 0;
  armed = false;
  captured = false;
  captureStartUs = 0;
  captureEndUs = 0;
  captureTimeUs = 0.0;
  captureRealSampleRate = 0.0;
  interrupts();

  lastSyncState = digitalRead(SYNC_PIN);
}

void armCapture() {
  noInterrupts();
  count = 0;
  armed = true;
  captured = false;
  captureStartUs = 0;
  captureEndUs = 0;
  captureTimeUs = 0.0;
  captureRealSampleRate = 0.0;
  interrupts();

  lastSyncState = digitalRead(SYNC_PIN);
  Serial.println("RX_ARMED");
}

void finishCapture() {
  armed = false;
  captured = true;
  captureEndUs = micros();
  captureTimeUs = (double)(captureEndUs - captureStartUs);

  if (captureTimeUs > 0.0) {
    captureRealSampleRate = (double)RX_CAPTURE_POINTS * 1000000.0 / captureTimeUs;
  } else {
    captureRealSampleRate = 0.0;
  }

  Serial.println("RX_CAPTURE_DONE");
}

void dumpCapture() {
  if (!captured) {
    Serial.println("RX_NO_DATA");
    return;
  }

  Serial.println("BEGIN");
  for (int i = 0; i < RX_CAPTURE_POINTS; i++) {
    Serial.print(bufI[i]);
    Serial.print(',');
    Serial.println(bufQ[i]);

    if ((i % 64) == 63) delay(1);
  }
  Serial.println("END");
}

void printInfo() {
  Serial.print("UPS=");
  Serial.print(UPS);

  Serial.print(" TX_FRAME_SYMBOLS=");
  Serial.print(TX_FRAME_SYMBOLS);

  Serial.print(" TX_FRAME_POINTS=");
  Serial.print(TX_FRAME_POINTS);

  Serial.print(" RX_CAPTURE_POINTS=");
  Serial.print(RX_CAPTURE_POINTS);

  Serial.print(" count=");
  Serial.print(count);

  Serial.print(" armed=");
  Serial.print(armed ? 1 : 0);

  Serial.print(" captured=");
  Serial.print(captured ? 1 : 0);

  Serial.print(" CAPTURE_TIME_US=");
  Serial.print(captureTimeUs, 3);

  Serial.print(" REAL_SAMPLE_RATE=");
  Serial.println(captureRealSampleRate, 3);
}

void handleSerial() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == 'c' || c == 'C') {
      armCapture();
    }
    else if (c == 'd' || c == 'D') {
      dumpCapture();
    }
    else if (c == 'r' || c == 'R') {
      resetCapture();
      Serial.println("RX_RESET");
    }
    else if (c == 'i' || c == 'I') {
      printInfo();
    }
  }
}

// ======================================================
// CAPTURA SINCRONIZADA
// ======================================================
// Captura uma amostra por pulso no SYNC_PIN.
// Usa deteção de flanco ascendente para não repetir leituras
// enquanto o pino estiver HIGH.
// ======================================================

void captureStep() {
  if (!armed || captured) return;

  int syncNow = digitalRead(SYNC_PIN);

  // flanco ascendente
  if (lastSyncState == LOW && syncNow == HIGH) {
    if (count == 0) {
      captureStartUs = micros();
    }

    if (count < RX_CAPTURE_POINTS) {
      bufI[count] = analogRead(ADC_I);
      bufQ[count] = analogRead(ADC_Q);
      count++;
    }

    if (count >= RX_CAPTURE_POINTS) {
      finishCapture();
    }
  }

  lastSyncState = syncNow;
}

// ======================================================
// SETUP
// ======================================================

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  analogReadResolution(12);

  // A8/A9 são analógicos only
  pinMode(SYNC_PIN, INPUT);

  resetCapture();
  Serial.println("RX_READY");
}

// ======================================================
// LOOP
// ======================================================

void loop() {
  handleSerial();
  captureStep();
}