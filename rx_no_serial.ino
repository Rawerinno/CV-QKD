#include <Arduino.h>

#define ADC_I A8
#define ADC_Q A9
#define SYNC_PIN A7

// =========================
// CONFIG
// =========================

const int UPS = 8;

// frame TX: 1000 símbolos * 8 = 8000 amostras
const int TX_FRAME_SYMBOLS = 1000;
const int TX_FRAME_POINTS  = TX_FRAME_SYMBOLS * UPS;   // 8000

// captura longa RX para ver vários picos no gráfico
const int RX_CAPTURE_POINTS = 30000;

uint16_t bufI[RX_CAPTURE_POINTS];
uint16_t bufQ[RX_CAPTURE_POINTS];

int count = 0;
bool armed = false;
bool captured = false;

// =========================
// AUXILIARES
// =========================

void resetCapture() {
  count = 0;
  armed = false;
  captured = false;
}

void armCapture() {
  count = 0;
  armed = true;
  captured = false;
  Serial.println("RX_ARMED");
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
  Serial.println(captured ? 1 : 0);
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

// =========================
// SETUP
// =========================

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  analogReadResolution(12);

  // A8 e A9 são analógicos only no GIGA -> não usar pinMode neles
  pinMode(SYNC_PIN, INPUT);

  resetCapture();
  Serial.println("RX_READY");
}

// =========================
// LOOP
// =========================

void loop() {
  handleSerial();

  if (!armed || captured) return;

  while (digitalRead(SYNC_PIN) == LOW) {
    handleSerial();
    if (!armed) return;
  }

  if (count < RX_CAPTURE_POINTS) {
    bufI[count] = analogRead(ADC_I);
    bufQ[count] = analogRead(ADC_Q);
    count++;
  }

  while (digitalRead(SYNC_PIN) == HIGH) {
    handleSerial();
    if (!armed) return;
  }

  if (count >= RX_CAPTURE_POINTS) {
    armed = false;
    captured = true;
    Serial.println("RX_CAPTURE_DONE");
  }
}
