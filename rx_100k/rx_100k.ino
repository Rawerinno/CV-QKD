#include <Arduino.h>

#define ADC_I A8
#define ADC_Q A9
#define SYNC_PIN A7

const int UPS = 8;

const uint32_t TX_FRAME_SYMBOLS = 100000UL;
const uint32_t TX_FRAME_POINTS  = TX_FRAME_SYMBOLS * UPS;

// sem skip
const uint32_t RX_SKIP_SYMBOLS = 0UL;
const uint32_t RX_SKIP_POINTS  = RX_SKIP_SYMBOLS * UPS;

// 10 capturas
const uint32_t RX_BLOCK_SYMBOLS = 10000UL;
const uint32_t RX_BLOCK_POINTS  = RX_BLOCK_SYMBOLS * UPS;

uint16_t bufI[RX_BLOCK_POINTS];
uint16_t bufQ[RX_BLOCK_POINTS];

uint32_t count = 0;
uint32_t samplePos = 0;
uint32_t captureStartPoint = 0;

bool armed = false;
bool captured = false;

void resetCapture() {
  count = 0;
  samplePos = 0;
  armed = false;
  captured = false;
}

void armCapture() {
  count = 0;
  samplePos = 0;
  armed = true;
  captured = false;
  Serial.println("RX_ARMED");
}

void setCaptureOffset(uint32_t offsetPoints) {
  captureStartPoint = offsetPoints + RX_SKIP_POINTS;
  Serial.print("RX_OFFSET_SET=");
  Serial.println(captureStartPoint);
}

void dumpCapture() {
  if (!captured) {
    Serial.println("RX_NO_DATA");
    return;
  }

  Serial.println("BEGIN");
  for (uint32_t i = 0; i < RX_BLOCK_POINTS; i++) {
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

  Serial.print(" RX_SKIP_SYMBOLS=");
  Serial.print(RX_SKIP_SYMBOLS);

  Serial.print(" RX_SKIP_POINTS=");
  Serial.print(RX_SKIP_POINTS);

  Serial.print(" RX_BLOCK_SYMBOLS=");
  Serial.print(RX_BLOCK_SYMBOLS);

  Serial.print(" RX_BLOCK_POINTS=");
  Serial.print(RX_BLOCK_POINTS);

  Serial.print(" RX_CAPTURE_POINTS=");
  Serial.print(RX_BLOCK_POINTS);

  Serial.print(" captureStartPoint=");
  Serial.print(captureStartPoint);

  Serial.print(" samplePos=");
  Serial.print(samplePos);

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
    else if (c == 'o' || c == 'O') {
      uint32_t v = (uint32_t)Serial.parseInt();
      setCaptureOffset(v);
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  analogReadResolution(12);
  pinMode(SYNC_PIN, INPUT);

  resetCapture();
  Serial.println("RX_READY");
}

void loop() {
  handleSerial();

  if (!armed || captured) return;

  while (digitalRead(SYNC_PIN) == LOW) {
    handleSerial();
    if (!armed) return;
  }

  uint16_t sampleI = analogRead(ADC_I);
  uint16_t sampleQ = analogRead(ADC_Q);

  if (samplePos >= captureStartPoint && count < RX_BLOCK_POINTS) {
    bufI[count] = sampleI;
    bufQ[count] = sampleQ;
    count++;
  }

  samplePos++;

  while (digitalRead(SYNC_PIN) == HIGH) {
    handleSerial();
    if (!armed) return;
  }

  if (count >= RX_BLOCK_POINTS) {
    armed = false;
    captured = true;
    Serial.println("RX_CAPTURE_DONE");
  }
}