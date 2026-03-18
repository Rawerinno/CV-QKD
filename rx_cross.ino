#include <Arduino.h>

#define ADC_I A8
#define ADC_Q A9
#define SYNC_PIN A7

const int N_POINTS = 2000;

bool started = false;
int pointCount = 0;

void setup() {
  Serial.begin(230400);
  while (!Serial) {}

  analogReadResolution(12);
  pinMode(SYNC_PIN, INPUT);

  Serial.println("RX READY");
}

void loop() {
  if (!started) {
    if (Serial.available()) {
      char c = Serial.read();
      if (c == 's') {
        started = true;
        pointCount = 0;
        Serial.println("RX START OK");
      }
    }
    return;
  }

  if (pointCount >= N_POINTS) {
    Serial.println("RX DONE");
    while (true) {
      delay(1000);
    }
  }

  while (digitalRead(SYNC_PIN) == LOW) {}

  uint16_t I = analogRead(ADC_I);
  uint16_t Q = analogRead(ADC_Q);

  while (digitalRead(SYNC_PIN) == HIGH) {}

  Serial.print(I);
  Serial.print(",");
  Serial.println(Q);

  pointCount++;
}