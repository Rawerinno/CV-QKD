#include <Arduino.h>

#define ADC_I A8
#define ADC_Q A9
#define SYNC_PIN A7

const int N_POINTS = 2000;

uint16_t bufI[N_POINTS];
uint16_t bufQ[N_POINTS];

int count = 0;
bool captured = false;
bool sent = false;

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  analogReadResolution(12);
  pinMode(SYNC_PIN, INPUT);
}

void loop() {
  if (!captured) {
    while (digitalRead(SYNC_PIN) == LOW) {}

    bufI[count] = analogRead(ADC_I);
    bufQ[count] = analogRead(ADC_Q);

    while (digitalRead(SYNC_PIN) == HIGH) {}

    count++;

    if (count >= N_POINTS) {
      captured = true;
    }
  }

  if (captured && !sent) {
    if (Serial.available()) {
      char c = Serial.read();

      if (c == 'd') {
        Serial.println("BEGIN");

        for (int i = 0; i < N_POINTS; i++) {
          Serial.print(bufI[i]);
          Serial.print(",");
          Serial.println(bufQ[i]);

          if ((i % 32) == 31) {
            delay(2);
          }
        }

        Serial.println("END");
        Serial.flush();
        sent = true;
      }
    }
  }
}