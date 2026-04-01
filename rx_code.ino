#include <Arduino.h>

#define ADC_I A8
#define ADC_Q A9
#define SYNC_PIN A7

const int UPS = 8;

const uint32_t RX_BLOCK_SYMBOLS = 10000UL;
const uint32_t RX_BLOCK_POINTS  = RX_BLOCK_SYMBOLS * UPS;

uint16_t bufI[RX_BLOCK_POINTS];
uint16_t bufQ[RX_BLOCK_POINTS];

uint32_t count = 0;
bool capturing = false;

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  pinMode(SYNC_PIN, INPUT);
}

void loop() {
  if (!capturing && digitalRead(SYNC_PIN)==HIGH) {
    capturing = true;
    count = 0;
  }

  if (capturing && count < RX_BLOCK_POINTS) {
    bufI[count] = analogRead(ADC_I);
    bufQ[count] = analogRead(ADC_Q);
    count++;
  }

  if (count >= RX_BLOCK_POINTS) {
    Serial.println("BEGIN");
    for (uint32_t i=0;i<RX_BLOCK_POINTS;i++) {
      Serial.print(bufI[i]);
      Serial.print(",");
      Serial.println(bufQ[i]);
    }
    Serial.println("END");
    capturing = false;
  }
}
