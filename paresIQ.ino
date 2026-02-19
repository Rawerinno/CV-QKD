#include <Arduino.h>
#include "SignalData.h"

#define DAC_I A12
#define DAC_Q A13
#define FS_HZ 10000

unsigned long period_us;
unsigned long last_time = 0;
size_t idx = 0;

unsigned long t_report = 0;
uint32_t samples_sent = 0;

uint16_t minI=4095, maxI=0, minQ=4095, maxQ=0;

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  analogWriteResolution(12);
  pinMode(DAC_I, OUTPUT);
  pinMode(DAC_Q, OUTPUT);

  period_us = 1000000UL / FS_HZ;
  t_report = millis();
}

void loop() {
  unsigned long now = micros();

  if (now - last_time >= period_us) {
    last_time += period_us;

    uint16_t I = signal_data[idx];
    uint16_t Q = signal_data[idx + 1];

    analogWrite(DAC_I, I);
    analogWrite(DAC_Q, Q);

    // stats
    if (I < minI) minI = I;
    if (I > maxI) maxI = I;
    if (Q < minQ) minQ = Q;
    if (Q > maxQ) maxQ = Q;

    idx += 2;
    if (idx >= signal_words) idx = 0;

    samples_sent++;
  }

  // report a cada 1s
  if (millis() - t_report >= 1000) {
    t_report += 1000;

    Serial.print("FS alvo: "); Serial.print(FS_HZ);
    Serial.print(" | samples/s (pares I/Q): "); Serial.print(samples_sent);
    Serial.print(" | I[min,max]=["); Serial.print(minI); Serial.print(","); Serial.print(maxI); Serial.print("]");
    Serial.print(" | Q[min,max]=["); Serial.print(minQ); Serial.print(","); Serial.print(maxQ); Serial.println("]");

    samples_sent = 0;
    minI=4095; maxI=0; minQ=4095; maxQ=0;
  }
}
