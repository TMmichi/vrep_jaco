#include "AD770X.h"

// Init the ADC. The value in parenthesis is the scaler - because the ADC is
// 16-bit, we want the max. value to be 2^16
AD770X ad7705_1(65536,A0);
AD770X ad7705_2(65536,A1);
AD770X ad7705_3(65536,A2);
AD770X ad7705_4(65536,A3);
AD770X ad7705_5(65536,A4);
AD770X ad7705_6(65536,A5);
AD770X ad7705_7(65536,10);
AD770X ad7705_8(65536,9);
AD770X ad7705_9(65536,8);
AD770X ad7705_10(65536,7);
AD770X ad7705_11(65536,6);

const int intervalR=30;
unsigned long range_timer;
unsigned long count=0;
unsigned int ADCValue1, ADCValue2, ADCValue3, ADCValue4;
unsigned int ADCValue5, ADCValue6, ADCValue7, ADCValue8;
unsigned int ADCValue9, ADCValue10, ADCValue11, ADCValue12, ADCValue13;


void setup()
{
  Serial.begin(9600);
  SPI.begin();
  ad7705_1.reset();
  ad7705_2.reset();
  ad7705_3.reset();
  ad7705_4.reset();
  ad7705_5.reset();
  ad7705_6.reset();
  ad7705_7.reset();
  ad7705_8.reset();
  ad7705_9.reset();
  ad7705_10.reset();
  ad7705_11.reset();
  Serial.println("reset");
  ad7705_1.init(AD770X::CHN_AIN1);
  ad7705_2.init(AD770X::CHN_AIN1);
  ad7705_3.init(AD770X::CHN_AIN1);
  ad7705_4.init(AD770X::CHN_AIN1);
  ad7705_5.init(AD770X::CHN_AIN1);
  ad7705_6.init(AD770X::CHN_AIN1);
  ad7705_7.init(AD770X::CHN_AIN1);
  ad7705_8.init(AD770X::CHN_AIN1);
  ad7705_9.init(AD770X::CHN_AIN1);
  ad7705_10.init(AD770X::CHN_AIN1);
  ad7705_11.init(AD770X::CHN_AIN1);
  Serial.println("init");
  
}

void loop()
{
  unsigned long currentMillis = millis();
  
    if(currentMillis >= range_timer + intervalR){
      range_timer = currentMillis+intervalR;
      ADCValue1 = ad7705_1.readADResult(AD770X::CHN_AIN1);
      ADCValue2 = ad7705_2.readADResult(AD770X::CHN_AIN1);
      ADCValue3 = ad7705_3.readADResult(AD770X::CHN_AIN1);
      ADCValue4 = ad7705_4.readADResult(AD770X::CHN_AIN1);
      ADCValue5 = ad7705_5.readADResult(AD770X::CHN_AIN1);
      ADCValue6 = ad7705_6.readADResult(AD770X::CHN_AIN1);
      ADCValue7 = ad7705_7.readADResult(AD770X::CHN_AIN1);
      ADCValue8 = ad7705_8.readADResult(AD770X::CHN_AIN1);
      ADCValue9 = ad7705_9.readADResult(AD770X::CHN_AIN1);
      ADCValue10 = ad7705_10.readADResult(AD770X::CHN_AIN1);
      ADCValue11 = ad7705_11.readADResult(AD770X::CHN_AIN1);
      
      
      Serial.print(ADCValue1);
      Serial.print(", ");
      Serial.print(ADCValue2);
      Serial.print(", ");
      Serial.print(ADCValue3);
      Serial.print(", ");
      Serial.print(ADCValue4);
      Serial.print(", ");
      Serial.print(ADCValue5);
      Serial.print(", ");
      Serial.print(ADCValue6);
      Serial.print(", ");
      Serial.print(ADCValue7);
      Serial.print(", ");
      Serial.print(ADCValue8);
      Serial.print(", ");
      Serial.print(ADCValue9);
      Serial.print(", ");
      Serial.print(ADCValue10);
      Serial.print(", ");
      Serial.println(ADCValue11);
    }
}
