#include <ESP8266WiFi.h>
#include <WiFiUdp.h>

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#ifndef STASSID
#define STASSID "LOSTLITTLEROBOT"
#define STAPSK  "mousewalk"
#endif
#define MOTOR_NUM 11
#define PwnOffset 3   // For Install, the start pin of motors
/*
 * IO_2 of ESP_01s connect the SCL of driver board
 * IO_0 of ESP_01s connect the SDA of driver board
*/
#define SCL 2
#define SDA 0
/*
 * DSM44 --> [0, 180]
 * Reference pulse duration: [0.5ms, 2.5ms] --> [0, 180]
 * 50Hz --> 20ms
 * SERVOMIN = 4096*0.5/20 = 102.4 --> 102
 * SERVOMAX = 4096*2.5/20 = 512
*/
#define SERVOMIN  102 // This is the 'minimum' pulse length count (out of 4096)
//#define SERVOMAX  512 // This is the 'maximum' pulse length count (out of 4096)
#define SERVOMAX  580 // From Test
unsigned int localPort = 6666;      // local port to listen on
const int MPU=0x68; 

// buffers for receiving and sending data
char packetBuffer[UDP_TX_PACKET_MAX_SIZE + 1]; //buffer to hold incoming packet,
String Init_ReplyBuffer = "Reply!!";       // a string to send back
int find_flag = 0;

int motorInit[MOTOR_NUM+1] = {0};
int motorValues[MOTOR_NUM+1];

void streamSplit(char *streamString, const char* flag){
  int n = 0;
  char* result = NULL;

  result = strtok(streamString, flag);
  while(result != NULL){
    String temp = String(result);
    motorValues[n++] = temp.toInt();
    result = strtok(NULL, flag);
  }
}


WiFiUDP Udp;
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
//Adafruit_MPU6050 mpu;

float AccX, AccY, AccZ;
float GyroX, GyroY, GyroZ;
void setup() {
  Serial.begin(115200);
  while (!Serial)
    delay(10);
  WiFi.mode(WIFI_STA);
  WiFi.begin(STASSID, STAPSK);
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print('.');
    delay(500);
  }
  
  Serial.print("Connected! IP address: ");
  Serial.println(WiFi.localIP());
  Serial.printf("UDP server on port %d\n", localPort);
  Udp.begin(localPort);
  /************************************************************/
  Wire.begin(SDA, SCL);
  Wire.beginTransmission(MPU);       // Start communication with MPU6050 // MPU=0x68
  Wire.write(0x6B);                  // Talk to the register 6B
  Wire.write(0x00);                  // Make reset - place a 0 into the 6B register
  Wire.endTransmission(true);        //end the transmission
  
  pwm.begin();
  pwm.setPWMFreq(60);  // This is the maximum PWM frequency
  for(int i = 0; i < MOTOR_NUM+1; i++){
    motorInit[i] = 90;
    motorValues[i] = 0;
  }
  /*
  for (int i = 0; i < 5; i++){
    if (!mpu.begin()){
      Serial.println("Failed to find MPU6050 chip");
      delay(100);
    }else{
      find_flag = 1;
      mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
      mpu.setGyroRange(MPU6050_RANGE_500_DEG);
      mpu.setFilterBandwidth(MPU6050_BAND_5_HZ);
      break;
    }
  }
  */
}

void loop() {
  // if there's data available, read a packet
  // Control data update and response the requirement
  /*
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  String Acc = "[";
  Acc += String(a.acceleration.x)+",";
  Acc += String(a.acceleration.y)+",";
  Acc += String(a.acceleration.z)+"]";
  String Gyro = "[";
  Gyro += String(g.gyro.x)+",";
  Gyro += String(g.gyro.y)+",";
  Gyro += String(g.gyro.z)+"]";
  String ReplyBuffer = "Nothing!!";
  if(find_flag == 1)
    String ReplyBuffer = "ACC:"+Acc + ";" +"Gyro:"+Gyro;
  */
  Wire.beginTransmission(MPU);
  Wire.write(0x3B);  
  Wire.endTransmission(false);
  Wire.requestFrom(MPU,12,true);  
  AccX=(Wire.read()<<8|Wire.read());
  AccY=(Wire.read()<<8|Wire.read());
  AccZ=(Wire.read()<<8|Wire.read());
  GyroX=(Wire.read()<<8|Wire.read());
  GyroY=(Wire.read()<<8|Wire.read()); 
  GyroZ=(Wire.read()<<8|Wire.read());
  //Wire.endTransmission(true);
  String Acc = "["+String(AccX)+","+String(AccY)+","+String(AccZ)+"]";
  String Gyro = "["+String(GyroX)+","+String(GyroY)+","+String(GyroZ)+"]";
  String ReplyBuffer = "ACC:"+Acc + ";" +"Gyro:"+Gyro;
  
  int packetSize = Udp.parsePacket();
  if (packetSize) {
    // read the packet into packetBufffer
    int n = Udp.read(packetBuffer, UDP_TX_PACKET_MAX_SIZE);
    packetBuffer[n] = 0;
    streamSplit(packetBuffer, ",");
    // send a reply, to the IP address and port that sent us the packet we received
    Udp.beginPacket(Udp.remoteIP(), Udp.remotePort());
    Udp.write(ReplyBuffer.c_str());
    Udp.endPacket();
  }
  /************************************************************/
  int i;
  for(i = 0; i < MOTOR_NUM; i++){
    int pwnPos = i + PwnOffset;
    int curAngle = (motorInit[i] + motorValues[i]);//%180;
    int pulse = map(curAngle, 0, 180, SERVOMIN, SERVOMAX);
    pwm.setPWM(pwnPos, 0, pulse);
  }
  delay(10);
}
