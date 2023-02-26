#include <Servo.h>


int servo_pin[] = {4,5,6,7,8};

//int values[5]; // An array to hold the received values

Servo thumb, index, middle, ring, pinky;

char number[50];
char c;
int state = 0;
String myStringRec;
int stringCounter = 0;
bool stringCounterStart = false;
String myRevivedString;
int stringLength = 6;

class Finger{

    
      public:
      Servo servo;
      int max, min;
      
        Fingers(Servo servo,int max, int min){
            this->servo = servo;
            this->max = max;
            this->min = min;


        }
        
        void move(int response){
          if (response == 0){
            servo.write(min);  // Pull down
            Serial.println("Servo down");
          }

          else {
            servo.write(max);   // Release
            Serial.println("Servo up");
          }
  }
};

    Finger fingers[] = {Finger{thumb,160,45}, Finger{index,160,45},Finger{middle,160,45}, Finger{ring,100,15}, Finger{pinky,160,10}};
    //Finger Thumb{thumb,180,0};


void setup() {
  // put your setup code here, to run once:
    
    Serial.begin(9600);
      for (int i = 0; i < 5; i++){

        fingers[i].servo.attach(servo_pin[i]);
        fingers[i].move(1);
      }
      //Thumb.servo.attach(4);
      
}
int values;
void loop() {
  // put your main code here, to run repeatedly:

    
    // char incomingByte[5];
    // int numByte = Serial.readBytesUntil('x',incomingByte,6);
    
    //   //if (Serial.available() > 67) Serial.flush();
    //   incomingByte[numByte-1] = '\0';
      
    //   for (int i = 0; i < numByte; i++){

        
    //     if(incomingByte[i] != 'x'){
    //       fingers[i].move(incomingByte[i] - '0');
    //     delayMicroseconds(1000);
    //     if (i == 4) Serial.println(Serial.available());
    //     }
    //   }
      
    //   if (Serial.available() > 67) Serial.getTimeout();
    //   delay(100);

    // Read the five values
    int i = 0;
    if (Serial.available()) {
    char c = Serial.read();
    
      if (c == '$') {
        stringCounterStart = true;
      }
      if (stringCounterStart == true )
      {
        if (stringCounter < stringLength)
        {
          myRevivedString = String(myRevivedString + c);
          stringCounter++;
        }
        if (stringCounter >= stringLength) {
          stringCounter = 0; stringCounterStart = false;
          for(int i =0; i<5 ; i++)
          {
            int number = myRevivedString.substring(i+1, i+2).toInt();
            fingers[i].move(number);
            Serial.print(number);

          
          }
          Serial.println("........");
          Serial.println(" ");
        // servoPinky = myRevivedString.substring(1, 2).toInt();
        // servoRing = myRevivedString.substring(2, 3).toInt();
        // servoMiddle = myRevivedString.substring(3, 4).toInt();
        // servoIndex = myRevivedString.substring(4, 5).toInt();
        // servoThumb = myRevivedString.substring(5, 6).toInt();
//        Serial.print(servoPinky);
//        Serial.print(" ");
//        Serial.print(servoRing);
//        Serial.print(" ");
//        Serial.print(servoMiddle);
//        Serial.print(" ");
//        Serial.print(servoIndex);
//        Serial.print(" ");
//        Serial.println(servoThumb);       
         myRevivedString = "";
        }
        }
          }
        }
