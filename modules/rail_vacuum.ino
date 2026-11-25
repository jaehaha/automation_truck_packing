
#include <AccelStepper.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <avr/pgmspace.h>

// =====================================
// PIN SETTINGS
// =====================================

#define CLK_PIN 9
#define CW_PIN 8
#define RELAY_PIN 7

const int RELAY_ON = LOW;
const int RELAY_OFF = HIGH;

// 이동 방향이 반대이면 true로 변경
const bool INVERT_DIRECTION = false;

AccelStepper motor(
  AccelStepper::DRIVER,
  CLK_PIN,
  CW_PIN
);

// =====================================
// MOTOR SETTINGS
// =====================================

// 2000 pulses = 150 mm
float stepsPerMM = 2000.0 / 150.0;

float positionOffset = 0.0;

// Motor side = 1000 mm
// Opposite side = 0 mm
const float MIN_MM = 0.0;
const float MAX_MM = 1000.0;
const float HOME_MM = 0.0;

float speedSteps = 1000.0;
float accelSteps = 1000.0;

const float MAX_SPEED = 4000.0;
const float MAX_ACCEL = 3000.0;

// =====================================
// RANDOM TEST SETTINGS
// =====================================

const int TEST_COUNT = 100;
const unsigned long WAIT_TIME = 2000;

const int distances[] = {
  50, 100, 150, 200, 250, 300
};

const int DISTANCE_COUNT =
  sizeof(distances) / sizeof(distances[0]);

// =====================================
// SYSTEM STATE
// =====================================

bool initialized = false;
bool testing = false;
bool moving = false;
bool returning = false;
bool stopping = false;
bool suction = false;

int count = 0;

float targetMM = HOME_MM;

unsigned long lastFinish = 0;

// Fixed serial buffer
const uint8_t BUFFER_SIZE = 40;

char command[BUFFER_SIZE];

uint8_t commandIndex = 0;

bool overflow = false;

// =====================================
// FUNCTION DECLARATIONS
// =====================================

void processCommand(char* cmd);
void initializePosition();
void suctionOn();
void suctionOff();
void randomMove();
void moveMM(float mm);
void printPosition();
void printStatus();
void printHelp();

bool readNumber(const char* p, float &value);

float currentMM();

long minPositionSteps();
long maxPositionSteps();

// =====================================
// SETUP
// =====================================

void setup() {

  // Relay OFF
  digitalWrite(RELAY_PIN, RELAY_OFF);
  pinMode(RELAY_PIN, OUTPUT);

  Serial.begin(115200);

  motor.setMaxSpeed(speedSteps);
  motor.setAcceleration(accelSteps);
  motor.setMinPulseWidth(100);

  motor.setPinsInverted(
    INVERT_DIRECTION,
    false,
    false
  );

  randomSeed(analogRead(A0));

  Serial.println(F("=========================="));
  Serial.println(F("RAIL & SUCTION CONTROLLER"));
  Serial.println(F("=========================="));

  Serial.println(F("Motor side: 1000 mm"));
  Serial.println(F("Opposite side: 0 mm"));
  Serial.println(F("Home position: 0 mm"));

  Serial.println(F("SERIAL READY"));
  Serial.println(F("Use INITIALIZE or MEASURE"));
  Serial.println(F("Type HELP for commands"));

}

// =====================================
// MAIN LOOP
// =====================================

void loop() {

  motor.run();

  // Serial command processing
  while (Serial.available() > 0) {

    char c = Serial.read();

    if (c == '\n' || c == '\r') {

      if (overflow) {

        Serial.println(F("ERROR: Command too long"));

      } else if (commandIndex > 0) {

        command[commandIndex] = '\0';

        processCommand(command);

      }

      commandIndex = 0;
      overflow = false;

    } else if (c == 8 || c == 127) {

      if (commandIndex > 0) {
        commandIndex--;
      }

    } else {

      if (commandIndex < BUFFER_SIZE - 1) {

        command[commandIndex++] =
          toupper((unsigned char)c);

      } else {

        overflow = true;

      }

    }

    motor.run();

  }

  motor.run();

  // Movement completed
  if (moving && motor.distanceToGo() == 0) {

    moving = false;

    lastFinish = millis();

    if (stopping) {

      stopping = false;
      targetMM = currentMM();

      Serial.println(F("STOPPED"));

    } else {

      Serial.println(F("DONE"));

    }

    printPosition();

    if (returning) {

      returning = false;

      Serial.println(F("RETURNED TO HOME"));

    }

  }

  // Random movement test
  if (testing &&
      !moving &&
      millis() - lastFinish >= WAIT_TIME) {

    if (count >= TEST_COUNT) {

      testing = false;
      returning = true;

      Serial.println(F("TEST COMPLETE"));
      Serial.println(F("RETURNING TO HOME"));

      moveMM(HOME_MM);

    } else {

      randomMove();

    }

  }

}

// =====================================
// INITIALIZATION
// =====================================

void initializePosition() {

  motor.setCurrentPosition(0);

  positionOffset = 0.0;
  targetMM = 0.0;

  initialized = true;

  moving = false;
  stopping = false;
  testing = false;
  returning = false;

  count = 0;

  Serial.println(F("INITIALIZED"));

  printPosition();

}

// =====================================
// SUCTION CONTROL
// =====================================

void suctionOn() {

  digitalWrite(RELAY_PIN, RELAY_ON);

  suction = true;

  Serial.println(F("SUCTION ON"));

}

void suctionOff() {

  digitalWrite(RELAY_PIN, RELAY_OFF);

  suction = false;

  Serial.println(F("SUCTION OFF"));

}

// =====================================
// RANDOM MOVEMENT
// =====================================

void randomMove() {

  int distance = 0;
  int direction = 0;

  float nextMM = 0;

  bool valid = false;

  for (int i = 0; i < 100; i++) {

    distance =
      distances[random(DISTANCE_COUNT)];

    direction =
      random(2) == 0 ? -1 : 1;

    nextMM =
      targetMM + direction * distance;

    if (nextMM >= MIN_MM &&
        nextMM <= MAX_MM) {

      valid = true;
      break;

    }

  }

  if (!valid) {

    testing = false;

    Serial.println(F("ERROR: No valid movement"));

    return;

  }

  count++;

  Serial.println(F("--------------------------"));

  Serial.print(F("TEST: "));
  Serial.print(count);
  Serial.print(F(" / "));
  Serial.println(TEST_COUNT);

  Serial.print(F("FROM: "));
  Serial.print(currentMM(), 3);
  Serial.println(F(" mm"));

  Serial.print(F("MOVEMENT: "));

  if (direction > 0) {
    Serial.print('+');
  }

  Serial.print(direction * distance);
  Serial.println(F(" mm"));

  Serial.print(F("TARGET: "));
  Serial.print(nextMM, 3);
  Serial.println(F(" mm"));

  moveMM(nextMM);

}

// =====================================
// COMMAND PROCESSOR
// =====================================

void processCommand(char* cmd) {

  float value;

  // Remove leading spaces
  while (*cmd && isspace((unsigned char)*cmd)) {
    cmd++;
  }

  // Remove trailing spaces
  size_t len = strlen(cmd);

  while (len > 0 &&
         isspace((unsigned char)cmd[len - 1])) {

    cmd[--len] = '\0';

  }

  if (*cmd == '\0') {
    return;
  }

  // -----------------------------------
  // SUCTION ON
  // -----------------------------------

  if (strcmp_P(cmd, PSTR("ON")) == 0) {

    suctionOn();
    return;

  }

  // -----------------------------------
  // SUCTION OFF
  // -----------------------------------

  if (strcmp_P(cmd, PSTR("OFF")) == 0) {

    suctionOff();
    return;

  }

  // -----------------------------------
  // STOP
  // -----------------------------------

  if (strcmp_P(cmd, PSTR("STOP")) == 0 ||
      strcmp_P(cmd, PSTR("S")) == 0) {

    testing = false;
    returning = false;

    if (moving && !stopping) {

      motor.stop();

      long stopTarget = constrain(
        motor.targetPosition(),
        minPositionSteps(),
        maxPositionSteps()
      );

      motor.moveTo(stopTarget);

      stopping = true;

      Serial.println(F("STOPPING"));

    } else if (!moving) {

      Serial.println(F("ALREADY STOPPED"));

    }

    return;

  }

  // -----------------------------------
  // POSITION
  // -----------------------------------

  if (strcmp_P(cmd, PSTR("POS")) == 0) {

    printPosition();
    return;

  }

  // -----------------------------------
  // STATUS
  // -----------------------------------

  if (strcmp_P(cmd, PSTR("STATUS")) == 0) {

    printStatus();
    return;

  }

  // -----------------------------------
  // HELP
  // -----------------------------------

  if (strcmp_P(cmd, PSTR("HELP")) == 0) {

    printHelp();
    return;

  }

  // Block other commands while moving
  if (moving || testing) {

    Serial.println(F("ERROR: Motor busy"));

    return;

  }

  // -----------------------------------
  // INITIALIZE : Current position = 0mm
  // -----------------------------------

  if (strcmp_P(cmd, PSTR("INITIALIZE")) == 0 ||
      strcmp_P(cmd, PSTR("INIT")) == 0) {

    initializePosition();
    return;

  }

  // -----------------------------------
  // CAL : Distance calibration
  // -----------------------------------

  if (strncmp_P(cmd, PSTR("CAL "), 4) == 0) {

    if (!readNumber(cmd + 4, value) ||
        value < 1 ||
        value > 1000) {

      Serial.println(F("ERROR: Invalid CAL"));
      return;

    }

    stepsPerMM = 2000.0 / value;

    initialized = false;

    Serial.print(F("STEPS/MM: "));
    Serial.println(stepsPerMM, 6);

    Serial.println(
      F("Set INITIALIZE or MEASURE again")
    );

    return;

  }

  // -----------------------------------
  // MEASURE : Position correction
  // -----------------------------------

  if (strncmp_P(cmd, PSTR("MEASURE "), 8) == 0) {

    if (!readNumber(cmd + 8, value) ||
        value < MIN_MM ||
        value > MAX_MM) {

      Serial.println(F("ERROR: Invalid MEASURE"));
      return;

    }

    if (initialized) {

      Serial.print(F("POSITION ERROR: "));

      Serial.print(
        value - currentMM(),
        3
      );

      Serial.println(F(" mm"));

    }

    positionOffset =
      value -
      motor.currentPosition() / stepsPerMM;

    targetMM = value;

    initialized = true;

    Serial.println(F("POSITION CORRECTED"));

    printPosition();

    return;

  }

  // -----------------------------------
  // RATE : Pulse speed
  // -----------------------------------

  if (strncmp_P(cmd, PSTR("RATE "), 5) == 0) {

    if (!readNumber(cmd + 5, value) ||
        value < 10 ||
        value > MAX_SPEED) {

      Serial.println(F("ERROR: Invalid RATE"));
      return;

    }

    speedSteps = value;

    motor.setMaxSpeed(speedSteps);

    Serial.print(F("RATE SET: "));
    Serial.println(speedSteps);

    return;

  }

  // -----------------------------------
  // SPEED : mm/s
  // -----------------------------------

  if (strncmp_P(cmd, PSTR("SPEED "), 6) == 0) {

    if (!readNumber(cmd + 6, value) ||
        value <= 0 ||
        value * stepsPerMM > MAX_SPEED) {

      Serial.println(F("ERROR: Invalid SPEED"));
      return;

    }

    speedSteps = value * stepsPerMM;

    motor.setMaxSpeed(speedSteps);

    Serial.print(F("SPEED SET: "));
    Serial.print(value);
    Serial.println(F(" mm/s"));

    return;

  }

  // -----------------------------------
  // ACCEL : mm/s^2
  // -----------------------------------

  if (strncmp_P(cmd, PSTR("ACCEL "), 6) == 0) {

    if (!readNumber(cmd + 6, value) ||
        value <= 0 ||
        value * stepsPerMM > MAX_ACCEL) {

      Serial.println(F("ERROR: Invalid ACCEL"));
      return;

    }

    accelSteps = value * stepsPerMM;

    motor.setAcceleration(accelSteps);

    Serial.print(F("ACCEL SET: "));
    Serial.println(value);

    return;

  }

  // Check initialization
  if (!initialized) {

    Serial.println(F("ERROR: INITIALIZE first"));
    return;

  }

  // -----------------------------------
  // START : Random movement
  // -----------------------------------

  if (strcmp_P(cmd, PSTR("START")) == 0) {

    count = 0;

    targetMM = currentMM();

    testing = true;
    returning = false;

    lastFinish = millis();

    Serial.println(F("TEST START"));

    return;

  }

  // -----------------------------------
  // HOME : Return to 0mm
  // -----------------------------------

  if (strcmp_P(cmd, PSTR("HOME")) == 0 ||
      strcmp_P(cmd, PSTR("H")) == 0) {

    testing = false;
    returning = true;

    moveMM(HOME_MM);

    return;

  }

  // -----------------------------------
  // MOVE : Absolute position
  // -----------------------------------

  if (strncmp_P(cmd, PSTR("MOVE "), 5) == 0) {

    if (!readNumber(cmd + 5, value)) {

      Serial.println(F("ERROR: Invalid MOVE"));
      return;

    }

    moveMM(value);

    return;

  }

  // -----------------------------------
  // REL : Relative position
  // -----------------------------------

  if (strncmp_P(cmd, PSTR("REL "), 4) == 0) {

    if (!readNumber(cmd + 4, value)) {

      Serial.println(F("ERROR: Invalid REL"));
      return;

    }

    moveMM(currentMM() + value);

    return;

  }

  // -----------------------------------
  // F : +100mm (toward motor)
  // -----------------------------------

  if (strcmp_P(cmd, PSTR("F")) == 0) {

    moveMM(targetMM + 100.0);

    return;

  }

  // -----------------------------------
  // B : -100mm (away from motor)
  // -----------------------------------

  if (strcmp_P(cmd, PSTR("B")) == 0) {

    moveMM(targetMM - 100.0);

    return;

  }

  Serial.println(F("ERROR: Unknown command"));

}

// =====================================
// NUMBER VALIDATION
// =====================================

bool readNumber(const char* p, float &value) {

  char* end;

  value = strtod(p, &end);

  return end != p
      && *end == '\0'
      && isfinite(value);

}

// =====================================
// POSITION CALCULATION
// =====================================

float currentMM() {

  return motor.currentPosition()
         / stepsPerMM
         + positionOffset;

}

long minPositionSteps() {

  return (long)ceil(
    (MIN_MM - positionOffset) * stepsPerMM
  );

}

long maxPositionSteps() {

  return (long)floor(
    (MAX_MM - positionOffset) * stepsPerMM
  );

}

// =====================================
// POSITION CONTROL
// =====================================

void moveMM(float mm) {

  if (!isfinite(mm) ||
      mm < MIN_MM ||
      mm > MAX_MM) {

    returning = false;

    Serial.println(F("ERROR: Out of range"));

    return;

  }

  long steps = lround(
    (mm - positionOffset) * stepsPerMM
  );

  steps = constrain(
    steps,
    minPositionSteps(),
    maxPositionSteps()
  );

  targetMM = mm;

  motor.moveTo(steps);

  moving = true;
  stopping = false;

  Serial.print(F("MOVING TO: "));
  Serial.print(mm, 3);
  Serial.println(F(" mm"));

}

// =====================================
// PRINT POSITION
// =====================================

void printPosition() {

  if (!initialized) {

    Serial.println(F("POSITION: UNKNOWN"));

  } else {

    Serial.print(F("POSITION: "));
    Serial.print(currentMM(), 3);
    Serial.println(F(" mm (estimated)"));

  }

  Serial.print(F("STEPS: "));
  Serial.println(motor.currentPosition());

}

// =====================================
// PRINT STATUS
// =====================================

void printStatus() {

  Serial.println(F("=========================="));

  Serial.print(F("INITIALIZED: "));

  if (initialized) {
    Serial.println(F("YES"));
  } else {
    Serial.println(F("NO"));
  }

  Serial.print(F("MOVING: "));

  if (moving) {
    Serial.println(F("YES"));
  } else {
    Serial.println(F("NO"));
  }

  Serial.print(F("RANDOM TEST: "));

  if (testing) {
    Serial.println(F("ON"));
  } else {
    Serial.println(F("OFF"));
  }

  Serial.print(F("SUCTION: "));

  if (suction) {
    Serial.println(F("ON"));
  } else {
    Serial.println(F("OFF"));
  }

  Serial.print(F("STEPS/MM: "));
  Serial.println(stepsPerMM, 6);

  Serial.print(F("OFFSET: "));
  Serial.println(positionOffset, 3);

  Serial.print(F("SPEED mm/s: "));
  Serial.println(speedSteps / stepsPerMM);

  Serial.print(F("ACCEL mm/s^2: "));
  Serial.println(accelSteps / stepsPerMM);

  Serial.print(F("HOME: "));
  Serial.println(HOME_MM);

  Serial.print(F("MIN: "));
  Serial.println(MIN_MM);

  Serial.print(F("MAX: "));
  Serial.println(MAX_MM);

  printPosition();

  Serial.println(F("=========================="));

}

// =====================================
// HELP
// =====================================

void printHelp() {

  Serial.println(F("===== COMMANDS ====="));

  Serial.println(F("INITIALIZE : Set 0mm"));
  Serial.println(F("INIT       : Set 0mm"));

  Serial.println(F("ON         : Suction ON"));
  Serial.println(F("OFF        : Suction OFF"));

  Serial.println(F("MOVE 500   : Absolute mm"));
  Serial.println(F("REL 100    : Relative +100mm"));
  Serial.println(F("REL -100   : Relative -100mm"));

  Serial.println(F("F          : +100mm"));
  Serial.println(F("B          : -100mm"));

  Serial.println(F("HOME       : Return to 0mm"));

  Serial.println(F("START      : Random test"));
  Serial.println(F("STOP       : Stop movement"));

  Serial.println(F("MEASURE 502: Correct position"));
  Serial.println(F("CAL 150    : Calibrate distance"));

  Serial.println(F("RATE 1000  : Pulse speed"));
  Serial.println(F("SPEED 50   : Speed mm/s"));
  Serial.println(F("ACCEL 75   : Acceleration"));

  Serial.println(F("POS        : Current position"));
  Serial.println(F("STATUS     : System status"));
  Serial.println(F("HELP       : Command list"));

  Serial.println(F("===================="));

}
