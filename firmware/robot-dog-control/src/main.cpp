#include <Arduino.h>
#include <ArduinoJson.h>
#include <ArduinoOTA.h>
#include <Preferences.h>
#include <SPIFFS.h>
#include <WebServer.h>
#include <Wire.h>
#include <WiFi.h>
#include <WiFiManager.h>

#include "AK09918.h"
#include "QMI8658.h"

#ifndef ROBOT_DOG_VERSION
#define ROBOT_DOG_VERSION "dev"
#endif

#ifndef SERVO_UART_NUM
#define SERVO_UART_NUM 1
#endif

#ifndef SERVO_TX_PIN
#define SERVO_TX_PIN 19
#endif

#ifndef SERVO_RX_PIN
#define SERVO_RX_PIN 18
#endif

#ifndef SERVO_BAUD
#define SERVO_BAUD 1000000
#endif

#ifndef WIFI_SETUP_TIMEOUT_SECONDS
#define WIFI_SETUP_TIMEOUT_SECONDS 45
#endif

#ifndef IMU_SDA_PIN
#define IMU_SDA_PIN 32
#endif

#ifndef IMU_SCL_PIN
#define IMU_SCL_PIN 33
#endif

#ifndef MOTOR_A_PWM_PIN
#define MOTOR_A_PWM_PIN 25
#endif

#ifndef MOTOR_A_IN1_PIN
#define MOTOR_A_IN1_PIN 21
#endif

#ifndef MOTOR_A_IN2_PIN
#define MOTOR_A_IN2_PIN 17
#endif

#ifndef MOTOR_B_PWM_PIN
#define MOTOR_B_PWM_PIN 26
#endif

#ifndef MOTOR_B_IN1_PIN
#define MOTOR_B_IN1_PIN 22
#endif

#ifndef MOTOR_B_IN2_PIN
#define MOTOR_B_IN2_PIN 23
#endif

static constexpr uint8_t MAX_SERVOS = 12;
static constexpr uint8_t MAX_PROGRAM_STEPS = 24;
static constexpr size_t SERIAL_JSON_CAPACITY = 6144;
static constexpr uint32_t STATE_INTERVAL_MS = 500;
static constexpr uint8_t CONFIG_VERSION = 2;
static constexpr uint16_t DEFAULT_MONITOR_INTERVAL_MS = 250;
static constexpr uint16_t DEFAULT_IMU_INTERVAL_MS = 100;
static constexpr uint32_t MOTOR_PWM_FREQ = 20000;
static constexpr uint8_t MOTOR_PWM_RESOLUTION = 8;
static constexpr uint8_t MOTOR_A_PWM_CHANNEL = 0;
static constexpr uint8_t MOTOR_B_PWM_CHANNEL = 1;
static constexpr float MOTOR_DEFAULT_LIMIT = 0.35f;
static constexpr uint8_t NET_MESSAGE_LOG_SIZE = 80;
static constexpr uint16_t SERVO_VELOCITY_INTERVAL_MS = 50;
static constexpr float STS_FULL_SCALE_DEGREES = 360.0f;
static constexpr uint8_t STS_ID_ADDR = 0x05;
static constexpr uint8_t STS_MODE_ADDR = 0x21;
static constexpr uint8_t STS_TORQUE_ENABLE_ADDR = 0x28;
static constexpr uint8_t STS_ACC_ADDR = 0x29;
static constexpr uint8_t STS_GOAL_SPEED_ADDR = 0x2E;
static constexpr uint8_t STS_TORQUE_LIMIT_ADDR = 0x30;
static constexpr uint8_t STS_LOCK_ADDR = 0x37;
static constexpr uint8_t STS_PRESENT_POSITION_ADDR = 0x38;
static constexpr uint16_t STS_TORQUE_LIMIT_MAX = 1000;
static constexpr uint8_t STS_MODE_SERVO = 0;
static constexpr uint8_t STS_MODE_MOTOR = 1;

struct ServoConfig {
  uint8_t id = 0;
  String name;
  float minAngle = 0;
  float maxAngle = 240;
  float homeAngle = 120;
  bool invert = false;
  bool enabled = true;
  float lastAngle = 120;
  float measuredAngle = 120;
  bool hasMeasuredAngle = false;
  bool monitorEnabled = true;
  uint16_t monitorIntervalMs = DEFAULT_MONITOR_INTERVAL_MS;
  uint32_t nextMonitorAt = 0;
  float velocityDps = 0;
  uint32_t lastVelocityAt = 0;
  bool motorMode = false;
  float motorSpeed = 0;
  uint16_t torqueLimit = STS_TORQUE_LIMIT_MAX;
};

struct ProgramStep {
  uint16_t durationMs = 500;
  uint16_t speed = 900;
  uint8_t accel = 50;
  float pose[MAX_SERVOS];
  bool hasPose[MAX_SERVOS];
};

struct ImuState {
  bool available = false;
  bool magAvailable = false;
  bool monitorEnabled = true;
  uint16_t intervalMs = DEFAULT_IMU_INTERVAL_MS;
  uint32_t nextSampleAt = 0;
  float accel[3] = {0, 0, 0};
  float gyro[3] = {0, 0, 0};
  int16_t mag[3] = {0, 0, 0};
  float roll = 0;
  float pitch = 0;
  float yaw = 0;
  uint32_t sampleMs = 0;
};

struct MotorChannel {
  const char *name;
  uint8_t pwmPin;
  uint8_t in1Pin;
  uint8_t in2Pin;
  uint8_t pwmChannel;
  float command;
  float limit;
};

struct NetworkMessage {
  uint32_t seq = 0;
  String line;
};

struct ServoIdentifyJob {
  bool active = false;
  uint8_t id = 0;
  float centerAngle = 180;
  float amplitude = 12;
  uint8_t phase = 0;
  uint16_t speed = 700;
  uint8_t accel = 40;
  uint32_t nextAt = 0;
};

struct ServoIdChangeResult {
  uint8_t currentId = 0;
  uint8_t nextId = 0;
  bool valid = false;
  bool unlockAck = false;
  bool idWriteAck = false;
  bool lockNewAck = false;
  bool lockFallbackAck = false;
  bool pingNext = false;
  bool pingCurrent = false;
  bool ok = false;
};

struct DefaultServoDef {
  uint8_t id;
  const char *name;
};

const DefaultServoDef DEFAULT_SERVO_DEFS[MAX_SERVOS] = {
  {1, "Left front hip swing"},
  {2, "Left front femur"},
  {3, "Left front knee"},
  {4, "Right back hip swing"},
  {5, "Right back femur"},
  {6, "Right back knee"},
  {7, "Right front hip swing"},
  {8, "Right front femur"},
  {9, "Right front knee"},
  {10, "Left back hip swing"},
  {11, "Left back femur"},
  {12, "Left back knee"}
};

class ServoBusDriver {
public:
  void begin() {
    bus = new HardwareSerial(SERVO_UART_NUM);
    bus->begin(SERVO_BAUD, SERIAL_8N1, SERVO_RX_PIN, SERVO_TX_PIN);
  }

  void moveTo(uint8_t id, uint16_t position, uint16_t speed, uint8_t accel) {
    // ST-series serial bus servo write: start at goal acceleration, then position and speed.
    // If your Waveshare board uses a different onboard servo API, adapt only this method.
    uint8_t params[] = {
      STS_ACC_ADDR,
      accel,
      lowByte(position),
      highByte(position),
      0x00,
      0x00,
      lowByte(speed),
      highByte(speed)
    };
    writePacket(id, 0x03, params, sizeof(params));
  }

  bool readPosition(uint8_t id, uint16_t &position) {
    if (!bus) return false;

    clearRx();

    uint8_t params[] = {STS_PRESENT_POSITION_ADDR, 2};
    writePacket(id, 0x02, params, sizeof(params));

    uint8_t response[8];
    if (!readResponse(id, response, sizeof(response), 20)) return false;
    position = response[5] | (static_cast<uint16_t>(response[6]) << 8);
    return true;
  }

  bool ping(uint8_t id) {
    if (!bus) return false;
    clearRx();
    writePacket(id, 0x01, nullptr, 0);
    uint8_t response[6];
    return readStatus(id, response, sizeof(response), 25);
  }

  void writeByte(uint8_t id, uint8_t address, uint8_t value) {
    uint8_t params[] = {address, value};
    writePacket(id, 0x03, params, sizeof(params));
    delay(20);
  }

  void writeWord(uint8_t id, uint8_t address, uint16_t value) {
    uint8_t params[] = {address, lowByte(value), highByte(value)};
    writePacket(id, 0x03, params, sizeof(params));
    delay(20);
  }

  bool writeByteAck(uint8_t id, uint8_t address, uint8_t value, uint16_t timeoutMs = 80) {
    if (!bus) return false;
    clearRx();
    uint8_t params[] = {address, value};
    writePacket(id, 0x03, params, sizeof(params));
    uint8_t response[6];
    return readStatus(id, response, sizeof(response), timeoutMs);
  }

  void setMode(uint8_t id, uint8_t mode) {
    writeByte(id, STS_MODE_ADDR, mode);
  }

  void setTorque(uint8_t id, bool enabled) {
    writeByte(id, STS_TORQUE_ENABLE_ADDR, enabled ? 1 : 0);
  }

  void setTorqueLimit(uint8_t id, uint16_t limit) {
    writeWord(id, STS_TORQUE_LIMIT_ADDR, constrain(limit, 0, STS_TORQUE_LIMIT_MAX));
  }

  void writeMotorSpeed(uint8_t id, int16_t speed, uint8_t accel) {
    int16_t limited = constrain(speed, -4095, 4095);
    uint16_t encoded = abs(limited);
    if (limited < 0) encoded |= (1 << 15);
    writeByte(id, STS_ACC_ADDR, accel);
    writeWord(id, STS_GOAL_SPEED_ADDR, encoded);
  }

  ServoIdChangeResult changeId(uint8_t currentId, uint8_t nextId) {
    ServoIdChangeResult result;
    result.currentId = currentId;
    result.nextId = nextId;
    if (currentId == 0 || nextId == 0 || currentId > 253 || nextId > 253 || currentId == nextId) return result;

    result.valid = true;
    result.unlockAck = writeByteAck(currentId, STS_LOCK_ADDR, 0);
    delay(30);

    result.idWriteAck = writeByteAck(currentId, STS_ID_ADDR, nextId, 100);
    delay(120);

    result.lockNewAck = writeByteAck(nextId, STS_LOCK_ADDR, 1, 100);
    delay(80);

    result.pingNext = ping(nextId);
    delay(20);
    result.pingCurrent = ping(currentId);

    if (!result.lockNewAck && !result.pingNext) {
      result.lockFallbackAck = writeByteAck(currentId, STS_LOCK_ADDR, 1);
    }

    result.ok = result.unlockAck && result.idWriteAck && result.lockNewAck && result.pingNext;
    return result;
  }

private:
  HardwareSerial *bus = nullptr;

  void clearRx() {
    if (!bus) return;
    while (bus->available()) bus->read();
  }

  void writePacket(uint8_t id, uint8_t instruction, const uint8_t *params, uint8_t paramLen) {
    if (!bus) return;

    uint8_t length = paramLen + 2;
    uint16_t sum = id + length + instruction;
    bus->write(0xFF);
    bus->write(0xFF);
    bus->write(id);
    bus->write(length);
    bus->write(instruction);

    for (uint8_t i = 0; i < paramLen; i++) {
      bus->write(params[i]);
      sum += params[i];
    }

    bus->write(static_cast<uint8_t>(~sum));
    bus->flush();
  }

  bool readResponse(uint8_t id, uint8_t *buffer, uint8_t expectedLen, uint16_t timeoutMs) {
    uint8_t index = 0;
    uint32_t startedAt = millis();

    while (millis() - startedAt < timeoutMs) {
      if (!bus->available()) {
        delay(1);
        continue;
      }

      uint8_t value = bus->read();
      if (index == 0 && value != 0xFF) continue;
      if (index == 1 && value != 0xFF) {
        index = 0;
        continue;
      }

      buffer[index++] = value;
      if (index < expectedLen) continue;

      if (buffer[2] != id || buffer[3] != 4 || buffer[4] != 0) return false;

      uint16_t sum = 0;
      for (uint8_t i = 2; i < expectedLen - 1; i++) sum += buffer[i];
      return static_cast<uint8_t>(~sum) == buffer[expectedLen - 1];
    }

    return false;
  }

  bool readStatus(uint8_t id, uint8_t *buffer, uint8_t expectedLen, uint16_t timeoutMs) {
    uint8_t index = 0;
    uint32_t startedAt = millis();

    while (millis() - startedAt < timeoutMs) {
      if (!bus->available()) {
        delay(1);
        continue;
      }

      uint8_t value = bus->read();
      if (index == 0 && value != 0xFF) continue;
      if (index == 1 && value != 0xFF) {
        index = 0;
        continue;
      }

      buffer[index++] = value;
      if (index < expectedLen) continue;

      if (buffer[2] != id || buffer[3] != 2 || buffer[4] != 0) return false;
      uint16_t sum = 0;
      for (uint8_t i = 2; i < expectedLen - 1; i++) sum += buffer[i];
      return static_cast<uint8_t>(~sum) == buffer[expectedLen - 1];
    }

    return false;
  }
};

Preferences prefs;
ServoBusDriver servoBus;
QMI8658 imuAccelGyro;
AK09918 imuMag;
WebServer webServer(80);
ServoConfig servos[MAX_SERVOS];
uint8_t servoCount = 0;
ProgramStep programSteps[MAX_PROGRAM_STEPS];
ImuState imuState;
MotorChannel motors[] = {
  {"A", MOTOR_A_PWM_PIN, MOTOR_A_IN1_PIN, MOTOR_A_IN2_PIN, MOTOR_A_PWM_CHANNEL, 0, MOTOR_DEFAULT_LIMIT},
  {"B", MOTOR_B_PWM_PIN, MOTOR_B_IN1_PIN, MOTOR_B_IN2_PIN, MOTOR_B_PWM_CHANNEL, 0, MOTOR_DEFAULT_LIMIT}
};
static constexpr uint8_t MOTOR_COUNT = sizeof(motors) / sizeof(motors[0]);
ServoIdentifyJob identifyJob;
uint8_t programStepCount = 0;
uint8_t currentStep = 0;
bool programPlaying = false;
bool programLoop = false;
uint32_t nextProgramAt = 0;
uint32_t lastStateAt = 0;
String inputLine;
NetworkMessage networkMessages[NET_MESSAGE_LOG_SIZE];
uint32_t networkMessageSeq = 0;
uint8_t networkMessageSlot = 0;

ServoConfig *findServo(uint8_t id) {
  for (uint8_t i = 0; i < servoCount; i++) {
    if (servos[i].id == id) return &servos[i];
  }
  return nullptr;
}

float clampAngle(const ServoConfig &servo, float angle) {
  return constrain(angle, servo.minAngle, servo.maxAngle);
}

uint16_t angleToBusPosition(const ServoConfig &servo, float angle) {
  float clamped = clampAngle(servo, angle);
  float effective = servo.invert ? (servo.maxAngle - (clamped - servo.minAngle)) : clamped;
  return static_cast<uint16_t>(roundf(constrain(effective, 0, STS_FULL_SCALE_DEGREES) * 4095.0f / STS_FULL_SCALE_DEGREES));
}

uint16_t rawAngleToBusPosition(float angle) {
  return static_cast<uint16_t>(roundf(constrain(angle, 0, STS_FULL_SCALE_DEGREES) * 4095.0f / STS_FULL_SCALE_DEGREES));
}

float busPositionToAngle(const ServoConfig &servo, uint16_t position) {
  float effective = constrain(position, 0, 4095) * STS_FULL_SCALE_DEGREES / 4095.0f;
  float angle = servo.invert ? (servo.minAngle + (servo.maxAngle - effective)) : effective;
  return clampAngle(servo, angle);
}

float busPositionToMeasuredAngle(const ServoConfig &servo, uint16_t position) {
  float effective = constrain(position, 0, 4095) * STS_FULL_SCALE_DEGREES / 4095.0f;
  return servo.invert ? (servo.minAngle + (servo.maxAngle - effective)) : effective;
}

float busPositionToSetupAngle(const ServoConfig &servo, uint16_t position) {
  float angle = constrain(position, 0, 4095) * STS_FULL_SCALE_DEGREES / 4095.0f;
  return servo.invert ? (STS_FULL_SCALE_DEGREES - angle) : angle;
}

uint16_t setupAngleToBusPosition(const ServoConfig &servo, float angle) {
  float limited = constrain(angle, 0, STS_FULL_SCALE_DEGREES);
  float effective = servo.invert ? (STS_FULL_SCALE_DEGREES - limited) : limited;
  return static_cast<uint16_t>(roundf(constrain(effective, 0, STS_FULL_SCALE_DEGREES) * 4095.0f / STS_FULL_SCALE_DEGREES));
}

float rawBusPositionToAngle(uint16_t position) {
  return constrain(position, 0, 4095) * STS_FULL_SCALE_DEGREES / 4095.0f;
}

uint16_t velocityToBusSpeed(float velocityDps) {
  float magnitude = fabsf(velocityDps);
  if (magnitude < 0.1f) return 1;
  return constrain(static_cast<uint16_t>(roundf(magnitude * 4095.0f / STS_FULL_SCALE_DEGREES)), 1, 4095);
}

void rememberNetworkMessage(const String &line) {
  if (line.length() == 0) return;

  networkMessageSeq++;
  networkMessages[networkMessageSlot].seq = networkMessageSeq;
  networkMessages[networkMessageSlot].line = line;
  networkMessageSlot = (networkMessageSlot + 1) % NET_MESSAGE_LOG_SIZE;
}

void sendJson(JsonDocument &doc) {
  String line;
  serializeJson(doc, line);
  Serial.println(line);
  rememberNetworkMessage(line);
}

void sendError(const char *message) {
  StaticJsonDocument<192> doc;
  doc["type"] = "error";
  doc["message"] = message;
  sendJson(doc);
}

void sendOk(const char *cmd) {
  StaticJsonDocument<160> doc;
  doc["type"] = "ok";
  doc["cmd"] = cmd;
  sendJson(doc);
}

void addConfigToJson(JsonDocument &doc) {
  JsonArray arr = doc["servos"].to<JsonArray>();
  for (uint8_t i = 0; i < servoCount; i++) {
    JsonObject item = arr.add<JsonObject>();
    item["id"] = servos[i].id;
    item["name"] = servos[i].name;
    item["min"] = servos[i].minAngle;
    item["max"] = servos[i].maxAngle;
    item["home"] = servos[i].homeAngle;
    item["invert"] = servos[i].invert;
    item["enabled"] = servos[i].enabled;
    item["last"] = servos[i].lastAngle;
    if (servos[i].hasMeasuredAngle) item["measured"] = servos[i].measuredAngle;
    item["monitor"] = servos[i].monitorEnabled;
    item["monitorInterval"] = servos[i].monitorIntervalMs;
    item["velocity"] = servos[i].velocityDps;
    item["motorMode"] = servos[i].motorMode;
    item["motorSpeed"] = servos[i].motorSpeed;
  }
}

void sendConfig() {
  DynamicJsonDocument doc(4096);
  doc["type"] = "config";
  addConfigToJson(doc);
  sendJson(doc);
}

void sendState(bool setupRange = false) {
  DynamicJsonDocument doc(2048);
  doc["type"] = "state";
  if (setupRange) doc["setup"] = true;
  doc["playing"] = programPlaying;
  doc["wifi"] = WiFi.isConnected();
  doc["ip"] = WiFi.isConnected() ? WiFi.localIP().toString() : "";
  JsonObject positions = doc["positions"].to<JsonObject>();
  JsonObject measured = doc["measured"].to<JsonObject>();
    JsonObject velocities = doc["velocities"].to<JsonObject>();
    JsonObject motorModes = doc["motorModes"].to<JsonObject>();
    JsonObject motorSpeeds = doc["motorSpeeds"].to<JsonObject>();
    JsonObject torqueLimits = doc["torqueLimits"].to<JsonObject>();
    for (uint8_t i = 0; i < servoCount; i++) {
      positions[String(servos[i].id)] = servos[i].lastAngle;
      velocities[String(servos[i].id)] = servos[i].velocityDps;
      motorModes[String(servos[i].id)] = servos[i].motorMode;
      motorSpeeds[String(servos[i].id)] = servos[i].motorSpeed;
      torqueLimits[String(servos[i].id)] = servos[i].torqueLimit;
      if (servos[i].hasMeasuredAngle) measured[String(servos[i].id)] = servos[i].measuredAngle;
    }
  sendJson(doc);
}

void addImuToJson(JsonDocument &doc) {
  doc["available"] = imuState.available;
  doc["magAvailable"] = imuState.magAvailable;
  doc["monitor"] = imuState.monitorEnabled;
  doc["interval"] = imuState.intervalMs;
  doc["sample_ms"] = imuState.sampleMs;
  doc["roll"] = imuState.roll;
  doc["pitch"] = imuState.pitch;
  doc["yaw"] = imuState.yaw;

  JsonObject accel = doc["accel"].to<JsonObject>();
  accel["x"] = imuState.accel[0];
  accel["y"] = imuState.accel[1];
  accel["z"] = imuState.accel[2];

  JsonObject gyro = doc["gyro"].to<JsonObject>();
  gyro["x"] = imuState.gyro[0];
  gyro["y"] = imuState.gyro[1];
  gyro["z"] = imuState.gyro[2];

  JsonObject mag = doc["mag"].to<JsonObject>();
  mag["x"] = imuState.mag[0];
  mag["y"] = imuState.mag[1];
  mag["z"] = imuState.mag[2];
}

void sendImuState() {
  DynamicJsonDocument doc(1024);
  doc["type"] = "imu";
  addImuToJson(doc);
  sendJson(doc);
}

void sendMotorState() {
  DynamicJsonDocument doc(512);
  doc["type"] = "motors";
  JsonArray arr = doc["motors"].to<JsonArray>();
  for (uint8_t i = 0; i < MOTOR_COUNT; i++) {
    JsonObject motor = arr.add<JsonObject>();
    motor["id"] = motors[i].name;
    motor["speed"] = motors[i].command;
    motor["limit"] = motors[i].limit;
  }
  sendJson(doc);
}

void sendServoIdResult(const char *type, uint8_t currentId, uint8_t nextId, bool ok) {
  StaticJsonDocument<192> doc;
  doc["type"] = type;
  doc["current"] = currentId;
  doc["next"] = nextId;
  doc["ok"] = ok;
  sendJson(doc);
}

void sendServoIdChangeResult(const ServoIdChangeResult &result) {
  StaticJsonDocument<384> doc;
  doc["type"] = "servo_set_id";
  doc["current"] = result.currentId;
  doc["next"] = result.nextId;
  doc["ok"] = result.ok;
  doc["unlock"] = result.unlockAck;
  doc["write"] = result.idWriteAck;
  doc["lock"] = result.lockNewAck;
  doc["pingNext"] = result.pingNext;
  doc["pingCurrent"] = result.pingCurrent;
  doc["fallbackLock"] = result.lockFallbackAck;

  const char *stage = "ok";
  if (!result.unlockAck) stage = "unlock";
  else if (!result.idWriteAck) stage = "write_id";
  else if (!result.lockNewAck) stage = "lock_new_id";
  else if (!result.pingNext) stage = "ping_new_id";
  doc["stage"] = stage;

  sendJson(doc);
}

void sendWifiStatus() {
  StaticJsonDocument<256> doc;
  doc["type"] = "wifi";
  doc["connected"] = WiFi.isConnected();
  doc["ssid"] = WiFi.SSID();
  doc["ip"] = WiFi.isConnected() ? WiFi.localIP().toString() : "";
  doc["rssi"] = WiFi.isConnected() ? WiFi.RSSI() : 0;
  sendJson(doc);
}

void setDefaultConfig() {
  servoCount = MAX_SERVOS;

  for (uint8_t i = 0; i < MAX_SERVOS; i++) {
    servos[i].id = DEFAULT_SERVO_DEFS[i].id;
    servos[i].name = DEFAULT_SERVO_DEFS[i].name;
    servos[i].minAngle = 0;
    servos[i].maxAngle = STS_FULL_SCALE_DEGREES;
    servos[i].homeAngle = 180;
    servos[i].invert = false;
    servos[i].enabled = true;
    servos[i].lastAngle = 180;
    servos[i].measuredAngle = 180;
    servos[i].hasMeasuredAngle = false;
    servos[i].monitorEnabled = false;
    servos[i].monitorIntervalMs = DEFAULT_MONITOR_INTERVAL_MS;
    servos[i].nextMonitorAt = 0;
    servos[i].velocityDps = 0;
    servos[i].lastVelocityAt = 0;
    servos[i].motorMode = false;
    servos[i].motorSpeed = 0;
    servos[i].torqueLimit = STS_TORQUE_LIMIT_MAX;
  }
}

void saveConfig() {
  DynamicJsonDocument doc(4096);
  addConfigToJson(doc);
  String json;
  serializeJson(doc, json);
  prefs.putString("config", json);
  prefs.putUChar("configVersion", CONFIG_VERSION);
}

void loadConfig() {
  prefs.begin("robotdog", false);
  String json = prefs.getString("config", "");
  uint8_t configVersion = prefs.getUChar("configVersion", 0);
  if (json.length() == 0 || configVersion < CONFIG_VERSION) {
    setDefaultConfig();
    saveConfig();
    return;
  }

  DynamicJsonDocument doc(4096);
  DeserializationError err = deserializeJson(doc, json);
  if (err || !doc["servos"].is<JsonArray>()) {
    setDefaultConfig();
    saveConfig();
    return;
  }

  servoCount = 0;
  for (JsonObject item : doc["servos"].as<JsonArray>()) {
    if (servoCount >= MAX_SERVOS) break;
    servos[servoCount].id = item["id"] | 0;
    servos[servoCount].name = String(item["name"] | "servo");
    servos[servoCount].minAngle = item["min"] | 0.0f;
    servos[servoCount].maxAngle = item["max"] | STS_FULL_SCALE_DEGREES;
    servos[servoCount].homeAngle = item["home"] | 180.0f;
    servos[servoCount].invert = item["invert"] | false;
    servos[servoCount].enabled = item["enabled"] | true;
    servos[servoCount].lastAngle = servos[servoCount].homeAngle;
    servos[servoCount].measuredAngle = servos[servoCount].homeAngle;
    servos[servoCount].hasMeasuredAngle = false;
    servos[servoCount].monitorEnabled = item["monitor"] | true;
    servos[servoCount].monitorIntervalMs = constrain(item["monitorInterval"] | DEFAULT_MONITOR_INTERVAL_MS, 100, 5000);
    servos[servoCount].nextMonitorAt = 0;
    servos[servoCount].velocityDps = 0;
    servos[servoCount].lastVelocityAt = 0;
    servos[servoCount].motorMode = false;
    servos[servoCount].motorSpeed = 0;
    if (servos[servoCount].id > 0) servoCount++;
  }

  if (servoCount == 0) {
    setDefaultConfig();
    saveConfig();
  }
}

void writeServoTarget(ServoConfig &servo, float angle, uint16_t speed, uint8_t accel) {
  float clamped = clampAngle(servo, angle);
  servo.lastAngle = clamped;
  servoBus.moveTo(servo.id, angleToBusPosition(servo, clamped), speed, accel);
}

void writeServoSetupTarget(ServoConfig &servo, float angle, uint16_t speed, uint8_t accel) {
  float savedMin = servo.minAngle;
  float savedMax = servo.maxAngle;
  servo.minAngle = 0.0f;
  servo.maxAngle = STS_FULL_SCALE_DEGREES;
  writeServoTarget(servo, constrain(angle, 0.0f, STS_FULL_SCALE_DEGREES), speed, accel);
  servo.minAngle = savedMin;
  servo.maxAngle = savedMax;
}

void setServoMode(ServoConfig &servo, bool motorMode) {
  if (motorMode) {
    servo.velocityDps = 0;
    servo.lastVelocityAt = 0;
    servoBus.setMode(servo.id, STS_MODE_MOTOR);
    servo.motorMode = true;
    return;
  }

  servoBus.writeMotorSpeed(servo.id, 0, 0);
  servoBus.setMode(servo.id, STS_MODE_SERVO);
  servo.motorMode = false;
  servo.motorSpeed = 0;
}

void ensureServoMode(ServoConfig &servo) {
  if (servo.motorMode) setServoMode(servo, false);
}

bool moveServo(uint8_t id, float angle, uint16_t speed, uint8_t accel) {
  ServoConfig *servo = findServo(id);
  if (!servo || !servo->enabled) return false;

  servo->velocityDps = 0;
  servo->lastVelocityAt = 0;
  ensureServoMode(*servo);
  writeServoTarget(*servo, angle, speed, accel);
  return true;
}

bool readServoAngle(uint8_t id, bool setupRange = false) {
  ServoConfig *servo = findServo(id);
  if (!servo || !servo->enabled) return false;

  uint16_t position = 0;
  if (!servoBus.readPosition(servo->id, position)) return false;

  servo->measuredAngle = setupRange ? busPositionToSetupAngle(*servo, position) : busPositionToMeasuredAngle(*servo, position);
  servo->hasMeasuredAngle = true;
  return true;
}

void readAllServoAngles(bool setupRange = false) {
  for (uint8_t i = 0; i < servoCount; i++) {
    if (servos[i].enabled) readServoAngle(servos[i].id, setupRange);
  }
}

void stopAllServoVelocities() {
  for (uint8_t i = 0; i < servoCount; i++) {
    servos[i].velocityDps = 0;
    servos[i].lastVelocityAt = 0;
  }
}

void stopServoMotor(ServoConfig &servo) {
  servoBus.writeMotorSpeed(servo.id, 0, 0);
  servo.motorSpeed = 0;
}

void stopAllServoMotors() {
  for (uint8_t i = 0; i < servoCount; i++) {
    if (servos[i].motorMode || fabsf(servos[i].motorSpeed) > 0.001f) stopServoMotor(servos[i]);
  }
}

bool setServoTorque(uint8_t id, bool enabled) {
  ServoConfig *servo = findServo(id);
  if (!servo || !servo->enabled) return false;

  programPlaying = false;
  servo->velocityDps = 0;
  servo->lastVelocityAt = 0;
  stopServoMotor(*servo);

  uint16_t presentPosition = 0;
  bool hasPosition = servoBus.readPosition(servo->id, presentPosition);
  if (hasPosition) {
    servo->measuredAngle = busPositionToMeasuredAngle(*servo, presentPosition);
    servo->hasMeasuredAngle = true;
    servo->lastAngle = clampAngle(*servo, servo->measuredAngle);
  }

  if (enabled) {
    setServoMode(*servo, false);
    if (hasPosition) {
      servoBus.moveTo(servo->id, constrain(presentPosition, 0, 4095), 1, 0);
    } else {
      writeServoTarget(*servo, servo->lastAngle, 1, 0);
    }
  }

  servoBus.setTorque(servo->id, enabled);
  return true;
}

bool setServoTorqueLimit(uint8_t id, uint16_t limit) {
  ServoConfig *servo = findServo(id);
  if (!servo || !servo->enabled) return false;

  servo->torqueLimit = constrain(limit, 0, STS_TORQUE_LIMIT_MAX);
  servoBus.setTorqueLimit(servo->id, servo->torqueLimit);
  return true;
}

bool jogServo(uint8_t id, float delta, uint16_t speed, uint8_t accel, bool setupRange, bool hasBaseAngle = false, float requestedBaseAngle = 0.0f) {
  ServoConfig *servo = findServo(id);
  if (!servo || !servo->enabled) return false;

  programPlaying = false;
  servo->velocityDps = 0;
  servo->lastVelocityAt = 0;
  stopServoMotor(*servo);

  uint16_t position = 0;
  float baseAngle = hasBaseAngle
    ? constrain(requestedBaseAngle, 0.0f, STS_FULL_SCALE_DEGREES)
    : servo->lastAngle;
  if (!hasBaseAngle && servoBus.readPosition(servo->id, position)) {
    servo->measuredAngle = setupRange ? busPositionToSetupAngle(*servo, position) : busPositionToMeasuredAngle(*servo, position);
    servo->hasMeasuredAngle = true;
    baseAngle = servo->measuredAngle;
  }

  setServoMode(*servo, false);
  servoBus.setTorque(servo->id, true);
  if (setupRange) {
    float target = constrain(baseAngle + delta, 0, STS_FULL_SCALE_DEGREES);
    writeServoSetupTarget(*servo, target, speed, accel);
    servo->measuredAngle = target;
    servo->hasMeasuredAngle = true;
  } else {
    writeServoTarget(*servo, baseAngle + delta, speed, accel);
  }
  return true;
}

bool setServoMotorSpeed(uint8_t id, float speed, uint8_t accel) {
  ServoConfig *servo = findServo(id);
  if (!servo || !servo->enabled) return false;

  setServoMode(*servo, true);
  float limited = constrain(speed, -1.0f, 1.0f);
  servo->motorSpeed = limited;
  int16_t rawSpeed = static_cast<int16_t>(roundf(limited * 4095.0f));
  servoBus.writeMotorSpeed(servo->id, rawSpeed, accel);
  programPlaying = false;
  return true;
}

bool setServoVelocity(uint8_t id, float velocityDps) {
  ServoConfig *servo = findServo(id);
  if (!servo || !servo->enabled) return false;

  ensureServoMode(*servo);
  servo->velocityDps = constrain(velocityDps, -720.0f, 720.0f);
  servo->lastVelocityAt = millis();
  programPlaying = false;
  return true;
}

void runServoVelocities() {
  uint32_t now = millis();
  for (uint8_t i = 0; i < servoCount; i++) {
    ServoConfig &servo = servos[i];
    if (!servo.enabled || fabsf(servo.velocityDps) < 0.1f) continue;
    if (servo.lastVelocityAt == 0) servo.lastVelocityAt = now;
    if (now - servo.lastVelocityAt < SERVO_VELOCITY_INTERVAL_MS) continue;

    float dt = (now - servo.lastVelocityAt) / 1000.0f;
    servo.lastVelocityAt = now;

    float nextAngle = clampAngle(servo, servo.lastAngle + servo.velocityDps * dt);
    if ((servo.velocityDps > 0 && nextAngle >= servo.maxAngle) || (servo.velocityDps < 0 && nextAngle <= servo.minAngle)) {
      servo.velocityDps = 0;
    }

    writeServoTarget(servo, nextAngle, velocityToBusSpeed(servo.velocityDps), 30);
  }
}

void runServoMonitors() {
  if (programPlaying) return;

  uint32_t now = millis();
  for (uint8_t i = 0; i < servoCount; i++) {
    if (!servos[i].enabled || !servos[i].monitorEnabled) continue;
    if (now < servos[i].nextMonitorAt) continue;

    readServoAngle(servos[i].id);
    servos[i].nextMonitorAt = now + servos[i].monitorIntervalMs;
  }
}

void setServoMonitor(ServoConfig &servo, bool enabled, uint16_t intervalMs) {
  servo.monitorEnabled = enabled;
  servo.monitorIntervalMs = constrain(intervalMs, 100, 5000);
  servo.nextMonitorAt = 0;
}

float readServoCenterForIdentify(uint8_t id) {
  ServoConfig *servo = findServo(id);
  uint16_t position = 0;
  if (servo && servo->enabled && servoBus.readPosition(id, position)) {
    servo->measuredAngle = busPositionToMeasuredAngle(*servo, position);
    servo->hasMeasuredAngle = true;
    servo->lastAngle = servo->measuredAngle;
    return servo->measuredAngle;
  }

  if (servo) return servo->lastAngle;
  if (servoBus.readPosition(id, position)) return rawBusPositionToAngle(position);
  return 180.0f;
}

void writeIdentifyTarget(uint8_t id, float angle, uint16_t speed, uint8_t accel) {
  ServoConfig *servo = findServo(id);
  if (servo) {
    servo->velocityDps = 0;
    servo->lastVelocityAt = 0;
    ensureServoMode(*servo);
    writeServoTarget(*servo, angle, speed, accel);
    return;
  }

  servoBus.moveTo(id, rawAngleToBusPosition(angle), speed, accel);
}

bool startServoIdentify(uint8_t id, float amplitude, uint16_t speed, uint8_t accel) {
  if (id == 0 || !servoBus.ping(id)) return false;

  ServoConfig *servo = findServo(id);
  if (servo) {
    servo->velocityDps = 0;
    servo->lastVelocityAt = 0;
  }

  float center = readServoCenterForIdentify(id);
  float minAngle = servo ? servo->minAngle : 0.0f;
  float maxAngle = servo ? servo->maxAngle : STS_FULL_SCALE_DEGREES;
  float safeAmplitude = constrain(fabsf(amplitude), 4.0f, 45.0f);
  float span = maxAngle - minAngle;
  if (span < safeAmplitude * 2.0f) {
    safeAmplitude = max(1.0f, span * 0.25f);
  }
  center = constrain(center, minAngle + safeAmplitude, maxAngle - safeAmplitude);

  identifyJob.active = true;
  identifyJob.id = id;
  identifyJob.centerAngle = center;
  identifyJob.amplitude = safeAmplitude;
  identifyJob.phase = 0;
  identifyJob.speed = constrain(speed, 1, 4095);
  identifyJob.accel = constrain(accel, 0, 254);
  identifyJob.nextAt = 0;
  return true;
}

void runServoIdentify() {
  if (!identifyJob.active || millis() < identifyJob.nextAt) return;

  float target = identifyJob.centerAngle;
  if (identifyJob.phase == 0 || identifyJob.phase == 2) {
    target = identifyJob.centerAngle + identifyJob.amplitude;
  } else if (identifyJob.phase == 1) {
    target = identifyJob.centerAngle - identifyJob.amplitude;
  }

  writeIdentifyTarget(identifyJob.id, target, identifyJob.speed, identifyJob.accel);
  identifyJob.phase++;
  if (identifyJob.phase >= 4) {
    identifyJob.active = false;
    return;
  }

  identifyJob.nextAt = millis() + 350;
}

MotorChannel *findMotor(const char *id) {
  if (!id || strlen(id) == 0) return nullptr;
  for (uint8_t i = 0; i < MOTOR_COUNT; i++) {
    if (strcasecmp(id, motors[i].name) == 0) return &motors[i];
  }
  return nullptr;
}

void applyMotor(MotorChannel &motor, float speed) {
  float limited = constrain(speed, -motor.limit, motor.limit);
  motor.command = limited;

  if (limited > 0.001f) {
    digitalWrite(motor.in1Pin, HIGH);
    digitalWrite(motor.in2Pin, LOW);
  } else if (limited < -0.001f) {
    digitalWrite(motor.in1Pin, LOW);
    digitalWrite(motor.in2Pin, HIGH);
  } else {
    digitalWrite(motor.in1Pin, LOW);
    digitalWrite(motor.in2Pin, LOW);
    limited = 0;
    motor.command = 0;
  }

  ledcWrite(motor.pwmChannel, static_cast<uint32_t>(roundf(abs(limited) * 255.0f)));
}

void stopAllMotors() {
  for (uint8_t i = 0; i < MOTOR_COUNT; i++) {
    applyMotor(motors[i], 0);
  }
}

void setupMotors() {
  for (uint8_t i = 0; i < MOTOR_COUNT; i++) {
    pinMode(motors[i].in1Pin, OUTPUT);
    pinMode(motors[i].in2Pin, OUTPUT);
    ledcSetup(motors[i].pwmChannel, MOTOR_PWM_FREQ, MOTOR_PWM_RESOLUTION);
    ledcAttachPin(motors[i].pwmPin, motors[i].pwmChannel);
    applyMotor(motors[i], 0);
  }
}

bool sampleImu() {
  if (!imuState.available) return false;

  float accel[3] = {0, 0, 0};
  float gyro[3] = {0, 0, 0};
  imuAccelGyro.read_sensor_data(accel, gyro);

  for (uint8_t i = 0; i < 3; i++) {
    imuState.accel[i] = accel[i];
    imuState.gyro[i] = gyro[i];
  }

  imuState.roll = atan2f(accel[1], accel[2]) * 57.2957795f;
  imuState.pitch = atan2f(-accel[0], sqrtf(accel[1] * accel[1] + accel[2] * accel[2])) * 57.2957795f;

  if (imuState.magAvailable && imuMag.isDataReady() == AK09918_ERR_OK) {
    int16_t mx = 0;
    int16_t my = 0;
    int16_t mz = 0;
    if (imuMag.getData(&mx, &my, &mz) == AK09918_ERR_OK) {
      imuState.mag[0] = mx;
      imuState.mag[1] = my;
      imuState.mag[2] = mz;
      imuState.yaw = atan2f(static_cast<float>(my), static_cast<float>(mx)) * 57.2957795f;
    }
  }

  imuState.sampleMs = millis();
  return true;
}

void runImuMonitor() {
  if (!imuState.monitorEnabled || millis() < imuState.nextSampleAt) return;
  imuState.nextSampleAt = millis() + imuState.intervalMs;

  if (sampleImu()) {
    sendImuState();
  }
}

void setupImu() {
  Wire.begin(IMU_SDA_PIN, IMU_SCL_PIN);
  Wire.setClock(400000);

  imuState.available = imuAccelGyro.begin() == 1;
  imuState.magAvailable = imuMag.initialize(AK09918_CONTINUOUS_100HZ) == AK09918_ERR_OK;
  imuState.nextSampleAt = 0;
  if (imuState.available) sampleImu();
}

void homeAll(uint16_t speed = 700, uint8_t accel = 40) {
  for (uint8_t i = 0; i < servoCount; i++) {
    if (servos[i].enabled) moveServo(servos[i].id, servos[i].homeAngle, speed, accel);
  }
}

void handleConfigSet(JsonDocument &doc) {
  if (!doc["servos"].is<JsonArray>()) {
    sendError("config_set requires a servos array");
    return;
  }

  uint8_t nextCount = 0;
  ServoConfig next[MAX_SERVOS];
  for (JsonObject item : doc["servos"].as<JsonArray>()) {
    if (nextCount >= MAX_SERVOS) break;
    uint8_t id = item["id"] | 0;
    if (id == 0) continue;

    next[nextCount].id = id;
    next[nextCount].name = String(item["name"] | "servo");
    next[nextCount].minAngle = item["min"] | 0.0f;
    next[nextCount].maxAngle = item["max"] | STS_FULL_SCALE_DEGREES;
    next[nextCount].homeAngle = item["home"] | 180.0f;
    next[nextCount].invert = item["invert"] | false;
    next[nextCount].enabled = item["enabled"] | true;
    next[nextCount].lastAngle = next[nextCount].homeAngle;
    next[nextCount].measuredAngle = next[nextCount].homeAngle;
    next[nextCount].hasMeasuredAngle = false;
    next[nextCount].monitorEnabled = item["monitor"] | true;
    next[nextCount].monitorIntervalMs = constrain(item["monitorInterval"] | DEFAULT_MONITOR_INTERVAL_MS, 100, 5000);
    next[nextCount].nextMonitorAt = 0;
    next[nextCount].velocityDps = 0;
    next[nextCount].lastVelocityAt = 0;
    next[nextCount].motorMode = false;
    next[nextCount].motorSpeed = 0;
    nextCount++;
  }

  if (nextCount == 0) {
    sendError("config must contain at least one servo");
    return;
  }

  servoCount = nextCount;
  for (uint8_t i = 0; i < servoCount; i++) servos[i] = next[i];
  saveConfig();
  sendOk("config_set");
  sendConfig();
}

void startProgram(JsonDocument &doc) {
  if (!doc["steps"].is<JsonArray>()) {
    sendError("play requires a steps array");
    return;
  }

  programStepCount = 0;
  String unusableIds;
  bool unusableIdSeen[254] = {false};
  for (JsonObject stepDoc : doc["steps"].as<JsonArray>()) {
    if (programStepCount >= MAX_PROGRAM_STEPS) break;
    ProgramStep &step = programSteps[programStepCount];
    step.durationMs = constrain(stepDoc["ms"] | 500, 20, 10000);
    step.speed = constrain(stepDoc["speed"] | 900, 1, 4095);
    step.accel = constrain(stepDoc["accel"] | 50, 0, 254);
    for (uint8_t i = 0; i < MAX_SERVOS; i++) step.hasPose[i] = false;

    JsonObject poses = stepDoc["poses"].as<JsonObject>();
    for (JsonPair pair : poses) {
      uint8_t id = atoi(pair.key().c_str());
      ServoConfig *servo = findServo(id);
      if (!servo || !servo->enabled) {
        if (id < 254 && !unusableIdSeen[id]) {
          unusableIdSeen[id] = true;
          if (unusableIds.length() > 0) unusableIds += ",";
          unusableIds += id;
        }
        continue;
      }
      uint8_t index = servo - servos;
      step.pose[index] = pair.value().as<float>();
      step.hasPose[index] = true;
    }
    programStepCount++;
  }

  if (unusableIds.length() > 0) {
    programStepCount = 0;
    String message = String("program references unknown or disabled servo id(s): ") + unusableIds;
    sendError(message.c_str());
    return;
  }

  if (programStepCount == 0) {
    sendError("program has no usable steps");
    return;
  }

  stopAllServoVelocities();
  stopAllServoMotors();
  programLoop = doc["loop"] | false;
  programPlaying = true;
  currentStep = 0;
  nextProgramAt = 0;
  sendOk("play");
}

void handleWifiSet(JsonDocument &doc) {
  const char *ssid = doc["ssid"] | "";
  const char *password = doc["password"] | "";

  if (strlen(ssid) == 0) {
    sendError("wifi_set requires ssid");
    return;
  }

  WiFi.persistent(true);
  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true);
  delay(200);
  WiFi.begin(ssid, password);

  uint32_t startedAt = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startedAt < 15000) {
    ArduinoOTA.handle();
    delay(100);
  }

  if (WiFi.isConnected()) {
    ArduinoOTA.setHostname("robot-dog-control");
    ArduinoOTA.begin();
    sendOk("wifi_set");
    sendWifiStatus();
  } else {
    sendError("wifi connection failed");
    sendWifiStatus();
  }
}

void handleMonitorSet(JsonDocument &doc) {
  bool enabled = doc["enabled"] | true;
  uint16_t intervalMs = doc["interval"] | DEFAULT_MONITOR_INTERVAL_MS;

  if (doc["all"] | false) {
    for (uint8_t i = 0; i < servoCount; i++) {
      setServoMonitor(servos[i], enabled, intervalMs);
    }
    saveConfig();
    sendOk("monitor_set");
    sendConfig();
    return;
  }

  uint8_t id = doc["id"] | 0;
  ServoConfig *servo = findServo(id);
  if (!servo) {
    sendError("unknown servo id");
    return;
  }

  setServoMonitor(*servo, enabled, intervalMs);
  saveConfig();
  sendOk("monitor_set");
  sendConfig();
}

void handleImuMonitorSet(JsonDocument &doc) {
  imuState.monitorEnabled = doc["enabled"] | true;
  imuState.intervalMs = constrain(doc["interval"] | DEFAULT_IMU_INTERVAL_MS, 50, 5000);
  imuState.nextSampleAt = 0;
  sendOk("imu_monitor_set");
  if (sampleImu()) sendImuState();
}

void handleMotorSet(JsonDocument &doc) {
  const char *id = doc["id"] | "";
  float speed = constrain(doc["speed"] | 0.0f, -1.0f, 1.0f);
  float limit = constrain(doc["limit"] | MOTOR_DEFAULT_LIMIT, 0.0f, 1.0f);

  if (doc["all"] | false) {
    for (uint8_t i = 0; i < MOTOR_COUNT; i++) {
      motors[i].limit = limit;
      applyMotor(motors[i], speed);
    }
    sendOk("motor_set");
    sendMotorState();
    return;
  }

  MotorChannel *motor = findMotor(id);
  if (!motor) {
    sendError("unknown motor id");
    return;
  }

  motor->limit = limit;
  applyMotor(*motor, speed);
  sendOk("motor_set");
  sendMotorState();
}

void handleServoPing(JsonDocument &doc) {
  uint8_t id = doc["id"] | 0;
  if (id == 0) {
    sendError("servo_ping requires id");
    return;
  }

  bool ok = servoBus.ping(id);
  sendServoIdResult("servo_ping", id, id, ok);
}

void handleServoScan(JsonDocument &doc) {
  uint8_t startId = constrain(doc["start"] | 1, 1, 253);
  uint8_t endId = constrain(doc["end"] | 20, 1, 253);
  if (endId < startId) {
    uint8_t tmp = startId;
    startId = endId;
    endId = tmp;
  }

  DynamicJsonDocument response(1024);
  response["type"] = "servo_scan";
  response["start"] = startId;
  response["end"] = endId;
  JsonArray found = response["found"].to<JsonArray>();

  for (uint8_t id = startId; id <= endId; id++) {
    if (servoBus.ping(id)) found.add(id);
    delay(5);
  }

  sendJson(response);
}

void handleServoSetId(JsonDocument &doc) {
  int currentRaw = doc["current"] | 0;
  int nextRaw = doc["next"] | 0;

  if (currentRaw < 1 || currentRaw > 253 || nextRaw < 1 || nextRaw > 253 || currentRaw == nextRaw) {
    sendError("servo_set_id requires different current and next ids from 1-253");
    return;
  }

  uint8_t currentId = static_cast<uint8_t>(currentRaw);
  uint8_t nextId = static_cast<uint8_t>(nextRaw);
  ServoIdChangeResult result = servoBus.changeId(currentId, nextId);
  if (result.ok) {
    ServoConfig *current = findServo(currentId);
    ServoConfig *existingNext = findServo(nextId);
    if (current && !existingNext) {
      current->id = nextId;
      current->name = String("Servo ") + nextId;
      saveConfig();
    }
  }

  sendServoIdChangeResult(result);
  sendConfig();
}

void handleServoVelocity(JsonDocument &doc) {
  float velocityDps = constrain(doc["dps"] | 0.0f, -720.0f, 720.0f);

  if (doc["all"] | false) {
    for (uint8_t i = 0; i < servoCount; i++) {
      setServoVelocity(servos[i].id, velocityDps);
    }
    sendOk("servo_velocity");
    sendState();
    return;
  }

  uint8_t id = doc["id"] | 0;
  if (!setServoVelocity(id, velocityDps)) {
    sendError("unknown or disabled servo id");
    return;
  }

  sendOk("servo_velocity");
  sendState();
}

void handleServoVelocityStop(JsonDocument &doc) {
  if (doc["all"] | false) {
    stopAllServoVelocities();
    sendOk("servo_velocity_stop");
    sendState();
    return;
  }

  uint8_t id = doc["id"] | 0;
  ServoConfig *servo = findServo(id);
  if (!servo) {
    sendError("unknown servo id");
    return;
  }

  servo->velocityDps = 0;
  servo->lastVelocityAt = 0;
  sendOk("servo_velocity_stop");
  sendState();
}

void handleServoMode(JsonDocument &doc) {
  const char *mode = doc["mode"] | "servo";
  bool motorMode = strcasecmp(mode, "motor") == 0;

  if (doc["all"] | false) {
    for (uint8_t i = 0; i < servoCount; i++) {
      setServoMode(servos[i], motorMode);
    }
    sendOk("servo_mode");
    sendState();
    return;
  }

  uint8_t id = doc["id"] | 0;
  ServoConfig *servo = findServo(id);
  if (!servo || !servo->enabled) {
    sendError("unknown or disabled servo id");
    return;
  }

  setServoMode(*servo, motorMode);
  sendOk("servo_mode");
  sendState();
}

void handleServoMotorSet(JsonDocument &doc) {
  float speed = constrain(doc["speed"] | 0.0f, -1.0f, 1.0f);
  uint8_t accel = constrain(doc["accel"] | 40, 0, 254);

  if (doc["all"] | false) {
    for (uint8_t i = 0; i < servoCount; i++) {
      setServoMotorSpeed(servos[i].id, speed, accel);
    }
    sendOk("servo_motor_set");
    sendState();
    return;
  }

  uint8_t id = doc["id"] | 0;
  if (!setServoMotorSpeed(id, speed, accel)) {
    sendError("unknown or disabled servo id");
    return;
  }

  sendOk("servo_motor_set");
  sendState();
}

void handleServoMotorStop(JsonDocument &doc) {
  if (doc["all"] | false) {
    stopAllServoMotors();
    sendOk("servo_motor_stop");
    sendState();
    return;
  }

  uint8_t id = doc["id"] | 0;
  ServoConfig *servo = findServo(id);
  if (!servo) {
    sendError("unknown servo id");
    return;
  }

  stopServoMotor(*servo);
  sendOk("servo_motor_stop");
  sendState();
}

void handleServoTorque(JsonDocument &doc) {
  bool enabled = doc["enabled"] | true;

  if (doc["all"] | false) {
    for (uint8_t i = 0; i < servoCount; i++) {
      setServoTorque(servos[i].id, enabled);
    }
    sendOk("servo_torque");
    sendState();
    return;
  }

  uint8_t id = doc["id"] | 0;
  if (!setServoTorque(id, enabled)) {
    sendError("unknown or disabled servo id");
    return;
  }

  sendOk("servo_torque");
  sendState();
}

void handleServoTorqueLimit(JsonDocument &doc) {
  uint16_t limit = constrain(doc["limit"] | STS_TORQUE_LIMIT_MAX, 0, STS_TORQUE_LIMIT_MAX);

  if (doc["all"] | false) {
    for (uint8_t i = 0; i < servoCount; i++) {
      setServoTorqueLimit(servos[i].id, limit);
    }
    sendOk("servo_torque_limit");
    sendState();
    return;
  }

  uint8_t id = doc["id"] | 0;
  if (!setServoTorqueLimit(id, limit)) {
    sendError("unknown or disabled servo id");
    return;
  }

  sendOk("servo_torque_limit");
  sendState();
}

void handleServoJog(JsonDocument &doc) {
  uint8_t id = doc["id"] | 0;
  float delta = constrain(doc["delta"] | 0.0f, -45.0f, 45.0f);
  uint16_t speed = constrain(doc["speed"] | 250, 1, 4095);
  uint8_t accel = constrain(doc["accel"] | 8, 0, 254);
  bool setupRange = doc["setup"] | false;
  bool hasBaseAngle = setupRange && (doc["base"].is<int>() || doc["base"].is<float>());
  float baseAngle = doc["base"] | 0.0f;

  if (!jogServo(id, delta, speed, accel, setupRange, hasBaseAngle, baseAngle)) {
    sendError("unknown or disabled servo id");
    return;
  }

  sendOk("servo_jog");
  sendState(setupRange);
}

void handleServoIdentify(JsonDocument &doc) {
  uint8_t id = doc["id"] | 0;
  float amplitude = doc["amplitude"] | 12.0f;
  uint16_t speed = constrain(doc["speed"] | 700, 1, 4095);
  uint8_t accel = constrain(doc["accel"] | 40, 0, 254);

  if (!startServoIdentify(id, amplitude, speed, accel)) {
    sendError("servo identify failed");
    return;
  }

  StaticJsonDocument<192> response;
  response["type"] = "servo_identify";
  response["id"] = id;
  response["ok"] = true;
  sendJson(response);
}

void runProgram() {
  if (!programPlaying || millis() < nextProgramAt) return;

  ProgramStep &step = programSteps[currentStep];
  for (uint8_t i = 0; i < servoCount; i++) {
    if (step.hasPose[i]) moveServo(servos[i].id, step.pose[i], step.speed, step.accel);
  }

  nextProgramAt = millis() + step.durationMs;
  currentStep++;
  if (currentStep >= programStepCount) {
    if (programLoop) {
      currentStep = 0;
    } else {
      programPlaying = false;
    }
  }
}

void handleCommand(const String &line) {
  DynamicJsonDocument doc(SERIAL_JSON_CAPACITY);
  DeserializationError err = deserializeJson(doc, line);
  if (err) {
    sendError("invalid json");
    return;
  }

  const char *cmd = doc["cmd"] | "";
  if (strcmp(cmd, "hello") == 0) {
    DynamicJsonDocument response(4096);
    response["type"] = "hello";
    response["version"] = ROBOT_DOG_VERSION;
    response["uptime_ms"] = millis();
    response["wifi"] = WiFi.isConnected();
    response["ip"] = WiFi.isConnected() ? WiFi.localIP().toString() : "";
    JsonObject imu = response["imu"].to<JsonObject>();
    imu["available"] = imuState.available;
    imu["monitor"] = imuState.monitorEnabled;
    JsonArray motorArr = response["motors"].to<JsonArray>();
    for (uint8_t i = 0; i < MOTOR_COUNT; i++) {
      JsonObject motor = motorArr.add<JsonObject>();
      motor["id"] = motors[i].name;
      motor["speed"] = motors[i].command;
      motor["limit"] = motors[i].limit;
    }
    addConfigToJson(response);
    sendJson(response);
  } else if (strcmp(cmd, "config_get") == 0) {
    sendConfig();
  } else if (strcmp(cmd, "config_set") == 0) {
    handleConfigSet(doc);
  } else if (strcmp(cmd, "move") == 0) {
    uint8_t id = doc["id"] | 0;
    float angle = doc["angle"] | 120.0f;
    uint16_t speed = constrain(doc["speed"] | 900, 1, 4095);
    uint8_t accel = constrain(doc["accel"] | 50, 0, 254);
    if (!moveServo(id, angle, speed, accel)) {
      sendError("unknown or disabled servo id");
      return;
    }
    sendOk("move");
  } else if (strcmp(cmd, "read") == 0) {
    bool setupRange = doc["setup"] | false;
    if (doc["all"] | false) {
      readAllServoAngles(setupRange);
      sendState(setupRange);
    } else {
      uint8_t id = doc["id"] | 0;
      if (!readServoAngle(id, setupRange)) {
        sendError("servo read failed");
        return;
      }
      sendState(setupRange);
    }
  } else if (strcmp(cmd, "home") == 0) {
    if (doc["all"] | false) {
      homeAll();
    } else {
      uint8_t id = doc["id"] | 0;
      ServoConfig *servo = findServo(id);
      if (!servo) {
        sendError("unknown servo id");
        return;
      }
      moveServo(id, servo->homeAngle, 700, 40);
    }
    sendOk("home");
  } else if (strcmp(cmd, "stop") == 0) {
    programPlaying = false;
    stopAllServoVelocities();
    stopAllServoMotors();
    sendOk("stop");
  } else if (strcmp(cmd, "play") == 0) {
    startProgram(doc);
  } else if (strcmp(cmd, "wifi_set") == 0) {
    handleWifiSet(doc);
  } else if (strcmp(cmd, "wifi_status") == 0) {
    sendWifiStatus();
  } else if (strcmp(cmd, "monitor_set") == 0) {
    handleMonitorSet(doc);
  } else if (strcmp(cmd, "imu_monitor_set") == 0) {
    handleImuMonitorSet(doc);
  } else if (strcmp(cmd, "imu_status") == 0) {
    if (sampleImu()) sendImuState();
    else sendImuState();
  } else if (strcmp(cmd, "motor_set") == 0) {
    handleMotorSet(doc);
  } else if (strcmp(cmd, "motor_stop") == 0) {
    stopAllMotors();
    sendOk("motor_stop");
    sendMotorState();
  } else if (strcmp(cmd, "servo_ping") == 0) {
    handleServoPing(doc);
  } else if (strcmp(cmd, "servo_scan") == 0) {
    handleServoScan(doc);
  } else if (strcmp(cmd, "servo_set_id") == 0) {
    handleServoSetId(doc);
  } else if (strcmp(cmd, "servo_velocity") == 0) {
    handleServoVelocity(doc);
  } else if (strcmp(cmd, "servo_velocity_stop") == 0) {
    handleServoVelocityStop(doc);
  } else if (strcmp(cmd, "servo_mode") == 0) {
    handleServoMode(doc);
  } else if (strcmp(cmd, "servo_motor_set") == 0) {
    handleServoMotorSet(doc);
  } else if (strcmp(cmd, "servo_motor_stop") == 0) {
    handleServoMotorStop(doc);
  } else if (strcmp(cmd, "servo_torque") == 0) {
    handleServoTorque(doc);
  } else if (strcmp(cmd, "servo_torque_limit") == 0) {
    handleServoTorqueLimit(doc);
  } else if (strcmp(cmd, "servo_jog") == 0) {
    handleServoJog(doc);
  } else if (strcmp(cmd, "servo_identify") == 0) {
    handleServoIdentify(doc);
  } else {
    sendError("unknown command");
  }
}

void addCorsHeaders() {
  webServer.sendHeader("Access-Control-Allow-Origin", "*");
  webServer.sendHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  webServer.sendHeader("Access-Control-Allow-Headers", "Content-Type");
  webServer.sendHeader("Cache-Control", "no-store");
}

void sendCorsOk() {
  addCorsHeaders();
  webServer.send(204);
}

const String *findNetworkMessage(uint32_t seq) {
  for (uint8_t i = 0; i < NET_MESSAGE_LOG_SIZE; i++) {
    if (networkMessages[i].seq == seq) return &networkMessages[i].line;
  }
  return nullptr;
}

String escapeJsonString(const String &value) {
  String escaped;
  escaped.reserve(value.length() + 8);

  for (uint16_t i = 0; i < value.length(); i++) {
    char c = value[i];
    if (c == '\\' || c == '"') {
      escaped += '\\';
      escaped += c;
    } else if (c == '\n') {
      escaped += "\\n";
    } else if (c == '\r') {
      escaped += "\\r";
    } else if (c == '\t') {
      escaped += "\\t";
    } else if (static_cast<uint8_t>(c) < 0x20) {
      char buffer[7];
      snprintf(buffer, sizeof(buffer), "\\u%04x", c);
      escaped += buffer;
    } else {
      escaped += c;
    }
  }

  return escaped;
}

void handleApiMessages() {
  addCorsHeaders();
  uint32_t after = 0;
  if (webServer.hasArg("after")) {
    after = strtoul(webServer.arg("after").c_str(), nullptr, 10);
  }

  uint32_t firstAvailable = networkMessageSeq >= NET_MESSAGE_LOG_SIZE
    ? networkMessageSeq - NET_MESSAGE_LOG_SIZE + 1
    : 1;
  uint32_t startSeq = after < firstAvailable ? firstAvailable : after + 1;

  String body;
  body.reserve(4096);
  body += "{\"seq\":";
  body += String(networkMessageSeq);
  body += ",\"lines\":[";

  bool first = true;
  for (uint32_t seq = startSeq; seq <= networkMessageSeq; seq++) {
    const String *line = findNetworkMessage(seq);
    if (!line) continue;
    if (!first) body += ",";
    body += "\"";
    body += escapeJsonString(*line);
    body += "\"";
    first = false;
  }

  body += "]}";
  webServer.send(200, "application/json", body);
}

void handleApiCommand() {
  addCorsHeaders();
  String body = webServer.arg("plain");
  if (body.length() == 0) {
    webServer.send(400, "application/json", "{\"ok\":false,\"error\":\"missing command body\"}");
    return;
  }

  handleCommand(body);

  String response = "{\"ok\":true,\"seq\":";
  response += String(networkMessageSeq);
  response += "}";
  webServer.send(200, "application/json", response);
}

void handleWebIndex() {
  if (SPIFFS.exists("/index.html")) {
    File file = SPIFFS.open("/index.html", "r");
    webServer.streamFile(file, "text/html");
    file.close();
    return;
  }

  webServer.send(
    200,
    "text/html",
    "<!doctype html><html><head><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
    "<title>Robot Dog</title></head><body><h1>Robot Dog</h1>"
    "<p>The board is online, but the web UI filesystem has not been uploaded yet.</p>"
    "<p>Run <code>python -m platformio run -e esp32dev_ota -t uploadfs</code>.</p>"
    "</body></html>"
  );
}

void setupWebServer() {
  if (!SPIFFS.begin(true)) {
    Serial.println("{\"type\":\"error\",\"message\":\"spiffs mount failed\"}");
  }

  webServer.on("/", HTTP_GET, handleWebIndex);
  webServer.on("/index.html", HTTP_GET, handleWebIndex);
  webServer.on("/api/messages", HTTP_OPTIONS, sendCorsOk);
  webServer.on("/api/messages", HTTP_GET, handleApiMessages);
  webServer.on("/api/command", HTTP_OPTIONS, sendCorsOk);
  webServer.on("/api/command", HTTP_POST, handleApiCommand);
  webServer.onNotFound([]() {
    if (webServer.method() == HTTP_OPTIONS) {
      sendCorsOk();
      return;
    }
    handleWebIndex();
  });
  webServer.begin();
}

void pollSerial() {
  while (Serial.available()) {
    char c = static_cast<char>(Serial.read());
    if (c == '\r') continue;
    if (c == '\n') {
      if (inputLine.length() > 0) {
        handleCommand(inputLine);
        inputLine = "";
      }
    } else if (inputLine.length() < SERIAL_JSON_CAPACITY - 1) {
      inputLine += c;
    } else {
      inputLine = "";
      sendError("serial line too long");
    }
  }
}

void setupWiFiAndOta() {
  WiFi.mode(WIFI_STA);
  WiFiManager manager;
  manager.setConfigPortalTimeout(WIFI_SETUP_TIMEOUT_SECONDS);
  manager.autoConnect("RobotDog-Setup");

  ArduinoOTA.setHostname("robot-dog-control");
  ArduinoOTA.begin();
  setupWebServer();
}

void setup() {
  Serial.begin(115200);
  delay(300);
  loadConfig();
  setupMotors();
  setupImu();
  servoBus.begin();
  setupWiFiAndOta();
  homeAll();
  sendConfig();
  sendImuState();
  sendMotorState();
}

void loop() {
  ArduinoOTA.handle();
  webServer.handleClient();
  pollSerial();
  runProgram();
  runServoVelocities();
  runServoIdentify();
  runServoMonitors();
  runImuMonitor();

  if (millis() - lastStateAt > STATE_INTERVAL_MS) {
    lastStateAt = millis();
    sendState();
  }
}
