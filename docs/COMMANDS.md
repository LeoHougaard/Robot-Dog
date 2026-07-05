# Robot Dog Commands

The UI and ESP32 use the same newline-delimited JSON command objects over USB serial at `115200` baud and over the Wi-Fi HTTP bridge.

## Wi-Fi HTTP Bridge

After the board joins Wi-Fi, open its IP address in a browser, for example:

```text
http://10.1.39.32/
```

The ESP32 serves the web UI from SPIFFS and exposes these API endpoints:

`POST /api/command`

The request body is one JSON command object, without the trailing newline:

```json
{"cmd":"hello"}
```

`GET /api/messages?after=0`

Returns recent board responses as JSON strings:

```json
{"seq":12,"lines":["{\"type\":\"ok\",\"cmd\":\"hello\"}"]}
```

The standalone `ui/index.html` file can also connect over Wi-Fi by using the **Board URL** field.

## Commands

`hello`

```json
{"cmd":"hello"}
```

Returns firmware version, uptime, OTA/Wi-Fi state, and the current servo config.

`config_get`

```json
{"cmd":"config_get"}
```

Returns the saved servo list.

`config_set`

```json
{"cmd":"config_set","servos":[{"id":1,"name":"hip","min":0,"max":360,"home":180,"invert":false,"enabled":true,"monitor":true,"monitorInterval":250}]}
```

Stores up to 12 servo definitions in ESP32 preferences.

`move`

```json
{"cmd":"move","id":1,"angle":120,"speed":900,"accel":50}
```

Moves one enabled servo to an angle in degrees. The firmware clamps the command to that servo's configured min/max.

`servo_velocity`

```json
{"cmd":"servo_velocity","id":1,"dps":45}
{"cmd":"servo_velocity","id":1,"dps":-45}
{"cmd":"servo_velocity","all":true,"dps":0}
```

Drives one enabled servo by joint velocity in degrees per second. The firmware keeps the servo in position mode, steps the target angle in the background, and clamps motion to that servo's configured min/max.

`servo_velocity_stop`

```json
{"cmd":"servo_velocity_stop","id":1}
{"cmd":"servo_velocity_stop","all":true}
```

Stops active velocity control for one servo or all servos.

`servo_mode`

```json
{"cmd":"servo_mode","id":1,"mode":"servo"}
{"cmd":"servo_mode","id":1,"mode":"motor"}
```

Switches an ST3215/ST-series servo between normal position servo mode and continuous motor mode. Motor mode can spin continuously, so lift the robot or remove horns before testing.

`servo_motor_set`

```json
{"cmd":"servo_motor_set","id":1,"speed":0.25,"accel":40}
{"cmd":"servo_motor_set","id":1,"speed":-0.25,"accel":40}
```

Runs one enabled servo in real ST-series motor mode. `speed` is `-1.0` to `1.0` and maps to the servo's signed goal-speed register.

`servo_motor_stop`

```json
{"cmd":"servo_motor_stop","id":1}
{"cmd":"servo_motor_stop","all":true}
```

Stops continuous motor-mode speed for one servo or all servos.

`servo_torque`

```json
{"cmd":"servo_torque","id":1,"enabled":false}
{"cmd":"servo_torque","id":1,"enabled":true}
{"cmd":"servo_torque","all":true,"enabled":true}
```

Turns ST-series torque lock off or on for one enabled servo or all enabled servos. With torque off, the servo can be moved by hand and its reported position can still be read. When torque is turned back on, the firmware first updates the goal position to the current reported position to avoid snapping back to an older target.

`servo_torque_limit`

```json
{"cmd":"servo_torque_limit","id":1,"limit":250}
{"cmd":"servo_torque_limit","all":true,"limit":1000}
```

Sets the ST-series SRAM torque limit for one enabled servo or all enabled servos. `limit` is `0` to `1000`, where `1000` is full torque. This is useful during setup before jogging or holding a servo near a mechanical end stop.

`servo_jog`

```json
{"cmd":"servo_jog","id":1,"delta":2,"speed":250,"accel":8}
{"cmd":"servo_jog","id":1,"delta":-0.0879,"base":180,"speed":700,"accel":20,"setup":true}
```

Enables torque and nudges one enabled servo by a relative angle in degrees. For normal jogs, the firmware reads the current reported position when it can and clamps the result to the servo's configured min/max. With `setup:true`, it uses the full `0-360` calibration range instead, so you can jog to new endpoints before saving tighter limits. During endpoint setup, include `base` with the UI's current setup angle so the jog does not depend on a successful servo read. One ST3215 encoder tick is about `0.088` degrees.

`read`

```json
{"cmd":"read","id":1}
{"cmd":"read","id":1,"setup":true}
{"cmd":"read","all":true}
```

Reads the current reported angle from one servo or all enabled servos. The value is returned in the `measured` object on the next `state` response. With `setup:true`, the value is reported in the raw full `0-360` endpoint-setup range instead of the servo's saved min/max range, and that `state` response includes `"setup":true`.

`monitor_set`

```json
{"cmd":"monitor_set","id":1,"enabled":true,"interval":250}
{"cmd":"monitor_set","all":true,"enabled":false}
```

Starts or pauses continuous reported-angle polling for one servo or all servos. Each servo has its own monitor setting and interval.

`servo_ping`

```json
{"cmd":"servo_ping","id":1}
```

Checks whether a servo with that ID replies on the serial bus.

`servo_scan`

```json
{"cmd":"servo_scan","start":1,"end":8}
```

Scans a range of servo IDs and returns the IDs that reply. If multiple servos share one ID, they still appear as one replying ID.

`servo_set_id`

```json
{"cmd":"servo_set_id","current":1,"next":2}
```

Changes a servo's bus ID. If two servos currently share the same ID, disconnect all but one servo before running this command; otherwise every servo with that ID can change together.

The firmware follows Waveshare's ST-series EEPROM programming sequence: unlock EEPROM at the old ID, write register `SMS_STS_ID` / address `5`, then lock EEPROM at the new ID and ping the new ID.

Every serial bus servo on the same bus must have a unique ID before it can be controlled independently. If all connected servos share ID `1`, a command to ID `1` will move all of them.

`servo_identify`

```json
{"cmd":"servo_identify","id":2,"amplitude":12,"speed":700,"accel":40}
```

Wiggles one replying servo around its current position so you can match a bus ID to a physical joint before naming and saving it in the UI.

`imu_monitor_set`

```json
{"cmd":"imu_monitor_set","enabled":true,"interval":100}
```

Starts or pauses the onboard IMU serial monitor. The board returns `imu` messages with roll, pitch, yaw, accelerometer, gyroscope, and magnetometer values.

`imu_status`

```json
{"cmd":"imu_status"}
```

Reads and returns one IMU sample.

`motor_set`

```json
{"cmd":"motor_set","id":"A","speed":0.25,"limit":0.35}
{"cmd":"motor_set","id":"B","speed":-0.25,"limit":0.35}
{"cmd":"motor_set","all":true,"speed":0,"limit":0.35}
```

Drives one onboard TB6612 DC motor channel or both channels. `speed` is `-1.0` to `1.0`, and `limit` caps the command for safer bench testing.

`motor_stop`

```json
{"cmd":"motor_stop"}
```

Stops both onboard DC motor outputs.

`home`

```json
{"cmd":"home","id":1}
{"cmd":"home","all":true}
```

Moves one servo or all enabled servos to their configured home angle.

`stop`

```json
{"cmd":"stop"}
```

Stops program playback. It does not cut servo torque, because that can make a leg collapse unexpectedly. Use `servo_torque` when you intentionally want to release a servo for setup or limit capture.

`play`

```json
{
  "cmd": "play",
  "loop": false,
  "steps": [
    {"ms": 500, "speed": 700, "accel": 40, "poses": {"1": 90, "2": 150}},
    {"ms": 500, "speed": 700, "accel": 40, "poses": {"1": 120, "2": 120}}
  ]
}
```

Loads and starts a simple pose sequence. Each step duration is in milliseconds.
`speed` and `accel` are optional per-step servo move settings; omitted steps use `900` and `50`.

`wifi_set`

```json
{"cmd":"wifi_set","ssid":"YourNetwork","password":"YourPassword"}
```

Connects the board to Wi-Fi and saves the credentials in ESP32 flash.

`wifi_status`

```json
{"cmd":"wifi_status"}
```

Returns connection state, SSID, IP address, and RSSI.

## Responses

`ok`

```json
{"type":"ok","cmd":"move"}
```

`error`

```json
{"type":"error","message":"unknown servo id"}
```

`config`

```json
{"type":"config","servos":[...]}
```

`servo_scan`

```json
{"type":"servo_scan","start":1,"end":8,"found":[1,2]}
```

`servo_set_id`

```json
{"type":"servo_set_id","current":1,"next":2,"ok":true,"unlock":true,"write":true,"lock":true,"pingNext":true,"pingCurrent":false,"fallbackLock":false,"stage":"ok"}
```

If `ok` is false, `stage` is the first failed step: `unlock`, `write_id`, `lock_new_id`, or `ping_new_id`.

`servo_identify`

```json
{"type":"servo_identify","id":2,"ok":true}
```

`state`

```json
{"type":"state","playing":false,"positions":{"1":120,"2":120},"measured":{"1":119.8,"2":120.1},"velocities":{"1":0,"2":45},"motorModes":{"1":false,"2":true},"motorSpeeds":{"1":0,"2":0.25}}
```

`wifi`

```json
{"type":"wifi","connected":true,"ssid":"YourNetwork","ip":"192.168.1.50","rssi":-52}
```

`imu`

```json
{"type":"imu","available":true,"roll":0.4,"pitch":-1.2,"yaw":85.0,"accel":{"x":0.01,"y":0.02,"z":9.78},"gyro":{"x":0.0,"y":0.0,"z":0.0},"mag":{"x":12,"y":42,"z":-7}}
```

`motors`

```json
{"type":"motors","motors":[{"id":"A","speed":0,"limit":0.35},{"id":"B","speed":0,"limit":0.35}]}
```
