#include <ArduinoBLE.h>
#include <Arduino_APDS9960.h>
#include <Arduino_LSM9DS1.h>

const char *deviceServiceUuid = "96b1c8ed-fd4b-4bc3-b5da-9e3ed654f1b1";
const char *deviceServiceCharacteristicUuid = "551de921-bbaa-4e0a-9374-3e30e88a9073";

BLEService accelerometerService(deviceServiceUuid);
BLECharacteristic accelerometerCharacteristic(deviceServiceCharacteristicUuid, BLERead | BLEWrite, 16);

union DeviceData {
    float values[4];
    unsigned char bytes[16];
};

const unsigned char initializerAcc[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

const int buttonPin = 2;

void setup() {
    Serial.begin(9600);
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);
    //    while (!Serial)
    //        ;
    Serial.println("Started");

    // Setting up bluetooth connection
    if (!BLE.begin()) {
        Serial.println("- Starting Bluetooth® Low Energy module failed!");
        while (1)
            ;
    }
    BLE.setDeviceName("Arduino Nano 33 BLE (Accelerometer)");
    BLE.setLocalName("Arduino Nano 33 BLE (Accelerometer)");
    // assign service and characteristic to bluetooth
    BLE.setAdvertisedService(accelerometerService);
    accelerometerService.addCharacteristic(accelerometerCharacteristic);
    BLE.addService(accelerometerService);
    accelerometerCharacteristic.writeValue(initializerAcc, 16);
    BLE.advertise();

    // Set up accelerometer
    if (!IMU.begin()) {
        Serial.println("Failed to initialize IMU!");
        while (1)
            ;
    }

    //  Serial.print("Accelerometer sample rate = ");
    //  Serial.print(IMU.accelerationSampleRate());
    //  Serial.println(" Hz");
    //  Serial.println();
    //  Serial.println("Acceleration in G's");
    //  Serial.println("time\tX\tY\tZ");
}

void loop() {

    BLEDevice central = BLE.central();
    digitalWrite(LED_BUILTIN, LOW);
    digitalWrite(LEDB, HIGH);
    digitalWrite(LEDR, LOW);

    Serial.println("- Discovering central device...");

    if (central) {
        Serial.println("* Connected to central device!");
        Serial.print("* Device MAC address: ");
        Serial.println(central.address());
        Serial.println(" ");
        digitalWrite(LEDB, LOW);
        while (central.connected()) {
            digitalWrite(LEDR, HIGH);
            DeviceData data = getAccelerometer();
            bool button_pressed = buttonPressed();
            data.values[3] = button_pressed ? 1 : 0;
            if (button_pressed) {
                Serial.println("button pressed");
                digitalWrite(LED_BUILTIN, HIGH);
            } else {
                digitalWrite(LED_BUILTIN, LOW);
            }
            unsigned char *acc = (unsigned char *)&data;
            int success = accelerometerCharacteristic.writeValue(acc, 16);
            if (!success) {
                Serial.println("value not written");
            }
            delay(10);
        }
        Serial.println("* Disconnected from central device!");
        digitalWrite(LEDR, LOW);
    }
    delay(500);
    digitalWrite(LEDB, HIGH);
    digitalWrite(LED_BUILTIN, LOW);
    delay(200);
}
bool setValue(const unsigned char value[], unsigned short length) {
    Serial.println("setValue");
    return true;
}

DeviceData getAccelerometer() {
    float x = 0, y = 0, z = 0;
    if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(x, y, z);
        Serial.print(millis());
        Serial.print('\t');
        Serial.print(x);
        Serial.print('\t');
        Serial.print(y);
        Serial.print('\t');
        Serial.println(z);
    }
    union DeviceData data;
    data.values[0] = x;
    data.values[1] = y;
    data.values[2] = z;
    return data;
}

bool buttonPressed() {
    int buttonState = digitalRead(buttonPin);
    return buttonState == LOW;
}
