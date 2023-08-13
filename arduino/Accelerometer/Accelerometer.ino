#include <ArduinoBLE.h>
#include <Arduino_APDS9960.h>
#include <Arduino_LSM9DS1.h>

const char *deviceServiceUuid = "96b1c8ed-fd4b-4bc3-b5da-9e3ed654f1b1";
const char *deviceServiceCharacteristicUuid = "551de921-bbaa-4e0a-9374-3e30e88a9073";

BLEService accelerometerService(deviceServiceUuid);
BLECharacteristic accelerometerCharacteristic(deviceServiceCharacteristicUuid, BLERead | BLEWrite, 12);

union AccelerometerData {
    float values[3];
    unsigned char bytes[12];
};

const unsigned char initializerAcc[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

void setup() {
    Serial.begin(9600);
    digitalWrite(LED_BUILTIN, HIGH);
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
    accelerometerCharacteristic.writeValue(initializerAcc, 12);
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
    digitalWrite(LED_BUILTIN, HIGH);
    Serial.println("- Discovering central device...");

    if (central) {
        Serial.println("* Connected to central device!");
        Serial.print("* Device MAC address: ");
        Serial.println(central.address());
        Serial.println(" ");
        while (central.connected()) {
            digitalWrite(LED_BUILTIN, HIGH);
            AccelerometerData data = getAccelerometer();
            unsigned char *acc = (unsigned char *)&data;
            int success = accelerometerCharacteristic.writeValue(acc, 12);
            if (!success) {
                Serial.println("value not written");
            }
            digitalWrite(LED_BUILTIN, LOW);
            delay(10);
        }
        Serial.println("* Disconnected from central device!");
    }
    delay(500);
    digitalWrite(LED_BUILTIN, LOW);
    delay(200);
}
bool setValue(const unsigned char value[], unsigned short length) {
    Serial.println("setValue");
    return true;
}

AccelerometerData getAccelerometer() {
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
    union AccelerometerData data;
    data.values[0] = x;
    data.values[1] = y;
    data.values[2] = z;
    return data;
}
