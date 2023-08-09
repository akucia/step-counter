#include <ArduinoBLE.h>
#include <Arduino_APDS9960.h>
#include <Arduino_LSM9DS1.h>

const char *deviceServiceUuid = "96b1c8ed-fd4b-4bc3-b5da-9e3ed654f1b1";
const char *deviceServiceCharacteristicUuid = "551de921-bbaa-4e0a-9374-3e30e88a9073";

BLEService accelerometerService(deviceServiceUuid);
BLEByteCharacteristic accelerometerCharacteristic(deviceServiceCharacteristicUuid, BLERead | BLEWrite);

void setup() {
    Serial.begin(9600);
    digitalWrite(LED_BUILTIN, LOW);
    while (!Serial)
        ;
    Serial.println("Started");

    // Setting up bluetooth connection
    if (!BLE.begin()) {
        Serial.println("- Starting Bluetooth® Low Energy module failed!");
        while (1)
            ;
    }

    BLE.setLocalName("Arduino Nano 33 BLE (Accelerometer)");
    // assign service and characteristic to bluetooth
    BLE.setAdvertisedService(accelerometerService);
    accelerometerService.addCharacteristic(accelerometerCharacteristic);
    BLE.addService(accelerometerService);
    accelerometerCharacteristic.writeValue(-1);
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
    Serial.println("- Discovering central device...");
    delay(500);

    if (central) {
        Serial.println("* Connected to central device!");
        Serial.print("* Device MAC address: ");
        Serial.println(central.address());
        Serial.println(" ");
        while (central.connected()) {
            if (accelerometerCharacteristic.written()) {
                writeAccelerometer();
            };
        }
        Serial.println("* Disconnected from central device!");
    }
}

void writeAccelerometer() {
    float x, y, z;

    if (IMU.accelerationAvailable()) {
        digitalWrite(LED_BUILTIN, HIGH);

        IMU.readAcceleration(x, y, z);
        Serial.print(millis());
        Serial.print('\t');
        Serial.print(x);
        Serial.print('\t');
        Serial.print(y);
        Serial.print('\t');
        Serial.println(z);
        digitalWrite(LED_BUILTIN, LOW);

        delay(100);
    }
}