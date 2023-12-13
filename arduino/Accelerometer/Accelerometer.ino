#include "Model.h"

#include <ArduinoBLE.h>
#include <Arduino_APDS9960.h>
#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;

constexpr int kTensorArenaSize = 11 * 1024;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
} // namespace

const char *deviceServiceUuid = "96b1c8ed-fd4b-4bc3-b5da-9e3ed654f1b1";
const char *deviceServiceCharacteristicUuid = "551de921-bbaa-4e0a-9374-3e30e88a9073";

BLEService accelerometerService(deviceServiceUuid);
BLECharacteristic accelerometerCharacteristic(deviceServiceCharacteristicUuid, BLERead | BLEWrite, 24);

union DeviceData {
    float values[6];
    unsigned char bytes[24];
};

struct ModelResult {
    bool decision;
    float score;
};

const unsigned char initializerAcc[24] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

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
    while (!Serial)
        ;
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
    accelerometerCharacteristic.writeValue(initializerAcc, 24);
    BLE.advertise();

    // Set up accelerometer
    if (!IMU.begin()) {
        Serial.println("Failed to initialize IMU!");
        while (1)
            ;
    }

    Serial.print("Accelerometer sample rate = ");
    Serial.print(IMU.accelerationSampleRate());
    Serial.println(" Hz");
    Serial.println();
    Serial.println("Acceleration in G's");

    // Tensorflow Model
    Serial.println("Setting up model");
    tflite::InitializeTarget();
    model = tflite::GetModel(models_dnn_tflite_model_quantized_tflite);
    static tflite::MicroMutableOpResolver<9> op_resolver;

    if (op_resolver.AddFullyConnected() != kTfLiteOk) {
        Serial.println("error, could not add fully connected");
        return;
    }
    if (op_resolver.AddSub() != kTfLiteOk) {
        Serial.println("error, could not add sub");
        return;
    }
    if (op_resolver.AddLogistic() != kTfLiteOk) {
        Serial.println("error, could not add logistic");
        return;
    }
    if (op_resolver.AddAdd() != kTfLiteOk) {
        Serial.println("error, could not add add");
        return;
    }
    if (op_resolver.AddMul() != kTfLiteOk) {
        Serial.println("error, could not add mul");
        return;
    }
    if (op_resolver.AddSqrt() != kTfLiteOk) {
        Serial.println("error, could not add sqrt");
        return;
    }
    if (op_resolver.AddConcatenation() != kTfLiteOk) {
        Serial.println("error, could not add concatenation");
        return;
    }
    if (op_resolver.AddGreater() != kTfLiteOk) {
        Serial.println("error, could not add greater");
        return;
    }

    static tflite::MicroInterpreter static_interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("AllocateTensors() failed");
        return;
    }
    Serial.println("Model setup done");
    Serial.println(" ");
    Serial.println("Data Format:");
    Serial.println("time\tX\tY\tZ\tdecision\tscore");
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
            float inputs[9] = {
                0.,          // y-2
                0.111450195, // y-1
                0.13586426,  // y
                -0.42260742, // x-1
                -0.46069336, // x
                0.90930176,  // z-1
                0.,          // z-2
                0.8876953,   // z
                0.,          // x-2
            };
            for (int i = 0; i < 9; i++) {
                interpreter->input(i)->data.f[0] = inputs[i];
            }
            TfLiteStatus invoke_status = interpreter->Invoke();
            if (invoke_status != kTfLiteOk) {
                Serial.println("Invoke failed");
                return;
            }
            TfLiteTensor *output_decision = interpreter->output(0);
            TfLiteTensor *output_score = interpreter->output(1);
            bool decision = output_decision->data.b[0];
            float score = output_score->data.f[0];

            Serial.print("\t");
            Serial.print(decision);
            Serial.print("\t");
            Serial.println(score);

            bool button_pressed = buttonPressed();
            data.values[3] = button_pressed ? 1 : 0;
            data.values[4] = decision ? 1 : 0;
            data.values[5] = score;
            if (button_pressed) {
                Serial.println("button pressed");
                digitalWrite(LED_BUILTIN, HIGH);
            } else {
                digitalWrite(LED_BUILTIN, LOW);
            }
            unsigned char *acc = (unsigned char *)&data;
            int success = accelerometerCharacteristic.writeValue(acc, 24);
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
        Serial.print(z);
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

// ModelResult getModelResult(float inputs[9], tflite::MicroInterpreter *interpreter) {
//     for (int i = 0; i < 9; i++) {
//         interpreter->input(i)->data.f[0] = inputs[i];
//     }
//     TfLiteStatus invoke_status = interpreter->Invoke();
//     if (invoke_status != kTfLiteOk) {
//         Serial.println("Invoke failed");
//         return ModelResult();
//     }
//     TfLiteTensor *output_decision = interpreter->output(0);
//     TfLiteTensor *output_score = interpreter->output(1);
//     bool decision = output_decision->data.b[0];
//     float score = output_score->data.f[0];
//
//     ModelResult result;
//     result.decision = decision;
//     result.score = score;
//     return result;
// }
