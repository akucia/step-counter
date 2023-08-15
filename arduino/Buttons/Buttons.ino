
void setup() {
    Serial.begin(9600);
    digitalWrite(LED_BUILTIN, HIGH);
       while (!Serial)
           ;
    Serial.println("Started");
}

void loop() {

    digitalWrite(LED_BUILTIN, HIGH);
    delay(500);
    digitalWrite(LED_BUILTIN, LOW);
    delay(500);
    digitalWrite(LEDR, HIGH); //RED
    digitalWrite(LEDG, HIGH); //GREEN
    digitalWrite(LEDB, HIGH); //BLUE
    delay(500);
    digitalWrite(LEDR, LOW); //RED
    digitalWrite(LEDG, LOW); //GREEN
    digitalWrite(LEDB, LOW); //BLUE
}


