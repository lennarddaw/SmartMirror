#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>


const char* ssid = "Schule"; //Schule
const char* password = "WlanAccess"; //WlanAccess

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

WebServer server(80);

void handle_jpg_stream() {
  WiFiClient client = server.client();
  camera_fb_t *fb = nullptr;
  String boundary = "esp32cam";

  // HTTP‑Header richtig senden
  client.print("HTTP/1.0 200 OK\r\n");
  client.print("Content-Type: multipart/x-mixed-replace; boundary=" + boundary + "\r\n\r\n");

  while (client.connected()) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Framebuffer konnte nicht geholt werden");
      continue;
    }

    client.print("--" + boundary + "\r\n");
    client.print("Content-Type: image/jpeg\r\n\r\n");
    client.write(fb->buf, fb->len);
    client.print("\r\n");

    esp_camera_fb_return(fb);
    delay(10);
  }
}


void setup() {
  Serial.begin(115200);

  // Kamera-Konfiguration
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000; 
  config.pixel_format = PIXFORMAT_JPEG;

  // Kleinere Auflösung für stabile Streams
  config.frame_size = FRAMESIZE_VGA; // VGA (640x480)
  config.jpeg_quality = 12;          // (0-63), niedriger = bessere Qualität
  config.fb_count = 2;               // Erhöht Stabilität des Streams

  // Kamera init
   esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Kamerainit fehlgeschlagen (0x%x)\n", err);
    return;
  }

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.println("Mit dem WLAN verbunden");
  Serial.println(WiFi.localIP());

  server.on("/stream", handle_jpg_stream);
  server.begin();
  Serial.println("Server wurde gestartet!");
}

void loop(){
  server.handleClient();
}


