import time
import os
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# Set up GPIO pins
PIR_PIN = 27
LED_PIN = 23
INFRARET_LED_PIN = 25
NUM_IMAGES = 3

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(INFRARET_LED_PIN, GPIO.OUT)

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())

# Create directory to save images if it doesn't exist
IMAGE_DIR = "/home/henri/Documents/Raspi-Face-Detection/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Start the camera
picam2.start()

def capture_image(image_count):
    image_path = os.path.join(IMAGE_DIR, f"image_{image_count}.jpg")
    picam2.capture_file(image_path)
    print(f"Captured {image_path}")
    
def get_next_image_number():
    """Get the next image number based on existing images in the directory."""
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.startswith("image_") and f.endswith(".jpg")]
    if not image_files:
        return 1
    # Extract numbers from filenames and find the maximum
    image_numbers = [int(f.split("_")[1].split(".")[0]) for f in image_files]
    return max(image_numbers) + 1

try:
    print("Waiting for motion...")
    image_count = get_next_image_number()
    while True:
        # Check for motion
        if GPIO.input(PIR_PIN):
            print("Motion detected!")
            GPIO.output(INFRARET_LED_PIN, GPIO.HIGH)
            for _ in range(NUM_IMAGES):
                time.sleep(0.3)
                # Capture image
                capture_image(image_count)
                image_count += 1
            time.sleep(1)
            GPIO.output(INFRARET_LED_PIN, GPIO.LOW)
        
        # Short delay to avoid multiple triggers
        time.sleep(3)

except KeyboardInterrupt:
    print("Exiting program...")

finally:
    # Clean up GPIO and stop camera
    GPIO.cleanup()
    picam2.stop()
