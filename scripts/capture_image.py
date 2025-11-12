import time
import os
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import glob

# Set up GPIO pins
PIR_PIN = 27
LED_PIN = 23
INFRARET_LED_PIN = 25
NUM_IMAGES = 6
COOLDOWN = 10

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)   # << pull-down!
GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(INFRARET_LED_PIN, GPIO.OUT, initial=GPIO.LOW)

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())

# Create directory to save images if it doesn't exist
IMAGE_DIR = "/home/henri/Documents/Raspi-Face-Detection/data/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def handle_motion(channel):
    global image_count

    GPIO.remove_event_detect(PIR_PIN)
    print("Motion detected!")
    GPIO.output(INFRARET_LED_PIN, GPIO.HIGH)
    GPIO.output(LED_PIN, GPIO.HIGH)

    for i in range(NUM_IMAGES):
        capture_image(image_count, i)
        image_count += 1
        time.sleep(0.7)

    GPIO.output(INFRARET_LED_PIN, GPIO.LOW)
    GPIO.output(LED_PIN, GPIO.LOW)
    time.sleep(COOLDOWN)
    GPIO.add_event_detect(PIR_PIN, GPIO.RISING,
                        callback=handle_motion, bouncetime=100)


def capture_image(image_count, idx):
    image_path = os.path.join(IMAGE_DIR, f"image_{image_count}_{idx}idx.jpg")
    picam2.capture_file(image_path)
    
def get_next_image_number():
    """Get the next image number based on existing images in the directory."""
    pattern = os.path.join(IMAGE_DIR, "**", "image_*.jpg")
    image_files = [os.path.basename(f) for f in glob.glob(pattern, recursive=True)]
    if not image_files:
        return 1
    # Extract numbers from filenames and find the maximum
    image_numbers = [int(f.split("_")[1]) for f in image_files]
    return max(image_numbers) + 1


try:
    # Start the camera
    picam2.start()
    image_count = get_next_image_number()

    GPIO.add_event_detect(PIR_PIN, GPIO.RISING,
                        callback=handle_motion, bouncetime=100)
    print("Waiting for motion...")
    while True:
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting program...")

finally:
    # Clean up GPIO and stop camera
    GPIO.cleanup()
    picam2.stop()
