from picamera2 import Picamera2

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())

# Start the camera, take a picture, and save it
picam2.start()
picam2.capture_file("/home/henri/Documents/Projects/Raspi-Face-Detection/images/test_pic.jpg")
picam2.stop()
