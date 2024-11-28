import os

# Path to your audio file
audio_file = "/home/henri/Documents/Projects/Raspi-Face-Detection/BabyElephantWalk60_stereo.wav"

# Play the audio using aplay
os.system(f"aplay -Dhw:2 {audio_file}")
