from moviepy.video.io.VideoFileClip import VideoFileClip
import librosa
import numpy as np
import tensorflow_hub as hub

# Load YAMNet model
model = hub.load('https://tfhub.dev/google/yamnet/1')
print("Model loaded:", model)

# Preprocess audio
def preprocess_audio(file_path):
    waveform, sr = librosa.load(file_path, sr=16000, mono=True)
    waveform = librosa.util.fix_length(waveform, size=16000)
    print("Waveform shape:", waveform.shape, "Sample rate:", sr)
    return waveform

# Classify audio
def classify_audio(file_path):
    waveform = preprocess_audio(file_path)
    scores, _, _ = model(waveform)
    predicted_idx = np.argmax(scores, axis=-1)[0]
    intensity = np.mean(np.abs(waveform))
    print("Predicted Class ID:", predicted_idx)
    if predicted_idx == 321:  # Traffic noise (rush hour)
        level = "Low" if intensity < 0.1 else "Moderate" if intensity < 0.5 else "Severe"
        print(f"Predicted: Traffic Jam, Blockage: {level}")
    elif predicted_idx == 316:  # Emergency (ambulance sound)
        level = "Low" if intensity < 0.1 else "Medium" if intensity < 0.5 else "High"
        print(f"Predicted: Ambulance, Urgency: {level}")
    elif predicted_idx == 294:  # Vehicle sound (vehicles crashing)
        level = "Mild" if intensity < 0.3 else "Severe"
        print(f"Predicted: Accident, Severity: {level}")
    elif predicted_idx == 494:  # Silence (normal traffic)
        print("Predicted: Normal Traffic (Silence)")
    else:
        print("No relevant sound detected.")

# Extract audio from video and classify
video = VideoFileClip('vid1-indamb.mp4')
audio = video.audio
audio.write_audiofile('extracted_audio1.wav')
classify_audio('extracted_audio1.wav')