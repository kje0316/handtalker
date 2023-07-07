from transformers import AutoProcessor, AutoModelForCTC
from moviepy.editor import *
import librosa
import numpy as np
import torch

class STT:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
        self.model = AutoModelForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

    def get_result(self, file_path: str) -> list:
        video = VideoFileClip(file_path)
        audio_nparray = video.audio.to_soundarray()

        sr = int(video.audio.fps)
        waveform = audio_nparray.mean(axis=1) if len(audio_nparray.shape) > 1 else audio_nparray
        num_samples = int(waveform.shape[0] * 16000.0 / sr)
        waveform = np.interp(np.linspace(0, waveform.shape[0], num_samples), np.arange(waveform.shape[0]), waveform)
        sr = 16000
        
        
        inputs = self.processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values
        with torch.no_grad():
            logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription


model = STT()
result = model.get_result('dog_sound_ex.mp4')
print(result)
