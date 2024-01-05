import torch
import os
import torchaudio

from torch.utils.data import Dataset
from encodec.utils import convert_audio

SAMPLE_RATE = 24_000
CODEBOOKS_SIZE = 1024

class AudioTextDataset(Dataset):
    def __init__(self, directory, encodec_model, tokenizer, max_text_tokens, max_audio_tokens, audio_duration):
        self.directory = directory
        self.encodec_model = encodec_model
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens
        self.max_audio_tokens = max_audio_tokens
        self.audio_duration = audio_duration
        
        self.wav_files = []
        self.txt_files = []

        for f in os.listdir(directory):
            if f.endswith('.txt'):
                txt_file_path = os.path.join(directory, f)
                with open(txt_file_path, 'r') as txtf:
                    content = txtf.read().strip()
                    if not content:
                        continue

                wav_file_name = os.path.splitext(f)[0] + '.wav'
                wav_file_path = os.path.join(directory, wav_file_name)
                if os.path.exists(wav_file_path):
                    waveform, sample_rate = torchaudio.load(wav_file_path)
                    duration = waveform.shape[1] / sample_rate
                    if duration == self.audio_duration:
                        self.wav_files.append(wav_file_name)
                        self.txt_files.append(f)

    def __len__(self):
        return len(self.wav_files)

    def load_audio_tokens(self, filename):
        waveform, sample_rate = torchaudio.load(filename)
        waveform = convert_audio(waveform, sample_rate, self.encodec_model.sample_rate, self.encodec_model.channels)
        waveform = waveform.unsqueeze(0).to("cuda")

        with torch.no_grad():
            encoded_frames = self.encodec_model.encode(waveform)
        audio_tokens = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)

        if audio_tokens.shape[-1] > self.max_audio_tokens:
            audio_tokens = audio_tokens[:, :, :self.max_audio_tokens]
        
        return audio_tokens

    def load_text_tokens(self, filename):
        with open(filename, 'r') as f:
            text = f.read().strip()

        tokens = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_text_tokens).input_ids
        return tokens

    def __getitem__(self, idx):
        audio_tokens = self.load_audio_tokens(os.path.join(self.directory, self.wav_files[idx]))
        text_tokens = self.load_text_tokens(os.path.join(self.directory, self.txt_files[idx]))

        return audio_tokens, text_tokens