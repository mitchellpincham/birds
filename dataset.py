import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import librosa
import librosa.display
from pydub import AudioSegment

mapping2 = {
    "kaka": 0,
    "kiwi": 1,
    "kokako": 2,
    "ruru": 3,
    "tui": 4,
    "tieke": 5
}

class BirdDataset(Dataset):

    def __init__(self, transformation, device, target_sample_rate, num_samples):
        self.annotations = pd.read_parquet("birds.parquet")
        self.audio_dir = "data/"
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        # length of dataset  len(usd)
        return len(self.annotations)

    def __getitem__(self, index):
        # get item  usd[index]
        audio_sample_path = self.annotations.iloc[index, 3]
        label = self._get_audio_sample_label(index)
        label = torch.tensor(label, dtype=torch.long)

        signal, sr = self._load_audio(audio_sample_path)

        #print(self.annotations.iloc[index, 3][-3:])
        #if self.annotations.iloc[index, 3][-3:] == "m4a":
        #    signal, sr = self._load_m4a(audio_sample_path)
        #else:
        #    signal, sr = librosa.load(audio_sample_path, sr=None)

        signal = torch.from_numpy(signal)

        signal = signal.to(self.device)
        signal = signal.to(torch.float32)

        signal = self._resample_if_necessary(signal, sr)    # change so all are the same sample rate
        #signal = self._mix_down_if_necessary(signal)        # incase signal is multiple channels

        signal = self._right_pad_if_necessary(signal)       # if the clip is too short
        signal = self._cut_if_necessary(signal)             # if the clip is too long
        
        signal = self.transformation(signal)                # returns a mel spectrogram
        return signal.unsqueeze(0), label
    

    # private methods
    
    def _load_audio(self, audio_sample_path):
        audio = AudioSegment.from_file(audio_sample_path)

        # Convert the audio to a numpy array (mono channel)
        samples = np.array(audio.get_array_of_samples())

        # Rescale to the correct float range (-1.0 to 1.0)
        y = samples / float(2**15)

        # Get the sampling rate (this should be 44100 by default for M4A files)
        sr = audio.frame_rate

        return y, sr

    
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 4]

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    """
    def _mix_down_if_necessary(self, signal):
        # agregate channels
        # signal -> (num_channels, samples), eg. (2, 16000) -> (1, 16000)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    """
    
    def _cut_if_necessary(self, signal):
        if len(signal) > self.num_samples:
            signal = signal[0 : self.num_samples]
        return signal
    
    def _right_pad_if_necessary(self, signal):
        if len(signal) < self.num_samples:
            missing_samples = self.num_samples - len(signal)
            padding = (0, missing_samples)
            signal = torch.nn.functional.pad(signal, padding)
        return signal
    


SAMPLE_RATE = 22050
NUM_SAMPLES = 22050  # one second


if __name__ == "__main__":

    # learn more about the mel_spectogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


    bird_dataset = BirdDataset(mel_spectrogram, device, SAMPLE_RATE, NUM_SAMPLES)

    print(f"There are {len(bird_dataset)} samples")

    signal, label = bird_dataset[0]

    print(signal.shape)
    print(label)
