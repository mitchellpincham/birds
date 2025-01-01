import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import librosa
import librosa.display




class BirdDataset(Dataset):

    def __init__(self, transformation, device):
        self.device = device
        self.transformation = transformation.to(self.device)

    def __len__(self):
        # length of dataset  len(usd)
        return len(self.annotations)

    def __getitem__(self, index):
        # get item  usd[index]
        #audio_sample_path = self._get_audio_sample_path(index)
        #label = self._get_audio_sample_label(index)
        #signal, sr = torchaudio.load(audio_sample_path)

        #signal = signal.to(self.device)

        '''
        signal = self._resample_if_necessary(signal, sr)    # change so all are the same sample rate
        signal = self._mix_down_if_necessary(signal)        # incase signal is multiple channels

        signal = self._right_pad_if_necessary(signal)       # if the clip is too short
        signal = self._cut_if_necessary(signal)             # if the clip is too long

        signal = self.transformation(signal)                # returns a mel spectrogram'''
        return 1#signal, label
    


if __name__ == "__main__":

    # learn more about the mel_spectogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(n_fft=1024, hop_length=512, n_mels=64)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


    bird_dataset = BirdDataset(mel_spectrogram, device)

    bird_dataset.download_files()
