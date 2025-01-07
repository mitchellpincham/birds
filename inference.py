import torch
import torchaudio
from cnn import CNNNetwork
from dataset import BirdDataset
from torch.utils.data import DataLoader
from train import SAMPLE_RATE, NUM_SAMPLES


class_mapping = [
    "kaka",
    "kiwi",
    "kokako",
    "ruru",
    "tui",
    "tieke"
]


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_dataloader


def predict(model, input, target, class_mapping):
    model.eval() # switches the model to prediction, use model.train() to switch to training mode
    
    with torch.no_grad():
        predictions = model(input) # -> [[0.1, 0.0, ..., 0.7]]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]

        expected = class_mapping[target]
    
    return predicted, expected


    

if __name__ == "__main__":

    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("cnn.pth")
    cnn.load_state_dict(state_dict)

    # load urban sound validation dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024,hop_length=512, n_mels=64)

    dataset = BirdDataset(mel_spectrogram, "cpu", SAMPLE_RATE, NUM_SAMPLES)
    
  
    # get a sample from the validaton dataset for inference
    input, target = dataset[0][0], dataset[0][1]  # [batch_size, num_channels, freq, time]
    input.unsqueeze_(0)


    # make an inference
    predicted, expected = predict(cnn, input, target, class_mapping)

    print(f"Predicted = {predicted}, Expected = {expected}")