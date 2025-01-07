import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from dataset import BirdDataset
from cnn import CNNNetwork


BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 0.001

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050  # half a minute


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_dataloader


def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()
    running_loss = 0.0
    loss = None

    for inputs, targets in data_loader:
        try:
            inputs, targets = inputs.to(device), targets.to(device)
        except Exception as e:
            print("Error: ", e)
            continue

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagate loss and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        running_loss += loss.item()

    avg_loss = running_loss / float(len(data_loader))

    print(f"AVERAGE LOSS: {avg_loss}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)


    print("Training done")


if __name__ == "__main__":
    '''
    if torch.cuda.is_available():
        device = "cuda"
    else: 
        device = "cpu"'''
    device = "cpu"

    # instantiate our dataset object
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024,hop_length=512, n_mels=64)

    dataset = BirdDataset(mel_spectrogram, device, SAMPLE_RATE, NUM_SAMPLES)

    # data loader
    train_data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)


    # build model and assign device
    print(f"Using {device} device")
    feed_forward_net = CNNNetwork().to(device)

    # instantiate loss function and optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(), lr=LEARNING_RATE)

    # train model
    train(feed_forward_net, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    # store model
    torch.save(feed_forward_net.state_dict(), "cnn.pth")

    print("Model trained and stored at cnn.pth")