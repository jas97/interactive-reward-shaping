import torch
from torch import nn
from torch.optim import RMSprop
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset

from src.reward_modelling.enc_dec import EncoderDecoder


class RewardNet(nn.Module):

    def __init__(self, input_size):
        super(RewardNet, self).__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RewardModelNN:

    def __init__(self, input_size, expert_data):
        self.input_size = input_size
        self.enc_size = input_size

        self.enc_dec = EncoderDecoder(self.input_size, self.enc_size)
        self.net = RewardNet(self.enc_size)

        self.criterion = nn.MSELoss()
        self.optimizer = RMSprop(self.net.parameters(), lr=0.001)

        # train, test = torch.utils.data.random_split(expert_data, [int(0.8*len(expert_data)), len(expert_data) - int(0.8*len(expert_data))])
        # self.enc_dec.train(DataLoader(train), DataLoader(test))

    def train(self, train_dataloader):
        self.net.train()
        print('Updating reward model...')

        batch_size = train_dataloader.batch_size
        for i in range(20):
            total_loss = 0.0
            for x, y in train_dataloader:
                self.optimizer.zero_grad()

                x_enc = self.enc_dec.encode(x)
                output = self.net.float()(x_enc.float())

                loss = self.criterion(output.flatten().float(), y.flatten().float())

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print('Epoch = {}. Training loss = {} Training examples = {}'.format(i, total_loss / (len(train_dataloader) * batch_size), batch_size*len(train_dataloader)))

    def evaluate(self, dataloader):
        self.net.eval()

        total_loss = 0.0
        batch_size = dataloader.batch_size

        for x, y in dataloader:
            enc_x = self.enc_dec.encode(x)
            output = self.net.float()(enc_x.float())

            loss = self.criterion(y.flatten().float(), output.flatten().float())

            total_loss += loss.item()

        print('MSE loss on test dataset = {}'.format(total_loss / (batch_size*len(dataloader))))

    def predict(self, x):
        self.net.eval()
        x = torch.Tensor(x)
        enc_x = self.enc_dec.encode(x)
        output = self.net.float()(enc_x.float())

        return output

