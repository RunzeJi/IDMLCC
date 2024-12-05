import torch, cfg
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

DEVICE = cfg.BACKEND
LR = 0.001
MOMENTUM = 0.9
BATCH_SIZE = 128
L2_REG = 1e-5
DAMPENING = 0

# EPOCHS only work in centralized
EPOCHS = 10

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features = self.data.iloc[:, :-1].values.astype(float)
        self.labels = self.data.iloc[:, -1].values.astype(int)
        
        # 标准化特征
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label

class Net(nn.Module):
    def __init__(self, input_size, num_classes) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 192)
        self.fc2 = nn.Linear(192, 96)
        self.fc3 = nn.Linear(96, 48)
        self.fc4 = nn.Linear(48, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

def train(net, trainloader, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=L2_REG, dampening=DAMPENING)
    for _ in range(epochs):
        for features, labels in trainloader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(features), labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)  # Ensure both features and labels are on the correct device
            outputs = net(features)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    # print(f'Local accuracy: {correct/total}, Local loss: {loss}')
    return loss / len(testloader.dataset), correct / total

def load_data(train_csv, test_csv):
    trainset = CustomDataset(csv_file=train_csv)
    testset = CustomDataset(csv_file=test_csv)
    
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    
    return trainloader, testloader

def load_model(input_size, num_classes):
    return Net(input_size, num_classes).to(DEVICE)


if __name__ == '__main__':
    train_csv = '../tpd_300k_train.csv'
    test_csv = '../tpd_20k_test.csv'

    # Assumes the input size and number of classes are known
    input_size = pd.read_csv(train_csv).shape[1] - 1
    num_classes = len(pd.read_csv(train_csv)['current_service'].unique())
    
    net = load_model(input_size, num_classes)
    trainloader, testloader = load_data(train_csv, test_csv)
    train(net, trainloader, EPOCHS)
    loss, accuracy = test(net, testloader)
    print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.3f}')
