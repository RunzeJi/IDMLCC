from central import load_data, load_model, train, test
import flwr as fl
import pandas as pd
import torch, csv, cfg
from collections import OrderedDict

LOCAL_EPOCHS = cfg.WORKER_1_RNDS
FILENAME = cfg.LOSS_FILE_NAME
loss_list = []

train_csv = cfg.WORKER_1_DS
test_csv = cfg.TEST_DS
print(train_csv)

input_size = pd.read_csv(train_csv).shape[1] - 1
# num_classes = len(pd.read_csv(train_csv)['current_service'].unique())
num_classes = 9

print(input_size, ', ', num_classes)

net = load_model(input_size=input_size, num_classes=num_classes)
trainloader, testloader = load_data(train_csv=train_csv, test_csv=test_csv)

class DNNClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=LOCAL_EPOCHS)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        loss_list.append(loss)
        print(f'Loss: {loss:.5f}, Accuracy: {accuracy:.5f}')
        
        return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}
    
fl.client.start_client(server_address='192.168.100.1:8080',
                       client=DNNClient().to_client(),
                       )

with open(FILENAME, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['iteration', 'loss'])

    for i, number in enumerate(loss_list, start=1):
        writer.writerow([i, number])

print(loss_list)
