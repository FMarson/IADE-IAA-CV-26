import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from matplotlib import pyplot

def train(model, device, train_loader, optimizer, epoch):
    model.train()

    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    return test_loss / len(test_loader.dataset)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features=9216, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        #print(f"0 - Input shape: {x.shape}")
        x = self.conv1(x)
        #print(f"1 - Depois da conv1: {x.shape}")
        x = F.relu(x)
        x = self.conv2(x)
        #print(f"2 - Depois da conv2: {x.shape}")
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        #print(f"3 - Depois do max pooling: {x.shape}")
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        #print(f"4 - Depois do flatten: {x.shape}")
        x = self.fc1(x)
        #print(f"5 - Depois da fc1: {x.shape}")
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #print(f"6 - Depois da fc2: {x.shape}")
        output = F.log_softmax(x, dim=1)
        return output
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor()])  

    dataset1 = datasets.MNIST('../mnist/', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../mnist/', train=False, download=True,transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=64)

    data = list(train_loader)[0][0]

    model = Net().to(device)
    #prediction = model(data.to(device))
    #print(prediction.shape)
    
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    
    scheduler = StepLR(optimizer=optimizer, step_size=1, gamma=0.7)
    
    loss_list = list()
    accuracy_list = list()

    for epoch in range(1, 15 + 1):
        loss = train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)

        print(f'Epoch: {epoch}, Train Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        scheduler.step() 

    #torch.save(model.state_dict(), "mnist_cnn.pth")

    # Plotando os gráficos de loss e acurácia
    pyplot.clf()
    pyplot.plot(loss_list, label='Train Loss')
    pyplot.savefig('loss.png')

    pyplot.clf()
    pyplot.plot(accuracy_list, label='Test Accuracy')   
    pyplot.savefig('accuracy.png')

    