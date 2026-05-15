import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from matplotlib import pyplot

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=28, bias=False)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)  # Flatten mantendo a dimensão batch
        output = F.log_softmax(x, dim=1)
        return output

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)  # Média da loss por batch
    return avg_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0.
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return accuracy

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])

    dataset1 = datasets.MNIST('../mnist/', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./mnist/', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64, shuffle=True)  # shuffle=True para treino
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=64)

    model = CNN().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    scheduler = StepLR(optimizer=optimizer, step_size=1, gamma=0.7)

    loss_list = []
    accuracy_list = []

    for epoch in range(15):
        loss = train(model, device, train_loader, optimizer, epoch)
        loss_list.append(loss)

        accuracy = test(model, device, test_loader)
        accuracy_list.append(accuracy)

        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")

        scheduler.step()

    pyplot.clf()
    pyplot.plot(loss_list, label='loss')
    pyplot.legend()
    pyplot.savefig('./ml_classification_pytorch/loss.png')

    pyplot.clf()
    pyplot.plot(accuracy_list, label='accuracy')
    pyplot.legend()
    pyplot.savefig('./ml_classification_pytorch/accuracy.png')