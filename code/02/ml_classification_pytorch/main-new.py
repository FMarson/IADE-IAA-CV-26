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
        # 1º camada conv: 1 canal entrada, 32 saída, kernel 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 2º camada conv: 32 entrada, 64 saída, kernel 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling 2x2
        self.pool = nn.MaxPool2d(2, 2)
        # Camada fully connected - calcular tamanho da entrada depois do pooling
        # Como MNIST é 28x28:
        # após duas conv com padding=1 e kernel=3: mantém tamanho 28x28
        # após pool 2x2: reduz para 14x14
        # após segunda pool: reduz para 7x7 (vamos usar pool duas vezes)
        # Para isso, aplicaremos pooling após cada conv
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes MNIST

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)                 # tamanho reduz para 14x14
        x = F.relu(self.conv2(x))
        x = self.pool(x)                 # tamanho reduz para 7x7
        x = torch.flatten(x, 1)          # achata mantendo batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x                        # saída sem softmax (CrossEntropyLoss aplica internamente)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)  # CrossEntropyLoss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    return avg_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0.
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    avg_loss = test_loss / len(test_loader.dataset)
    return avg_loss, accuracy

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))   # normalização MNIST padrão
    ])

    dataset_train = datasets.MNIST('./mnist/', train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST('./mnist/', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1000)

    model = CNN().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer=optimizer, step_size=1, gamma=0.7)

    train_losses = []
    test_losses = []
    accuracies = []

    epochs = 15

    for epoch in range(epochs):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss, accuracy = test(model, device, test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Accuracy: {accuracy:.4f}")

        scheduler.step()

    # Plotar perdas e acurácia
    pyplot.figure()
    pyplot.plot(range(1, epochs+1), train_losses, label='Train Loss')
    pyplot.plot(range(1, epochs+1), test_losses, label='Test Loss')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.savefig('./ml_classification_pytorch/new-losses.png')

    pyplot.figure()
    pyplot.plot(range(1, epochs+1), accuracies, label='Accuracy')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.savefig('./ml_classification_pytorch/new-accuracy.png')