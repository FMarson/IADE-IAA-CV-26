import torchvision
import torch
import sklearn.svm
from matplotlib import pyplot

if __name__ == '__main__':
    mnist = torchvision.datasets.MNIST(root='./mnist/', download=True, train=True, transform=torchvision.transforms.ToTensor())

    mnist_image = list()
    mnist_target = list()
    for image, target in mnist:
        mnist_image.append(image)
        mnist_target.append(torch.tensor([target]))

    mnist_image = torch.cat(mnist_image)
    mnist_target = torch.cat(mnist_target)

    print(mnist_image.shape, mnist_target.shape)

    split_n = int(len(mnist_image) * 0.8)

    train_image = mnist_image[:split_n].flatten(start_dim=1).detach().numpy()
    test_image = mnist_image[split_n:].flatten(start_dim=1).detach().numpy()
    train_target = mnist_target[:split_n].detach().numpy()
    test_target = mnist_target[split_n:].detach().numpy()

    print(train_image.shape, train_target.shape)

    svm = sklearn.svm.SVC(kernel='linear').fit(train_image, train_target)
    y_pred = svm.predict(test_image)

    confusion_matrix = sklearn.metrics.confusion_matrix(y_true=test_target, y_pred=y_pred)
    print(confusion_matrix)