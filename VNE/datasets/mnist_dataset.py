from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MnistDataset():
    def __init__(self, batch_size):
        
        self.batch_size = batch_size
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=self.transform)
        self.test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=self.transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
