import os
import torchvision.datasets as datasets

ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
DATASET_DIR = os.path.join(ROOT, "Datasets/")

def Digit10(root=DATASET_DIR, domain="m", train=True, transform=None):
    if domain.lower() in ["m", "minst"]:
        dataset = datasets.MNIST(root=root, train=train, transform=transform, download=True)
    elif domain.lower() in ["u", "usps"]:
        dataset = datasets.USPS(root=os.path.join(root, "USPS"), train=train, transform=transform, download=True)
    elif domain.lower() in ["s", "svhn"]:
        dataset = datasets.SVHN(root=os.path.join(root, "SVHN"), split='train' if train else 'test', transform=transform, download=True)

    return dataset
