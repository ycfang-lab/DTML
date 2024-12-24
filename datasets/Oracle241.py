import os
from tqdm import tqdm
from PIL import Image
from torch.utils.data.dataset import Dataset

ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
ORACLE241_DIR = os.path.join(ROOT, "Datasets/Oracle241")


class Oracle241(Dataset):
    """
    Unsupervised Structure-Texture Separation Network for Oracle Character Recognition
    """
    def __init__(self, root=ORACLE241_DIR, domain="h", train=True, transform=None, transform2=None, mode="RGB", all_source=False, preloading=True):
        self.imgs = []
        self.labels = []
        self.transform = transform
        self.transform2 = transform2
        self.mode = mode
        self.preloading = preloading

        domain = "handprint" if domain.lower() in ["h", "handprint"] else "scan"
        if all_source:
            file_name = os.path.join(root, "image_list", "{}.txt".format(domain))
        else:
            file_name = os.path.join(root, "split", "{}_{}.txt".format(domain, "train" if train else "test"))
        with open(file_name) as f:
            for line in tqdm(f.readlines()):
                img_path, label = line.strip().rsplit(maxsplit=1)
                if self.preloading:
                    self.imgs.append(Image.open(os.path.join(root, img_path)).convert(self.mode))
                else:
                    self.imgs.append(os.path.join(root, img_path))
                self.labels.append(int(label))
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img = self.imgs[index]
        if not self.preloading:
            img = Image.open(img).convert(self.mode)
        label = self.labels[index]
        if self.transform:
            img1 = self.transform(img)
        if self.transform and self.transform2:
            img2 = self.transform2(img)
            return img1, img2, label
        return img1, label


if __name__ == "__main__":
    data = Oracle241(domain="s", train=False)
    print(data[1])
     