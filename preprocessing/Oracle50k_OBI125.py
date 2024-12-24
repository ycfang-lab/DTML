import os
from tqdm import tqdm
from PIL import Image
from torch.utils.data.dataset import Dataset

ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
ORACLE50K_DIR = os.path.join(ROOT, "Datasets/Oracle-50K")
OBI125_DIR = os.path.join(ROOT, "Datasets/OBI125")
classes = set(os.listdir(ORACLE50K_DIR)) & set(os.listdir(os.path.join(OBI125_DIR, "test")))
classes = list(classes)
dic = {}
for c in classes:
    dic[c] = len(dic)

with open("/media/admin1/sdb/lzc/Datasets/Oracle50K_OBI125/handprint_train.txt", "w") as f:
    for c in classes:  # 类别
        files = os.listdir(os.path.join(ORACLE50K_DIR, c))
        for file_name in files:
            file_path = os.path.join("Oracle-50K", c, file_name)
            f.write(file_path + f' {dic[c]} \n')

with open("/media/admin1/sdb/lzc/Datasets/Oracle50K_OBI125/scan_train.txt", "w") as f:
    for c in classes:  # 类别
        files = os.listdir(os.path.join(OBI125_DIR, "train", c))
        for file_name in files:
            file_path = os.path.join("OBI125", "train", c, file_name)
            f.write(file_path + f' {dic[c]} \n')

with open("/media/admin1/sdb/lzc/Datasets/Oracle50K_OBI125/scan_test.txt", "w") as f:
    for c in classes:  # 类别
        files = os.listdir(os.path.join(OBI125_DIR, "test", c))
        for file_name in files:
            file_path = os.path.join("OBI125", "test", c, file_name)
            f.write(file_path + f' {dic[c]} \n')