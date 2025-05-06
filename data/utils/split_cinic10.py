import os
import shutil
import random
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import os
from collections import defaultdict
from torchvision.datasets import ImageFolder

def split_cinic10(root_dir, new_test_size=40000):
    id=0
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = ImageFolder(os.path.join(root_dir, 'train'), transform=transform)
    valid_dataset = ImageFolder(os.path.join(root_dir, 'valid'), transform=transform)
    test_dataset = ImageFolder(os.path.join(root_dir, 'test'), transform=transform)
    new_train_dir = os.path.join("../cinic10", 'train')
    new_test_dir = os.path.join("../cinic10", 'test')
    os.makedirs(new_train_dir, exist_ok=True)
    os.makedirs(new_test_dir, exist_ok=True)
    for class_name in train_dataset.classes:
        os.makedirs(os.path.join(new_train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(new_test_dir, class_name), exist_ok=True)
    for dataset in [train_dataset, valid_dataset]:
        for img, label in dataset:
            class_name = dataset.classes[label]
            img_path = os.path.join(new_train_dir, class_name, f"{id}.png")
            transforms.ToPILImage()(img).save(img_path)
            id+=1
    test_files = []
    for class_name in test_dataset.classes:
        class_dir = os.path.join(root_dir, 'test', class_name)
        test_files.extend([(os.path.join(class_dir, f), class_name) for f in os.listdir(class_dir)])
        random.shuffle(test_files)
        for i, (file_path, class_name) in enumerate(test_files):
            if i < new_test_size / 10:
                dest_dir = new_test_dir
            else:
                dest_dir = new_train_dir
            img_path = os.path.join(dest_dir, class_name, f"{id}.png")
            shutil.copy(file_path, img_path)
            id+=1
        test_files=[]
    print(id)
    
if __name__ == "__main__":
    root_dir = "../cinic10/raw"
    split_cinic10(root_dir)
    print("finished")
    train_dataset = ImageFolder(os.path.join("../cinic10", 'train'))
    test_dataset = ImageFolder(os.path.join("../cinic10", 'test'))
    train_count=defaultdict(int)
    for img, label in train_dataset:
        class_name = train_dataset.classes[label]
        train_count[class_name]+=1
    print(train_count)
    test_count=defaultdict(int)
    for img, label in test_dataset:
        class_name = test_dataset.classes[label]
        test_count[class_name]+=1
    print(test_count)
