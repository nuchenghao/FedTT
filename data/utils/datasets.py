import re
import os
from pathlib import Path
from typing import List, Type, Dict, Callable
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import torch
import torchvision
from os import path


class BaseDataset(Dataset):
    def __init__(self) -> None:
        self.classes: List = None
        self.data = None
        self.targets = None
        self.data_transform = None
        self.target_transform = None

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]
        if self.data_transform is not None:
            data = self.data_transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return data, targets

    def __len__(self):
        return len(self.targets)


class FEMNIST(BaseDataset):
    def __init__(
            self,
            root,
            args=None,
            general_data_transform=None,
            general_target_transform=None,
            train_data_transform=None,
            train_target_transform=None,
    ) -> None:
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isfile(root / "data.npy") or not os.path.isfile(
                root / "targets.npy"
        ):
            raise RuntimeError(
                "run data/utils/run.py -d femnist for generating the data.npy and targets.npy first."
            )

        data = np.load(root / "data.npy")
        targets = np.load(root / "targets.npy")

        self.data = torch.from_numpy(data).float().reshape(-1, 1, 28, 28)
        self.targets = torch.from_numpy(targets).long()
        self.classes = list(range(62))
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class CINIC10(datasets.ImageFolder):
    def __init__(
            self,
            root,
            which: str
    ):
        super().__init__(root=f"{root}/{which}", transform=DATA_TRANSFORMS['cinic10'][which])


class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, which):
        super().__init__(root=root, train=True if which == 'train' else False, download=True,
                         transform=DATA_TRANSFORMS['cifar100'][which])



GLOVE_NAME = "glove.840B.300d.txt"
GLOVE_DIM = 300
VOCAB_NAME = "vocab.pkl"
WORDVEC_NAME = "wordvec.pkl"

def read_snli(data_dir, is_train):
    """将SNLI数据集解析为前提、假设和标签
        data_dir:为snli_1.0的目录
    """

    def extract_text(s):
        # 删除我们不会使用的信息
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # 用一个空格替换两个或多个连续的空格
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()

    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    if is_train:
        file_name = os.path.join(data_dir, 'snli_1.0_train.txt')
        with open(file_name, 'r') as f:
            rows = [row.split('\t') for row in f.readlines()[1:]]
        file_name = os.path.join(data_dir, 'snli_1.0_dev.txt')
        with open(file_name, 'r') as f:
            for row in f.readlines()[1:]:
                rows.append(row.split('\t'))
    else:
        file_name = os.path.join(data_dir, 'snli_1.0_test.txt')
        with open(file_name, 'r') as f:
            rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels

class SNLIDataset(torch.utils.data.Dataset):

    def __init__(self, root, split):
        """ Initialize SNLI dataset. 
        root: 数据根目录 (snli_1.0的目录）
        split: 数据集类型(train/test)。
        """

        assert split in ["train", "test"]
        self.root = root
        self.split = split 
        self.embed_dim = GLOVE_DIM # embed_dim: 词向量维度。
        self.n_classes = 3 # 分类类别数（3 类：蕴含、中性、矛盾）

        """ Read and store data from files. """
        self.classes = ["entailment","contradiction", "neutral"]

        # Read sentence and label data for current split from files.
        self.s1_sentences ,self.s2_sentences , self.targets = read_snli(self.root , self.split == "train")
        self.targets = np.array(self.targets)
        assert len(self.s1_sentences) == len(self.s2_sentences)
        assert len(self.s1_sentences) == len(self.targets)
        self.dataset_size = len(self.s1_sentences)
        print(f"Loaded {self.dataset_size} sentence pairs for {self.split}.")

        # If vocab exists on file, load it. Otherwise, give some prompt and raise  NotImplementedError
        vocab_path = os.path.join(self.root,  VOCAB_NAME) # 如果 vocab.pkl 文件存在，直接加载词汇表
        if os.path.isfile(vocab_path):
            print("Loading vocab.")
            with open(vocab_path, "rb") as vocab_file:
                vocab = pickle.load(vocab_file)
        else:
            print("construct vocab first")
            raise NotImplementedError
        print(f"Loaded vocab with {len(vocab)} words.")

        #加载 GloVe 词向量 Read in GLOVE vectors and store mapping from words to vectors. 
        self.word_vec = {}
        wordvec_path = os.path.join(self.root,  WORDVEC_NAME)
        if os.path.isfile(wordvec_path): # 如果 wordvec.pkl 存在，直接加载预处理的词向量映射。
            print("Loading word vector mapping.")
            with open(wordvec_path, "rb") as wordvec_file:
                self.word_vec = pickle.load(wordvec_file)
        else:
            print("map word to vector first")
            raise NotImplementedError
        print(f"Found {len(self.word_vec)}/{len(vocab)} words with glove vectors.")

        # Split each sentence into words, add start/stop tokens to the beginning/end of
        # each sentence, and remove any words which do not have glove embeddings.
        #  预处理句子
        assert "<s>" in vocab
        assert "</s>" in vocab
        assert "<s>" in self.word_vec
        assert "</s>" in self.word_vec
        for i in range(len(self.s1_sentences)):
            sent = self.s1_sentences[i]
            self.s1_sentences[i] = np.array(
                ["<s>"] +
                [word for word in sent.split() if word in self.word_vec] +
                ["</s>"]
            ) # 为每个句子添加 <s> 和 </s> 标记 ; 过滤无词向量的单词：移除句子中未在 self.word_vec 中的单词
        for i in range(len(self.s2_sentences)):
            sent = self.s2_sentences[i]
            self.s2_sentences[i] = np.array(
                ["<s>"] +
                [word for word in sent.split() if word in self.word_vec] +
                ["</s>"]
            ) # 每个句子被转换为 numpy 数组，包含有效单词及其顺序。

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx): # 让自定义对象支持直观的下标访问语法 ; dataloader的fetcher是Map式的
        """ Return a single element of the dataset. """
        # 根据索引 idx 返回一个数据样本（句子的词向量表示和标签）。
        # Encode sentences as sequence of glove vectors.
        sent1 = self.s1_sentences[idx]
        sent2 = self.s2_sentences[idx]
        # 将句子中的每个单词替换为对应的 GloVe 词向量。
        # 若句子长度为 L，输出形状为 (L, GLOVE_DIM) 的张量。
        s1_embed = np.zeros((len(sent1), GLOVE_DIM))
        s2_embed = np.zeros((len(sent2), GLOVE_DIM))
        for j in range(len(sent1)):
            s1_embed[j] = self.word_vec[sent1[j]]
        for j in range(len(sent2)):
            s2_embed[j] = self.word_vec[sent2[j]]
        s1_embed = torch.from_numpy(s1_embed).float()
        s2_embed = torch.from_numpy(s2_embed).float()

        # Convert targets to tensor.
        target = torch.tensor([self.targets[idx]]).long()

        return (s1_embed, s2_embed), target

    @property
    def n_words(self): # 返回词汇表中有效单词的数量（包含 GloVe 词向量的单词数）
        return len(self.word_vec)

def collate_pad_double(data_points):
    """ Pad data points with zeros to fit length of longest data point in batch. """
    if type(data_points[0][0]) == tuple:
        s1_embeds = [x[0][0] for x in data_points]
        s2_embeds = [x[0][1] for x in data_points]
        targets = [x[1] for x in data_points]

        s1_lens = np.array([sent.shape[0] for sent in s1_embeds])
        max_s1_len = np.max(s1_lens)
        s2_lens = np.array([sent.shape[0] for sent in s2_embeds])
        max_s2_len = np.max(s2_lens)

        bs = len(data_points)
        s1_embed = np.zeros((max_s1_len, bs, GLOVE_DIM))
        s2_embed = np.zeros((max_s2_len, bs, GLOVE_DIM))
        for i in range(bs):
            e1 = s1_embeds[i]
            e2 = s2_embeds[i]
            s1_embed[: len(e1), i] = e1.clone() 
            s2_embed[: len(e2), i] = e2.clone()

        inputs = [torch.from_numpy(s1_embed).float(), torch.from_numpy(s1_lens), torch.from_numpy(s2_embed).float(), torch.from_numpy(s2_lens)]

        # Convert targets to tensor.
        targets = torch.cat(targets)

        return inputs, targets

    else:
        index = torch.tensor([elem[0] for elem in data_points])
        data_points = [elem[1] for elem in data_points]
        s1_embeds = [x[0][0] for x in data_points]
        s2_embeds = [x[0][1] for x in data_points]
        targets = [x[1] for x in data_points]

        s1_lens = np.array([sent.shape[0] for sent in s1_embeds])
        max_s1_len = np.max(s1_lens)
        s2_lens = np.array([sent.shape[0] for sent in s2_embeds])
        max_s2_len = np.max(s2_lens)

        bs = len(data_points)
        s1_embed = np.zeros((max_s1_len, bs, GLOVE_DIM))
        s2_embed = np.zeros((max_s2_len, bs, GLOVE_DIM))
        for i in range(bs):
            e1 = s1_embeds[i]
            e2 = s2_embeds[i]
            s1_embed[: len(e1), i] = e1.clone() 
            s2_embed[: len(e2), i] = e2.clone()

        inputs = [torch.from_numpy(s1_embed).float(), torch.from_numpy(s1_lens), torch.from_numpy(s2_embed).float(), torch.from_numpy(s2_lens)]

        # Convert targets to tensor.
        targets = torch.cat(targets)

        return index , (inputs, targets)




imgsize = 224
def read_domainnet(dataset_path ,split ,selected_classes, selected_domain=["clipart","painting","sketch","real"]):
    """
    dataset_path：指向domainnet文件夹所在位置
    selected_domain：选中的域
    """
    data_paths = []
    data_labels = []
    for domain_name in selected_domain:
        split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                data_path, label = line.split(' ')
                if int(label) >= selected_classes:
                    continue
                data_path = path.join(dataset_path, data_path)
                data_paths.append(data_path)
                data_labels.append(int(label))
    return data_paths, data_labels

class DomainNet(Dataset):
    def __init__(self, root , which, selected_classes = 200):
        super(DomainNet, self).__init__()
        self.data_paths,self.targets = read_domainnet(root, which, selected_classes)
        self.classes = list(range(selected_classes))
        self.transforms = DATA_TRANSFORMS["domainnet"][which]

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.targets[index] 
        img = self.transforms(img) # 做图像变换

        return img, label

    def __len__(self):
        return len(self.data_paths)





DATASETS: Dict[str, Type[BaseDataset]] = {
    "femnist": FEMNIST,
    "cinic10": CINIC10,
    "cifar100": CIFAR100,
    "snli": SNLIDataset,
    "domainnet": DomainNet
}

DATA_NUM_CLASSES_DICT: Dict[str, int] = {
    "femnist": 62,
    "cinic10": 10,
    "cifar100": 100,
    "snli": 3,
    "domainnet": 200
}

DATASETS_COLLATE_FN ={
    "cinic10": None,
    "cifar100": None,
    "snli": collate_pad_double,
    "domainnet": None
}

DATA_TRANSFORMS = {
    "cinic10": {
        "train": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)),
        ]),
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)),
        ])
    },
    "cifar100": {
        "train": transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
    },
    "domainnet": {
        "train":transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
          [0.6605686,0.6431999,0.61347884],
          [0.33127954, 0.32900678, 0.34953416]),
        ]),
        "test": transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
          [0.6605686,0.6431999,0.61347884],
          [0.33127954, 0.32900678, 0.34953416]),
        ])
    }
}
