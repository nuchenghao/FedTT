import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=100):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])



class TextClassificationNet(nn.Module):
    def __init__(
        self,
        word_embed_dim,
        encoder_dim,
        n_enc_layers,
        dpout_model,
        dpout_fc,
        fc_dim,
        classes,
    ):
        super(TextClassificationNet, self).__init__()

        # Store settings.
        self.encoder_dim = encoder_dim
        self.n_enc_layers = n_enc_layers
        self.dpout_fc = dpout_fc
        self.fc_dim = fc_dim
        self.classes = classes

        # Construct encoder and classifier.
        self.encoder = RecurrentEncoder(
            n_enc_layers, word_embed_dim, encoder_dim, dpout_model
        )
        feature_multiplier = 4 
        self.inputdim = feature_multiplier * self.encoder_dim
        self.inputdim *= 2
        self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc), # 在训练过程中随机失活（即设置为零）神经元的一部分输出，从而减少神经元之间的依赖性，提高模型的泛化能力。表示每个神经元被随机失活的概率
                nn.Linear(self.inputdim, self.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.classes),
            )

    def forward(self, s1, s2):
        u = self.encoder(s1)
        v = self.encoder(s2)
        features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        output = self.classifier(features)
        return output


class RecurrentEncoder(nn.Module):

    def __init__(
        self, n_enc_layers, word_embed_dim, encoder_dim, dpout_model
    ):
        super(RecurrentEncoder, self).__init__()
        self.n_enc_layers = n_enc_layers
        self.word_embed_dim = word_embed_dim
        self.encoder_dim = encoder_dim
        self.dpout_model = dpout_model
        
        self.encoder = nn.RNN (
            self.word_embed_dim,
            self.encoder_dim,
            self.n_enc_layers,
            bidirectional=True,
            dropout=self.dpout_model, # 如果层数大于1，指定每层之间的dropout概率
        )

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch_size)
        # sent: Variable(seqlen x batch_size x word_embed_dim)
        sent, sent_len = sent_tuple

        self.encoder.flatten_parameters()

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        idx_unsort = np.argsort(idx_sort)

        sent_len, idx_sort = sent_len.copy(), idx_sort.copy()
        idx_sort = torch.from_numpy(idx_sort).to("cuda:0") 
        sent = sent.index_select(1, Variable(idx_sort))

        # Handling padding in Recurrent Networks ; 打包序列以优化RNN计算
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        sent_output = self.encoder(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).to("cuda:0") 
        sent_output = sent_output.index_select(1, Variable(idx_unsort))

        # Pooling
        emb = torch.max(sent_output, 0)[0]
        if emb.ndimension() == 3:
            emb = emb.squeeze(0)
            assert emb.ndimension() == 2

        return emb


MODEL_DICT: dict = {
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "TextClassificationNet": TextClassificationNet,
}
