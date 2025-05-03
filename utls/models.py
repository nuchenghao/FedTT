import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from peft import inject_adapter_in_model, LoraConfig, get_peft_model,get_peft_model_state_dict
import timm
from timm.models.vision_transformer import VisionTransformer, PatchEmbed
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
GLOVE_DIM = 300
class biRNN(nn.Module):
    def __init__(
        self,
        classes,
        word_embed_dim = GLOVE_DIM,
        encoder_dim = 2048,
        n_enc_layers = 1,
        dpout_model = 0.0,
        dpout_fc = 0.0 ,
        fc_dim = 512,
    ):
        super(biRNN, self).__init__()
        self.encoder_dim = encoder_dim
        self.n_enc_layers = n_enc_layers
        self.dpout_fc = dpout_fc
        self.fc_dim = fc_dim
        self.classes = classes
        self.encoder = RecurrentEncoder(
            n_enc_layers, word_embed_dim, encoder_dim, dpout_model
        )
        feature_multiplier = 4 
        self.inputdim = feature_multiplier * self.encoder_dim
        self.inputdim *= 2
        self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.classes),
            )
    def forward(self, inputs):
        s1,s2 = (inputs[0],inputs[1]),(inputs[2],inputs[3])
        u = self.encoder(s1)
        v = self.encoder(s2)
        features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        output = self.classifier(features)
        return output
class RecurrentEncoder(nn.Module):
    def __init__(self, n_enc_layers, word_embed_dim, encoder_dim, dpout_model):
        super().__init__()
        self.n_enc_layers = n_enc_layers
        self.word_embed_dim = word_embed_dim
        self.encoder_dim = encoder_dim
        self.dpout_model = dpout_model
        self.encoder = nn.RNN(
            input_size=self.word_embed_dim,
            hidden_size=self.encoder_dim,
            num_layers=self.n_enc_layers,
            bidirectional=True,
            dropout=dpout_model if n_enc_layers > 1 else 0,
            batch_first=False
        )
    def forward(self, sent_tuple):
        sent, sent_len = sent_tuple
        self.encoder.flatten_parameters()
        sorted_sent_len, idx_sort = torch.sort(sent_len, descending=True)
        idx_unsort = torch.argsort(idx_sort)
        sent = sent.index_select(1, idx_sort)
        sorted_lengths_cpu = sorted_sent_len.cpu()
        sent_packed = nn.utils.rnn.pack_padded_sequence(
            sent, sorted_lengths_cpu, 
            enforce_sorted=True
        )
        sent_output, _ = self.encoder(sent_packed)
        sent_output, _ = nn.utils.rnn.pad_packed_sequence(sent_output)
        sent_output = sent_output.index_select(1, idx_unsort)
        emb = torch.max(sent_output, dim=0)[0]
        return emb
class VPT_ViT(VisionTransformer):
    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 embed_layer=PatchEmbed,
                 norm_layer=None,
                 act_layer=None,
                 Prompt_Token_num=1,
                 VPT_type="Shallow",
                 basic_state_dict=None,
                 ):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         norm_layer=norm_layer, act_layer=act_layer)
        self.depth = depth
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)
        self.VPT_type = VPT_type
        if VPT_type == "Deep":
            self.Prompt_Tokens = nn.Parameter(torch.zeros(depth, Prompt_Token_num, embed_dim))
        else:
            self.Prompt_Tokens = nn.Parameter(torch.zeros(1, Prompt_Token_num, embed_dim))
    def New_CLS_head(self, new_classes=15):
        self.head = nn.Linear(self.embed_dim, new_classes)
    def Freeze(self):
        for name,param in self.named_parameters():
            if 'head' in name or 'Prompt' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    def UnFreeze(self):
        for param in self.parameters():
            param.requires_grad = True
    def obtain_prompt(self):
        prompt_state_dict = {'head': self.head.state_dict(),
                             'Prompt_Tokens': self.Prompt_Tokens}
        return prompt_state_dict
    def load_prompt(self, prompt_state_dict):
        try:
            self.head.load_state_dict(prompt_state_dict['head'], False)
        except:
            print('head not match, so skip head')
        else:
            print('prompt head match')
        if self.Prompt_Tokens.shape == prompt_state_dict['Prompt_Tokens'].shape:
            Prompt_Tokens = nn.Parameter(prompt_state_dict['Prompt_Tokens'].cpu())
            Prompt_Tokens.to(torch.device(self.Prompt_Tokens.device))
            self.Prompt_Tokens = Prompt_Tokens
        else:
            print('\n !!! cannot load prompt')
            print('shape of model req prompt', self.Prompt_Tokens.shape)
            print('shape of model given prompt', prompt_state_dict['Prompt_Tokens'].shape)
            print('')
    def forward_features(self, x, clsnum = 0):
        self.mid_out = []
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        Prompt_Token_num = self.Prompt_Tokens.shape[1]
        Prompt_Tokens = self.Prompt_Tokens.expand(x.shape[0], -1, -1)
        x = torch.cat((x, Prompt_Tokens), dim=1)
        num_tokens = x.shape[1]
        x = self.blocks(x)[:, :num_tokens - Prompt_Token_num]
        x = self.norm(x)
        return x
    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc_norm(x[:, 0, :])
        x = self.head(x)
        return x
def build_promptmodel(num_classes=2, edge_size=224,  patch_size=16,
                      Prompt_Token_num=10, VPT_type="Shallow", depth = 12):
    basic_model = timm.create_model('vit_base_patch16_224',pretrained=True,)
    model = VPT_ViT(img_size=edge_size, patch_size=patch_size, Prompt_Token_num=Prompt_Token_num,
                    VPT_type=VPT_type,depth=depth)
    model.load_state_dict(basic_model.state_dict(), strict = False)
    model.New_CLS_head(num_classes)
    model.Freeze()
    return model
lora_config = LoraConfig(
                r=8,
                lora_alpha=8,
                target_modules=['proj','mlp.fc2'],
                lora_dropout=0.1,
                bias="none",
            )
class vit(nn.Module):
    def __init__(self, num_classes , layer = 12):
        super(vit, self).__init__()
        self.back = build_promptmodel(num_classes=num_classes, edge_size=224, patch_size=16,Prompt_Token_num=0, depth = layer)
        self.back = get_peft_model(self.back, lora_config)
    def forward(self, x ):
        x = self.back(x)
        return x
MODEL_DICT: dict = {
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "biRNN": biRNN,
    "vit": vit
}
