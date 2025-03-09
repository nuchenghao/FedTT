import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
# from peft import inject_adapter_in_model, LoraConfig, get_peft_model,get_peft_model_state_dict
# import timm
# from timm.models.vision_transformer import VisionTransformer, PatchEmbed

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
            batch_first=False  # 保持与原始实现一致的seq_len-first格式
        )

    def forward(self, sent_tuple):
        sent, sent_len = sent_tuple

        # 确保所有操作在同一个设备上
        self.encoder.flatten_parameters()

        # 直接在GPU上进行排序（要求sent_len在GPU）
        sorted_sent_len, idx_sort = torch.sort(sent_len, descending=True)
        idx_unsort = torch.argsort(idx_sort)

        # 索引操作保持在同一设备
        sent = sent.index_select(1, idx_sort)

        # 处理长度数据到CPU（pack_padded_sequence要求）
        sorted_lengths_cpu = sorted_sent_len.cpu()

        # 打包序列时使用CPU长度
        sent_packed = nn.utils.rnn.pack_padded_sequence(
            sent, sorted_lengths_cpu, 
            enforce_sorted=True  # 已排序可启用加速
        )

        sent_output, _ = self.encoder(sent_packed)
        sent_output, _ = nn.utils.rnn.pad_packed_sequence(sent_output)

        # 恢复原始顺序
        sent_output = sent_output.index_select(1, idx_unsort)

        # 优化的池化操作
        emb = torch.max(sent_output, dim=0)[0]
        return emb



# class VPT_ViT(VisionTransformer):
#     def __init__(self, img_size=224,  # 输入图像的边长，默认为 224
#                  patch_size=16,  #  图像分割的块大小，默认为 16
#                  in_chans=3, # 输入通道数，默认为 3（RGB 图像）
#                  num_classes=1000, # 分类任务的类别数，默认为 1000
#                  embed_dim=768, # 嵌入维度，默认为 768
#                  depth=12, # 模型的深度，默认为 12
#                  num_heads=12, # 多头注意力机制中的头数，默认为 12
#                  mlp_ratio=4., # MLP 层的扩展比例，默认为 4.0
#                  qkv_bias=True, # 是否在 QKV 线性层中使用偏置，默认为 True
#                  drop_rate=0., # Dropout 概率，默认为 0.0
#                  attn_drop_rate=0., #  注意力层的 Dropout 概率，默认为 0.0
#                  drop_path_rate=0., #  DropPath 概率，默认为 0.0
#                  embed_layer=PatchEmbed, # 嵌入层，默认为 PatchEmbed
#                  norm_layer=None, # 归一化层，默认为 None
#                  act_layer=None, #  激活层，默认为 None
#                  Prompt_Token_num=1, # 提示（Prompt）的标记数量，默认为 1
#                  VPT_type="Shallow", # 提示（Prompt）的类型，可以是 'Shallow' 或 'Deep'，默认为 'Shallow'
#                  basic_state_dict=None, # 预训练模型的权重，默认为 None
#                  ):
#         """
#         继承自 VisionTransformer 的类，用于实现带有提示（Prompt）的 Vision Transformer 模型
#         """
#         # Recreate ViT
#         super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
#                          embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
#                          qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
#                          drop_path_rate=drop_path_rate, embed_layer=embed_layer,
#                          norm_layer=norm_layer, act_layer=act_layer)

#         self.depth = depth

#         # load basic state_dict
#         if basic_state_dict is not None:
#             self.load_state_dict(basic_state_dict, False) # 如果提供了预训练模型的权重，加载这些权重到当前模型中。strict=False 允许部分加载

#         self.VPT_type = VPT_type
#         # Prompt_Tokens: 定义提示（Prompt）的可训练参数。根据 VPT_type 的值，可以是深层（每个 Transformer 层一个提示）或浅层（一个全局提示）
#         if VPT_type == "Deep":
#             self.Prompt_Tokens = nn.Parameter(torch.zeros(depth, Prompt_Token_num, embed_dim))
#         else:  # "Shallow"
#             self.Prompt_Tokens = nn.Parameter(torch.zeros(1, Prompt_Token_num, embed_dim))

#     def New_CLS_head(self, new_classes=15):
#         #  添加一个新的分类头，用于指定的类别数。
#         self.head = nn.Linear(self.embed_dim, new_classes)

#     # 冻结模型的某些部分，防止在训练过程中更新。通常冻结除分类头和提示（Prompt）以外的所有参数。
#     def Freeze(self):
#         for name,param in self.named_parameters():
#             if 'head' in name or 'Prompt' in name:
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False

#     def UnFreeze(self):
#         for param in self.parameters():
#             param.requires_grad = True

#     #  获取当前模型的提示（Prompt）状态字典
#     def obtain_prompt(self):
#         prompt_state_dict = {'head': self.head.state_dict(),
#                              'Prompt_Tokens': self.Prompt_Tokens}
#         # print(prompt_state_dict)
#         return prompt_state_dict
#     # 加载提示（Prompt）状态字典到当前模型中。如果提示的形状不匹配，会打印错误信息。
#     def load_prompt(self, prompt_state_dict):
#         try:
#             self.head.load_state_dict(prompt_state_dict['head'], False)
#         except:
#             print('head not match, so skip head')
#         else:
#             print('prompt head match')

#         if self.Prompt_Tokens.shape == prompt_state_dict['Prompt_Tokens'].shape:

#             # device check
#             Prompt_Tokens = nn.Parameter(prompt_state_dict['Prompt_Tokens'].cpu())
#             Prompt_Tokens.to(torch.device(self.Prompt_Tokens.device))

#             self.Prompt_Tokens = Prompt_Tokens

#         else:
#             print('\n !!! cannot load prompt')
#             print('shape of model req prompt', self.Prompt_Tokens.shape)
#             print('shape of model given prompt', prompt_state_dict['Prompt_Tokens'].shape)
#             print('')
#     # 前向传播特征提取部分。根据 depth_cls 的值，可以添加中间分类头的输出。
#     def forward_features(self, x, clsnum = 0):
#         self.mid_out = []
#         x = self.patch_embed(x)
#         # print(x.shape,self.pos_embed.shape)
#         cls_token = self.cls_token.expand(x.shape[0], -1, -1)

#         # concatenate CLS token
#         x = torch.cat((cls_token, x), dim=1)
#         x = self.pos_drop(x + self.pos_embed)
                    
#         # self.VPT_type == "Shallow"
#         Prompt_Token_num = self.Prompt_Tokens.shape[1]

#         # concatenate Prompt_Tokens
#         Prompt_Tokens = self.Prompt_Tokens.expand(x.shape[0], -1, -1)
#         x = torch.cat((x, Prompt_Tokens), dim=1)
#         num_tokens = x.shape[1]
#         # Sequntially procees
#         x = self.blocks(x)[:, :num_tokens - Prompt_Token_num]

#         x = self.norm(x)
#         return x

#     def forward(self, x):

#         x = self.forward_features(x)

#         # use cls token for cls head
#         x = self.fc_norm(x[:, 0, :])  # fixme for old timm: x = self.pre_logits(x[:, 0, :])
#         x = self.head(x)
#         return x



# def build_promptmodel(num_classes=2, edge_size=224,  patch_size=16,
#                       Prompt_Token_num=10, VPT_type="Shallow", depth = 12):
    
    
#     basic_model = timm.create_model('vit_base_patch16_224',pretrained=True,)

#     model = VPT_ViT(img_size=edge_size, patch_size=patch_size, Prompt_Token_num=Prompt_Token_num,
#                     VPT_type=VPT_type,depth=depth)
#     model.load_state_dict(basic_model.state_dict(), strict = False)
#     model.New_CLS_head(num_classes)
#     model.Freeze()

#     return model

# lora_config = LoraConfig(
#                 r=8,
#                 lora_alpha=8,
#                 target_modules=['proj','mlp.fc2'],
#                 lora_dropout=0.1,
#                 bias="none",
#             )
# class vit(nn.Module):
#     def __init__(self, num_classes , layer = 12):
#         super(vit, self).__init__()
#         self.back = build_promptmodel(num_classes=num_classes, edge_size=224, patch_size=16,Prompt_Token_num=0, depth = layer)
#         self.back = get_peft_model(self.back, lora_config)

#     def forward(self, x ):
#         x = self.back(x)
#         return x




MODEL_DICT: dict = {
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "biRNN": biRNN,
    # "vit": vit
}
