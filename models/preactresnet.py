'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

track_running_stats=True
affine=True
normal_func = nn.BatchNorm2d

# track_running_stats=False
# affine=True
# normal_func = nn.InstanceNorm2d



if not track_running_stats:
    print('BN track False')

#####################
## Model classes
#####################
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation='ReLU', softplus_beta=1):
        super(PreActBlock, self).__init__()
        self.bn1 = normal_func(in_planes, track_running_stats=track_running_stats, affine=affine)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = normal_func(planes, track_running_stats=track_running_stats, affine=affine)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
        if activation == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
            print('ReLU')
        elif activation == 'Softplus':
            self.relu = nn.Softplus(beta=softplus_beta, threshold=20)
            print('Softplus')
        

    def forward(self, x):
        out = self.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, activation='ReLU', softplus_beta=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = normal_func(in_planes, track_running_stats=track_running_stats, affine=affine)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = normal_func(planes, track_running_stats=track_running_stats, affine=affine)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = normal_func(planes, track_running_stats=track_running_stats, affine=affine)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, normalize = False, normalize_only_FN = False, scale = 15, activation='ReLU', softplus_beta=1, return_out=False):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.normalize = normalize
        self.normalize_only_FN = normalize_only_FN
        self.scale = scale
        self.return_out = return_out

        self.activation = activation
        self.softplus_beta = softplus_beta

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = normal_func(512 * block.expansion, track_running_stats=track_running_stats, affine=affine)

        if self.normalize:
            self.linear = nn.Linear(512*block.expansion, num_classes, bias=False)
        else:
            self.linear = nn.Linear(512*block.expansion, num_classes)


        if activation == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
            print('ReLU')
        elif activation == 'Softplus':
            self.relu = nn.Softplus(beta=softplus_beta, threshold=20)
            print('Softplus')
        print('Use activation of ' + activation)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, 
                activation=self.activation, softplus_beta=self.softplus_beta))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out) # 128, 64, 32, 32
        out = self.layer2(out) # 128, 128, 16, 16
        out = self.layer3(out) # 128, 256, 8, 8
        out = self.layer4(out) # 128, 512, 4, 4
        out = self.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # if self.normalize_only_FN:
        #     out = F.normalize(out, p=2, dim=1)

        if self.normalize:
            out = F.normalize(out, p=2, dim=1) * self.scale
            for _, module in self.linear.named_modules():
                if isinstance(module, nn.Linear):
                    module.weight.data = F.normalize(module.weight, p=2, dim=1)
        if self.return_out:
            return self.linear(out), out
        else:
            return self.linear(out)

class PreActResNet_twobranch_DenseV1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, activation='ReLU', softplus_beta=1, 
        out_dim=10, use_BN=False, along=False):
        super(PreActResNet_twobranch_DenseV1, self).__init__()
        self.in_planes = 64

        self.activation = activation
        self.softplus_beta = softplus_beta
        self.along = along
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = normal_func(512 * block.expansion, track_running_stats=track_running_stats, affine=affine)


        self.linear = nn.Linear(512*block.expansion, num_classes)

        if use_BN:
            self.dense = nn.Sequential(
                nn.Linear(512*block.expansion, 256*block.expansion),
                nn.BatchNorm1d(256*block.expansion),
                nn.ReLU(),
                nn.Linear(256*block.expansion, out_dim)
                )
            print('with BN')
        else:
            self.dense = nn.Sequential(
                nn.Linear(512*block.expansion, 256*block.expansion),
                nn.ReLU(),
                nn.Linear(256*block.expansion, out_dim)
                )

        if activation == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
            print('ReLU')
        elif activation == 'Softplus':
            self.relu = nn.Softplus(beta=softplus_beta, threshold=20)
            print('Softplus')
        print('Use activation of ' + activation)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, 
                activation=self.activation, softplus_beta=self.softplus_beta))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        classification_return = self.linear(out)
        if self.along:
            evidence_return = self.dense(out)
        else:
            evidence_return = self.linear(out) + self.dense(out)
        
        return classification_return, evidence_return

class PreActResNet_threebranch_DenseV1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, activation='ReLU', softplus_beta=1, 
        out_dim=10, use_BN=False, along=False):
        super(PreActResNet_threebranch_DenseV1, self).__init__()
        self.in_planes = 64

        self.activation = activation
        self.softplus_beta = softplus_beta
        self.along = along
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = normal_func(512 * block.expansion, track_running_stats=track_running_stats, affine=affine)


        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear_aux = nn.Linear(512*block.expansion, num_classes)

        if use_BN:
            self.dense = nn.Sequential(
                nn.Linear(512*block.expansion, 256*block.expansion),
                nn.BatchNorm1d(256*block.expansion),
                nn.ReLU(),
                nn.Linear(256*block.expansion, out_dim)
                )
            print('with BN')
        else:
            self.dense = nn.Sequential(
                nn.Linear(512*block.expansion, 256*block.expansion),
                nn.ReLU(),
                nn.Linear(256*block.expansion, out_dim)
                )

        if activation == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
            print('ReLU')
        elif activation == 'Softplus':
            self.relu = nn.Softplus(beta=softplus_beta, threshold=20)
            print('Softplus')
        print('Use activation of ' + activation)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, 
                activation=self.activation, softplus_beta=self.softplus_beta))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        classification_return = self.linear(out)
        aux_return = self.linear_aux(out)
        if self.along:
            evidence_return = self.dense(out)
        else:
            evidence_return = self.linear(out) + self.dense(out)
        
        return classification_return, evidence_return, aux_return        

class PreActResNet_twobranch_DenseV1Multi(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, activation='ReLU', softplus_beta=1, 
        out_dim=10, use_BN=False, along=False):
        super(PreActResNet_twobranch_DenseV1Multi, self).__init__()
        self.in_planes = 64

        self.activation = activation
        self.softplus_beta = softplus_beta
        self.along = along
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = normal_func(512 * block.expansion, track_running_stats=track_running_stats, affine=affine)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        ### Multi
        self.conv_layer1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(),
                                         nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU())
                                         #nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0))

        self.conv_layer2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(),
                                         nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU())
                                         #nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0))

        self.conv_layer3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU())
                                         #nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0))

        if use_BN:
            self.dense = nn.Sequential(
                nn.Linear(512*block.expansion*4, 512*block.expansion),
                nn.BatchNorm1d(512*block.expansion),
                nn.ReLU(),
                nn.Linear(512*block.expansion, out_dim)
                )
            print('with BN')
        else:
            self.dense = nn.Sequential(
                nn.Linear(512*block.expansion*4, 512*block.expansion),
                nn.ReLU(),
                nn.Linear(512*block.expansion, out_dim)
                )

        if activation == 'ReLU':
            # self.relu = nn.ReLU(inplace=True)
            self.relu = nn.ReLU()
            print('ReLU')
        elif activation == 'Softplus':
            self.relu = nn.Softplus(beta=softplus_beta, threshold=20)
            print('Softplus')
        print('Use activation of ' + activation)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, 
                activation=self.activation, softplus_beta=self.softplus_beta))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bs = x.size(0)
        out = self.conv1(x)

        out = self.layer1(out) # 128, 64, 32, 32
        out_layer1 = self.conv_layer1(out) # 128, 512, 4, 4
        out_layer1 = F.avg_pool2d(out_layer1, 4).view(bs, -1) # 128, 512

        out = self.layer2(out) # 128, 128, 16, 16
        out_layer2 = self.conv_layer2(out) # 128, 512, 4, 4
        out_layer2 = F.avg_pool2d(out_layer2, 4).view(bs, -1) # 128, 512

        out = self.layer3(out) # 128, 256, 8, 8
        out_layer3 = self.conv_layer3(out) # 128, 512, 4, 4
        out_layer3 = F.avg_pool2d(out_layer3, 4).view(bs, -1) # 128, 512

        out = self.layer4(out) # 128, 512, 4, 4
        out = self.relu(self.bn(out))
        out = F.avg_pool2d(out, 4).view(bs, -1) # 128, 512

        out_all = torch.cat((out, out_layer1, out_layer2, out_layer3), dim=1) # 128, 512 x 4

        classification_return = self.linear(out)

        if self.along:
            evidence_return = self.dense(out_all)
        else:
            evidence_return = classification_return + self.dense(out_all)
        
        return classification_return, evidence_return

class PreActResNet_twobranch_DenseV2(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, activation='ReLU', softplus_beta=1, 
        out_dim=10, use_BN=False, along=False):
        super(PreActResNet_twobranch_DenseV2, self).__init__()
        self.in_planes = 64

        self.activation = activation
        self.softplus_beta = softplus_beta
        self.along = along
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = normal_func(512 * block.expansion, track_running_stats=track_running_stats, affine=affine)


        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.dense1 = nn.Linear(512*block.expansion, 256*block.expansion)
        self.dense2 = nn.Linear(256*block.expansion, out_dim)
        self.IN = nn.InstanceNorm1d(1, affine=False, track_running_stats=False)
        print('with IN! ! !')
        

        if activation == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
            print('ReLU')
        elif activation == 'Softplus':
            self.relu = nn.Softplus(beta=softplus_beta, threshold=20)
            print('Softplus')
        print('Use activation of ' + activation)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, 
                activation=self.activation, softplus_beta=self.softplus_beta))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        evidence_return = self.dense1(out)
        evidence_return = self.IN(evidence_return.unsqueeze(dim=1))
        evidence_return = self.dense2(F.relu(evidence_return))
        
        return self.linear(out), evidence_return


#####################
## Architectures
#####################
def PreActResNet18(num_classes=10, normalize = False, normalize_only_FN = False, scale = 15, activation='ReLU', softplus_beta=1, return_out=False):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes=num_classes, normalize = normalize
        , normalize_only_FN = normalize_only_FN, scale = scale, activation=activation, softplus_beta=softplus_beta, return_out=return_out)

def PreActResNet18_twobranch_DenseV1(num_classes=10, activation='ReLU', softplus_beta=1, out_dim=10, use_BN=False, along=False):
    return PreActResNet_twobranch_DenseV1(PreActBlock, [2,2,2,2], num_classes=num_classes, activation=activation, softplus_beta=softplus_beta, 
        out_dim=out_dim, use_BN=use_BN, along=along)

def PreActResNet18_threebranch_DenseV1(num_classes=10, activation='ReLU', softplus_beta=1, out_dim=10, use_BN=False, along=False):
    return PreActResNet_threebranch_DenseV1(PreActBlock, [2,2,2,2], num_classes=num_classes, activation=activation, softplus_beta=softplus_beta, 
        out_dim=out_dim, use_BN=use_BN, along=along)

def PreActResNet18_twobranch_DenseV1Multi(num_classes=10, activation='ReLU', softplus_beta=1, out_dim=10, use_BN=False, along=False):
    return PreActResNet_twobranch_DenseV1Multi(PreActBlock, [2,2,2,2], num_classes=num_classes, activation=activation, softplus_beta=softplus_beta, 
        out_dim=out_dim, use_BN=use_BN, along=along)

def PreActResNet18_twobranch_DenseV2(num_classes=10, activation='ReLU', softplus_beta=1, out_dim=10, use_BN=False, along=False):
    return PreActResNet_twobranch_DenseV2(PreActBlock, [2,2,2,2], num_classes=num_classes, activation=activation, softplus_beta=softplus_beta, 
        out_dim=out_dim, use_BN=use_BN, along=along)

# def PreActResNet34():
#     return PreActResNet(PreActBlock, [3,4,6,3])

# def PreActResNet50():
#     return PreActResNet(PreActBottleneck, [3,4,6,3])

# def PreActResNet101():
#     return PreActResNet(PreActBottleneck, [3,4,23,3])

# def PreActResNet152():
#     return PreActResNet(PreActBottleneck, [3,8,36,3])
