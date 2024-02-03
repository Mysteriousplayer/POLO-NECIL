import torch.nn as nn
import torch
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module

class network(nn.Module):
    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.fc = nn.Linear(512, numclass, bias=True)

    def forward(self, input):
        feature = self.feature(input)
        x = self.fc(feature)
        return x, feature

    def feature_extractor(self,inputs):
        return self.feature(inputs)

class network_cosine(nn.Module):
    def __init__(self, numclass, feature_extractor):
        super(network_cosine, self).__init__()
        self.feature = feature_extractor
        #self.fc = nn.Linear(512, numclass, bias=True)
        self.fc = CosineLinear(512, numclass)
    def forward(self, input):
        x = self.feature(input)
        x = self.fc(x)
        return x

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        #bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features
        sigma = self.fc.sigma.data
        #self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc = CosineLinear(in_feature, numclass)
        self.fc.weight.data[:out_feature] = weight[:out_feature]
        #self.fc.bias.data[:out_feature] = bias[:out_feature]
        self.fc.sigma.data = sigma

    def feature_extractor(self,inputs):
        return self.feature(inputs)

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.feature.parameters(), 'lr': lr},
                {'params': self.fc.parameters(), 'lr': lrp},
                ]

class CosineLinear(Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) #for initializaiton of sigma

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), \
                F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out

class SplitCosineLinear(Module):
    #consists of two fc layers and concatenate their outputs
    def __init__(self, in_features, out_features1, out_features2, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out = torch.cat((out1, out2), dim=1) #concatenate along the channel
        if self.sigma is not None:
            out = self.sigma * out
        return out