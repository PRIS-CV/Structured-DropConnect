import torch
import numpy as np
import math
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init

class vbDropout(Module):
    def __init__(self, num, init_a=0.5, init_b=0.5):
        super(vbDropout, self).__init__()
        if init_a <= 0 or init_b <= 0:
            raise ValueError("Variational Bayes dropout parameters have to be larger than 0, but got init_a=" + str(init_a) 
                             + " and init_b=" + str(init_b))
        self.init_a = init_a
        self.init_b = init_b
        self.init_mu = np.log(init_a ** 1.25) - np.log(init_b ** 1.25)
        self.init_sigma = (1 / init_a + 1 / init_b) ** 0.75
        self.mu = Parameter(torch.Tensor([self.init_mu]))
        self.sigma = Parameter(torch.Tensor([self.init_sigma]))

    def forward(self, input):
        if self.training:
            c, n = input.size()
            epsilon = self.mu.repeat([c, 1]) + self.sigma.repeat([c, 1]) * torch.randn([c, n]).cuda()
            mask = torch.sigmoid(epsilon)

            out = input.mul(mask).div(torch.sigmoid(0.8 * self.mu.data))
        else:
            out = input

        return out

class vbdcLinear(Module):
    def __init__(self, in_features, out_features, rou, bias=True):
        super(vbdcLinear, self).__init__()
        if init_a <= 0 or init_b <= 0:
            raise ValueError("Variational Bayes dropconnect parameters have to be larger than 0, but got init_a=" + str(init_a) + " and init_b=" + str(init_b))
        self.in_features = in_features
        self.out_features = out_features
        self.rou = rou
        #Parameter()：将参数转换为可训练的类型，并且绑定在module的parameter列表当中，可以被训练、优化。
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        #这些都可以删掉，只要一个rou，就是伯努利分布
        '''
        self.init_a = init_a
        self.init_b = init_b
        self.init_mu = np.log(init_a ** 1.25) - np.log(init_b ** 1.25)
        self.init_sigma = (1 / init_a + 1 / init_b) ** 0.75
        self.mu = Parameter(torch.ones([out_features, in_features]) * self.init_mu)
        self.sigma = Parameter(torch.ones([out_features, in_features]) * self.init_sigma)
        '''


    #这个函数的作用是什么？需要保留吗？
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
        if self.training:
            #epsilon = self.mu + self.sigma.mul(torch.randn([self.out_features, self.in_features]).cuda())
            #mask = torch.sigmoid(epsilon)
            m = torch.ones(self.out_features, self.in_features) * (1 - self.rou) 
            mask = torch.bernoulli(m)

            # out = input.mul(mask).div(torch.sigmoid(0.8 * self.mu.data))
            out = F.linear(input, self.weight.mul(mask), self.bias)
        else:
            out = F.linear(input, self.weight, self.bias)

        return out

