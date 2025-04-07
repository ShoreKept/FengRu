import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""

#CLAM原注意力网络
class Attn_Net_Gated(nn.Module):  # zxb:输入是N*L; a,b,A是N*D
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)  # 逐元素相乘
        A = self.attention_c(A)  # N x n_classes
        return A, x

#并联的注意力网络，num_heads指示了多少个并联支路，支路的级联数均为1
class Attn_Net_Parralel(nn.Module):
    def __init__(self, L=1024, D=256, num_heads=8, dropout=False, n_classes=1):
        super(Attn_Net_Parralel, self).__init__()
        self.heads = nn.ModuleList([
            Attn_Net_Gated(L, D, dropout, n_classes)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        A_list = []
        for head in self.heads:
            A, _ = head(x)
            A_list.append(A)
        A_avg = torch.mean(torch.stack(A_list), dim=0)
        return A_avg, x

#二级串联的注意力网络
class Attn_Net_Double(nn.Module):
    def __init__(self, L=1024, D=256, M=512, dropout=False, n_classes=1):
        super(Attn_Net_Double, self).__init__()
        self.attention_a = [nn.Linear(L, M), nn.Tanh()]
        self.attention_b = [nn.Linear(L, M), nn.Sigmoid()]
        self.attention_c = [nn.Linear(M, D), nn.Tanh()]
        self.attention_d = [nn.Linear(M, D), nn.Sigmoid()]

        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
            self.attention_c.append(nn.Dropout(0.25))
            self.attention_d.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Sequential(*self.attention_c)
        self.attention_d = nn.Sequential(*self.attention_d)

        self.fc = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        c = self.attention_c(A)
        d = self.attention_d(A)
        A = c.mul(d)
        A = self.fc(A)  # N x n_classes
        return A, x


class Attn_Net_Gated_Without_Fc(nn.Module):  # zxb:输入是N*L; a,b,A是N*D
    def __init__(self, L=1024, D=256, dropout=False):
        super().__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]

        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        return A

class Attn_Net_Parallel_Fast(nn.Module):
    def __init__(self, L=1024, D=256, num_heads=8, dropout=False, n_classes=1):
        super(Attn_Net_Parallel_Fast, self).__init__()
        self.num_heads = num_heads

        self.attention_a = [nn.Linear(L, D * num_heads), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D * num_heads), nn.Sigmoid()]

        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = Parallel_Linear(D, n_classes, num_heads)

    def forward(self, x):
        # x: (K, L)
        K = x.shape[0]

        # a, b: (K, D * num_heads)
        a = self.attention_a(x)
        b = self.attention_b(x)

        # A: (K, num_heads, D)
        A = a * b
        A = A.view(K, self.num_heads, -1)

        # A: (K, num_heads, N)
        A = self.attention_c(A)

        A_avg = torch.mean(A, dim=1)
        
        return A_avg, x

class Attn_Net_Double_Without_Fc(nn.Module):
    def __init__(self, L=1024, D=256, M=512, dropout=False):
        super().__init__()
        self.attention_a = [nn.Linear(L, M), nn.Tanh()]
        self.attention_b = [nn.Linear(L, M), nn.Sigmoid()]
        self.attention_c = [nn.Linear(M, D), nn.Tanh()]
        self.attention_d = [nn.Linear(M, D), nn.Sigmoid()]

        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
            self.attention_c.append(nn.Dropout(0.25))
            self.attention_d.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Sequential(*self.attention_c)
        self.attention_d = nn.Sequential(*self.attention_d)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        c = self.attention_c(A)
        d = self.attention_d(A)
        A = c.mul(d)
        return A


class Attn_Net_Triple_Without_Fc(nn.Module):
    def __init__(self, L=1024, D=256, M=512, dropout=False):
        super().__init__()
        self.attention_a = [nn.Linear(L, M), nn.Tanh()]
        self.attention_b = [nn.Linear(L, M), nn.Sigmoid()]
        self.attention_c = [nn.Linear(M, M), nn.Tanh()]
        self.attention_d = [nn.Linear(M, M), nn.Sigmoid()]
        self.attention_e = [nn.Linear(M, D), nn.Tanh()]
        self.attention_f = [nn.Linear(M, D), nn.Sigmoid()]

        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
            self.attention_c.append(nn.Dropout(0.25))
            self.attention_d.append(nn.Dropout(0.25))
            self.attention_e.append(nn.Dropout(0.25))
            self.attention_f.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Sequential(*self.attention_c)
        self.attention_d = nn.Sequential(*self.attention_d)
        self.attention_e = nn.Sequential(*self.attention_e)
        self.attention_f = nn.Sequential(*self.attention_f)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        c = self.attention_c(A)
        d = self.attention_d(A)
        A = c.mul(d)
        e = self.attention_e(A)
        f = self.attention_f(A)
        A = e.mul(f)
        return A

#二级残差网络
class Attn_Net_Res2(nn.Module):
    def __init__(self, L=1024, D=256, M=512, dropout=False, n_classes=1):
        super(Attn_Net_Res2, self).__init__()
        self.Attn_Net = Attn_Net_Gated_Without_Fc(L=L, D=D, dropout=dropout)
        self.Attn_Net_Double = Attn_Net_Double_Without_Fc(L=L, D=D, M=M, dropout=dropout)

        # self.fc = nn.Linear(D, n_classes)
        self.parallel_linear = Parallel_Linear(in_features=D, out_features=n_classes, num_branches=2)

    def forward(self, x):
        A1 = self.Attn_Net(x)
        A2 = self.Attn_Net_Double(x)
        # A = A1 + A2
        # A = self.fc(A)  # N x n_classes
        A = torch.stack([A1, A2], dim=1)  #A: (K, 2, D)
        A = self.parallel_linear(A)  #A: (K, 2, n_classes)
        A = A.sum(dim=1)
        return A, x

#参数量多一倍的二级残差网络
class Attn_Net_Res2x2(nn.Module):
    def __init__(self, L=1024, D=256, M=512, dropout=False, n_classes=1):
        super(Attn_Net_Res2x2, self).__init__()
        self.Attn_Net = Attn_Net_Gated_Without_Fc(L=L, D=D, dropout=dropout)
        self.Attn_Net_Double = Attn_Net_Double_Without_Fc(L=L, D=D, M=M, dropout=dropout)
        self.Attn_Net_1 = Attn_Net_Gated_Without_Fc(L=L, D=D, dropout=dropout)
        self.Attn_Net_Double_1 = Attn_Net_Double_Without_Fc(L=L, D=D, M=M, dropout=dropout)

        # self.fc = nn.Linear(D, n_classes)
        self.parallel_linear = Parallel_Linear(in_features=D, out_features=n_classes, num_branches=4)

    def forward(self, x):
        A1 = self.Attn_Net(x)
        A2 = self.Attn_Net_Double(x)
        A1_1 = self.Attn_Net_1(x)
        A2_1 = self.Attn_Net_Double_1(x)
        A = torch.stack([A1, A2, A1_1, A2_1], dim=1)  #A: (K, 4, D)
        A = self.parallel_linear(A)  #A: (K, 4, n_classes)
        A = A.sum(dim=1)
        return A, x

#三级残差网络
class Attn_Net_Res3(nn.Module):
    def __init__(self, L=1024, D=256, M=512, dropout=False, n_classes=1):
        super().__init__()
        self.Attn_Net = Attn_Net_Gated_Without_Fc(L=L, D=D, dropout=dropout)
        self.Attn_Net_Double = Attn_Net_Double_Without_Fc(L=L, D=D, M=M, dropout=dropout)
        self.Attn_Net_Triple = Attn_Net_Triple_Without_Fc(L=L, D=D, M=M, dropout=dropout)

        # self.fc = nn.Linear(D, n_classes)
        self.parallel_linear = Parallel_Linear(in_features=D, out_features=n_classes, num_branches=3)

    def forward(self, x):
        A1 = self.Attn_Net(x)
        A2 = self.Attn_Net_Double(x)
        A3 = self.Attn_Net_Triple(x)
        A = torch.stack([A1, A2, A3], dim=1)  #A: (K, 3, D)
        A = self.parallel_linear(A)  #A: (K, 3, n_classes)
        A = A.sum(dim=1)
        return A, x

#参数量多一倍的三级残差网络
class Attn_Net_Res3x2(nn.Module):
    def __init__(self, L=1024, D=256, M=512, dropout=False, n_classes=1):
        super().__init__()
        self.Attn_Net = Attn_Net_Gated_Without_Fc(L=L, D=D, dropout=dropout)
        self.Attn_Net_Double = Attn_Net_Double_Without_Fc(L=L, D=D, M=M, dropout=dropout)
        self.Attn_Net_Triple = Attn_Net_Triple_Without_Fc(L=L, D=D, M=M, dropout=dropout)
        self.Attn_Net_1 = Attn_Net_Gated_Without_Fc(L=L, D=D, dropout=dropout)
        self.Attn_Net_Double_1 = Attn_Net_Double_Without_Fc(L=L, D=D, M=M, dropout=dropout)
        self.Attn_Net_Triple_1 = Attn_Net_Triple_Without_Fc(L=L, D=D, M=M, dropout=dropout)

        self.parallel_linear = Parallel_Linear(in_features=D, out_features=n_classes, num_branches=6)

    def forward(self, x):
        A1 = self.Attn_Net(x)
        A2 = self.Attn_Net_Double(x)
        A3 = self.Attn_Net_Triple(x)
        A1_1 = self.Attn_Net(x)
        A2_1 = self.Attn_Net_Double(x)
        A3_1 = self.Attn_Net_Triple(x)
        A = torch.stack([A1, A2, A3, A1_1, A2_1, A3_1], dim=1)  # A: (K, 6, D)
        A = self.parallel_linear(A)  # A: (K, 6, n_classes)
        A = A.sum(dim=1)
        return A, x

#自残差的一级连接
class Attn_Net_Res1(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        self.Attn_Net_1 = Attn_Net_Gated_Without_Fc(L=L, D=D, dropout=dropout)
        self.Attn_Net_2 = Attn_Net_Gated_Without_Fc(L=D, D=D, dropout=dropout)

        # self.fc = nn.Linear(D, n_classes)
        self.parallel_linear = Parallel_Linear(in_features=D, out_features=n_classes, num_branches=2)

    def forward(self, x):
        A1 = self.Attn_Net_1(x)
        A2 = self.Attn_Net_2(A1)
        # A = A1 + A2
        # A = self.fc(A)
        A = torch.stack([A1, A2], dim=1)  #A: (K, 2, D)
        A = self.parallel_linear(A)  #A: (K, 2, n_classes)
        A = A.sum(dim=1)
        return A, x


class Parallel_Linear(nn.Module):
    def __init__(self, in_features, out_features, num_branches):
        super(Parallel_Linear, self).__init__()
        # self.weight = nn.Parameter(torch.randn(num_branches, in_features, out_features)) / np.sqrt(in_features)
        # self.bias = nn.Parameter(torch.randn(num_branches, out_features)) / np.sqrt(in_features)

        self.num_branches = num_branches
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(num_branches, out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(num_branches, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_branches):
            nn.init.kaiming_uniform_(self.weight[i], a=np.sqrt(5))

        bound = 1 / np.sqrt(self.in_features)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x: (K, num_branches, in_features)
        out = torch.einsum('kci,coi->kco', x, self.weight)
        out = out + self.bias.unsqueeze(0)
        # out: (K, num_branches, out_features)
        return out

#第一种混联方式的注意力网络，先并后串
class Attn_Net_Mixed(nn.Module):
    def __init__(self, L=1024, D=256, M1=512, M2=256, num_heads=4, dropout=False, n_classes=1):
        super(Attn_Net_Mixed, self).__init__()
        self.num_heads = num_heads
        self.M1 = M1
        self.M2 = M2
        self.D = D

        self.attention_a = [nn.Linear(L, M1 * num_heads), nn.Tanh()]
        self.attention_b = [nn.Linear(L, M1 * num_heads), nn.Sigmoid()]
        self.attention_c = [nn.Linear(M2, D * num_heads), nn.Tanh()]
        self.attention_d = [nn.Linear(M2, D * num_heads), nn.Sigmoid()]

        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
            self.attention_c.append(nn.Dropout(0.25))
            self.attention_d.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Sequential(*self.attention_c)
        self.attention_d = nn.Sequential(*self.attention_d)

        self.fc1 = Parallel_Linear(M1, M2, num_heads)
        self.fc2 = Parallel_Linear(D, n_classes, num_heads)

    def forward(self, x):
        # K是patch数 L是输入维度 H是注意力的头数量
        N = x.shape[0]

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = A.view(N, self.num_heads, -1)
        A = self.fc1(A)
        A = torch.mean(A, dim=1)

        c = self.attention_c(A)
        d = self.attention_d(A)
        A = c.mul(d)
        A = A.view(N, self.num_heads, -1)
        A = self.fc2(A)
        A = torch.mean(A, dim=1)
        return A, x

#第二种混联方式的网络，先串后并
class Attn_Net_Mixed2(nn.Module):
    def __init__(self, L=1024, D=256, M=512, num_heads=4, dropout=False, n_classes=1):
        super(Attn_Net_Mixed2, self).__init__()
        self.heads = nn.ModuleList([
            Attn_Net_Double(L, D, M, dropout, n_classes)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        A_list = []
        for head in self.heads:
            A, _ = head(x)
            A_list.append(A)
        A_avg = torch.mean(torch.stack(A_list), dim=0)
        return A_avg, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""


class CLAM_SB(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 512, 256],
                          "big": [embed_dim, 512, 384]}  # zxb:256或384是注意力网络的隐藏层大小，输入是1*embed_dim
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]  # embed_dim*512的线性层
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)  # 返回一个数
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)  # 512*n_classes的线性分类器
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]  # n_classes个并行的512*2的线性分类器
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()  # 创建一个长为length的一维张量，元素全是1，类型为64位整数

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)  # zxb:将一维张量转化为2维张量，一行n列，n为A的元素数
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]  # 返回一个一维张量
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)  # 传进来的分类器
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        if self.k_sample < 1 or self.k_sample > A.size(-1):  # 我加的
            print(A.size(-1))
            raise ValueError(
                f"k_sample ({self.k_sample}) is out of the range of A's last dimension size ({A.size(-1)}).")

        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        # print("Test__:",logits.shape,p_preds.shape)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)
        logits = self.classifiers(M)  # N*num_classes N是x的特征维度
        Y_hat = torch.topk(logits, 1, dim=1)[1]  # 应该属于第几类
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict


class CLAM_MB(CLAM_SB):
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        nn.Module.__init__(self)
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Mixed2(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        self.attn_name = attention_net.__class__.__name__
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in
                           range(n_classes)]  # use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        
    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)
        # A: num_classes * patch_nums
        # h: patch_nums * 512
        # M: num_classes * 512

        logits = torch.empty(1, self.n_classes).float().to(M.device)
        # logits: 1 * num_classes

        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        # self.classifiers[c]: W_{c,i} 1 * 512
        # M[c]: 1 * 512
        # logits[0, c]: 1*1
        # logits: 1 * num_classes

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict

#添加了拓扑分数架构后的网络
class CLAM_MB_N(CLAM_SB):
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        nn.Module.__init__(self)
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes) #在此处修改你要使用的网络名
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in
                           range(n_classes)]  # use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False,
                patch_neighbors=None):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        N, K = A.shape

        # print(A.device, torch.is_tensor(patch_neighbors), patch_neighbors.device)

        # if not torch.is_tensor(patch_neighbors):
        #     patch_neighbors = torch.tensor(patch_neighbors, device=A.device, dtype=torch.long)
        # else:
        #     patch_neighbors = patch_neighbors.to(A.device)

        # 将无效邻居 (-1) 替换为 dummy 索引 K
        neighbor_idx = patch_neighbors.clone()
        neighbor_idx[neighbor_idx < 0] = K

        # 在 A 后面增加一列全 0 的 dummy 值，保证 dummy 索引合法
        A_padded = torch.cat([A, torch.zeros(N, 1, device=A.device, dtype=A.dtype)], dim=1)  # 形状 (N, K+1)

        # 利用 advanced indexing 一次性获取所有邻居的值
        # neighbor_idx 的形状为 (K, 4)，所以 A_neighbors 的形状为 (N, K, 4)
        A_neighbors = A_padded[:, neighbor_idx]

        # 将 A 每个 patch 的值扩展到 (N, K, 4)，便于与 A_neighbors 作差
        A_expanded = A.unsqueeze(2).expand(-1, K, 4)

        # 计算差值并经过 relu
        diff = A_expanded - A_neighbors
        relu_diff = torch.relu(diff)

        # 计算 exp( A_neighbor - relu(diff) )
        values = torch.exp(A_neighbors - relu_diff)

        # 构造 mask，dummy 索引 K 对应的位置为无效
        valid_mask = (neighbor_idx != K).unsqueeze(0)  # 形状 (1, K, 4)
        values_masked = values * valid_mask.to(A.dtype)

        # 沿邻居维度求和得到 (N, K)
        sum_values = values_masked.sum(dim=2)
        # 有效邻居个数，防止除以 0（无效邻居数最少为 1）
        counts = valid_mask.squeeze(0).sum(dim=1).clamp(min=1).to(A.dtype)
        T = sum_values / counts.unsqueeze(0)

        # 保留 T 的原始值
        T_raw = T.clone()
        # 对 T 做 softmax（沿 patch 维度，dim=1）
        T = F.softmax(T, dim=1)

        A = A + T

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)

        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict

#用于测试网络的代码
class CLAM_MB_NN(CLAM_SB):
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        nn.Module.__init__(self)
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        if gate:
            self.attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            self.attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)

        self.diffusion_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        # fc.append(attention_net)
        # self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in
                           range(n_classes)]  # use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False,
                patch_neighbors=None):
        h = self.fc(h)
        A, _ = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        D, _ = self.diffusion_net(h)
        D = torch.transpose(D, 1, 0)
        D = F.sigmoid(D)

        N, K = A.shape

        # print(A.device, torch.is_tensor(patch_neighbors), patch_neighbors.device)

        # if not torch.is_tensor(patch_neighbors):
        #     patch_neighbors = torch.tensor(patch_neighbors, device=A.device, dtype=torch.long)
        # else:
        #     patch_neighbors = patch_neighbors.to(A.device)

        # 将无效邻居 (-1) 替换为 dummy 索引 K
        neighbor_idx = patch_neighbors.clone()
        neighbor_idx[neighbor_idx < 0] = K

        # 在 A 后面增加一列全 0 的 dummy 值，保证 dummy 索引合法
        A_padded = torch.cat([A_raw, torch.zeros(N, 1, device=A_raw.device, dtype=A_raw.dtype)], dim=1)  # 形状 (N, K+1)
        D_padded = torch.cat([D, torch.zeros(N, 1, device=D.device, dtype=D.dtype)], dim=1)  # 形状 (N, K+1)

        # 利用 advanced indexing 一次性获取所有邻居的值
        # neighbor_idx 的形状为 (K, 4)，所以 A_neighbors 的形状为 (N, K, 4)
        A_neighbors = A_padded[:, neighbor_idx]
        D_neighbors = D_padded[:, neighbor_idx]

        # 将 A 每个 patch 的值扩展到 (N, K, 4)，便于与 A_neighbors 作差
        A_expanded = A_raw.unsqueeze(2).expand(-1, K, 4)

        # 计算差值并经过 relu
        diff = A_neighbors - A_expanded
        relu_diff = F.leaky_relu(diff)

        # 计算 exp( A_neighbor - relu(diff) )
        values = relu_diff * D_neighbors

        # 构造 mask，dummy 索引 K 对应的位置为无效
        valid_mask = (neighbor_idx != K).unsqueeze(0)  # 形状 (1, K, 4)
        values_masked = values * valid_mask.to(A_raw.dtype)

        # 沿邻居维度求和得到 (N, K)
        sum_values = values_masked.sum(dim=2)
        # 有效邻居个数，防止除以 0（无效邻居数最少为 1）
        counts = valid_mask.squeeze(0).sum(dim=1).clamp(min=1).to(A_raw.dtype)
        T = sum_values / counts.unsqueeze(0)

        # 保留 T 的原始值
        T_raw = T
        # 对 T 做 softmax（沿 patch 维度，dim=1）
        T = F.softmax(T, dim=1)
        # T = F.sigmoid(T)

        A = A + T

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)

        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict
