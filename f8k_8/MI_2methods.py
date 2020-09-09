# -*- coding:utf-8 -*-
"""
@author:Echo
@file:MI_2methods.py
@time:2020/7/20 16:15
"""
import os
import argparse
import time
from random import seed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

class contact(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(2*1024, 1024)
        self.l1 = nn.Linear(1024, 1024)
        self.l2 = nn.Linear(1024, 1)

    def forward(self, m1, m2):
        # print(Text)
        # print(Image)
        h = torch.cat((m1, m2),dim=1)  # [100, 2048]
        # print(h.shape[0])
        h = F.elu(self.l0(h))  # [100,1024]
        h = F.elu(self.l1(h))
        # print(h.shape[0])
        # print(self.l2(h).shape[0])
        return self.l2(h)  # [100,1]


class DIMLoss(nn.Module):   #T网络
    def __init__(self):
        super().__init__()
        self.contact = contact()

    # loss 最终输出的是 (None,1), 将 Text, Image 合并成一个score.
    def forward(self, Text, Image, Text_prime):
        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        Ej = -F.softplus(-self.contact(Image, Text)).mean()  # joint  同时输入的Text样本和Image样本作为联合分布  score of anchor and positive
        Em = F.softplus(self.contact(Image, Text_prime)).mean()  # marginal 打乱其中一个样本的顺序得到prime,和另一个样本一起作为边缘分布乘积   score of anchor and negative
        # Ej-Em是JSD公式
        mine_estimate=Ej - Em
        loss = -mine_estimate   # -1 * Jensen-Shannon MI estimator

        return loss



class KLLoss(nn.Module):  #T网络
    def __init__(self):
        super().__init__()
        self.contact = contact()

    def forward(self, Text, Image, Text_prime):
        T_joint=self.contact(Image, Text)  # joint  同时输入的Text样本和Image样本作为联合分布
        T_marginal= self.contact(Image, Text_prime)  # marginal 打乱其中一个样本的顺序得到prime,和另一个样本一起作为边缘分布乘积
        T_marginal_exp = torch.exp(T_marginal)
        avg_et = 1.0
        avg_et = 0.99 * avg_et + 0.01 * torch.mean(T_marginal_exp)
        # mine_estimate_unbiased = torch.mean(T_joint) - (1/avg_et).detach() * torch.mean(T_marginal_exp)
        mine_estimate = torch.mean(T_joint) - (torch.mean(T_marginal_exp) / avg_et).detach() * torch.log((torch.mean(T_marginal_exp)))# 参数a并不参与梯度的计算,detach
        loss = -1. * mine_estimate  # -1 * KL MI estimator
        return loss

#
# def trans_to_cuda(variable):
#    if torch.cuda.is_available():
#        return variable.cuda()
#    else:
#        return variable


def load_vec(file, threshold=0, dtype='float'):  # 读取txt文件格式的向量
    words = []
    matrix = np.empty((100, 1024), dtype=dtype)
    i = 0
    for line in file.readlines():
        word = line.strip().split(" ")[0]
        vec = [float(i) for i in line.strip().split(" ")[1:]]
        words.append(word)
        # print(len(vec))
        matrix[i] = np.array(vec)
        i += 1
        if i == 100:
            break
    return (words, matrix)


def Generate_data():
    txt_file = open('dd100text.txt', 'r', encoding='utf-8', errors='surrogateescape') #文本样本
    image_file = open('dd100image.txt', 'r', encoding='utf-8', errors='surrogateescape')#图像样本
    text, vt = load_vec(txt_file)  # 载入txt文件格式的文本向量
    image, vi = load_vec(image_file)  # 载入txt文件格式的图像向量
    Text = torch.tensor(vt, dtype=torch.float32)
    Image = torch.tensor(vi, dtype=torch.float32)
    return Text, Image


##########在大数据集上得使用minibatch实验###########
#def random_mini_batches(Text,minibatch_size,seed):
   # np.random.seed(seed)
    #Text = Text.transpose(1, 0)  # Text:[1024,100]
    #Image = Image.transpose(1, 0)

    #Text=float(str.strip(str(Text.detach().numpy())))
    #Image=float(str.strip(str(Image.detach().numpy())))
    #shuffled_Text=[]
    #t = Text.shape[1]  # 100
    #print(t)

    #permutation = list(np.random.permutation(t))  # 生成的0-t-1随机顺序的值作为下标 0~99
    #print(permutation)
    #shuffled_Text = Text[:, permutation]
    #shuffled_Image = Image[:, permutation]

    # 按照minibatch_size分割数据集
    #num_minibatch = np.math.floor(t / minibatch_size)  # 得到总的子集数目，math.floor表示向下取整
    #for k in range(0, num_minibatch):
        #mini_Text = shuffled_Text[:, k * minibatch_size:(k + 1) * minibatch_size]  #: 表示取所有行，a：b表示取a列到b-1列
        #mini_Image = shuffled_Image[:, k * minibatch_size:(k + 1) * minibatch_size]
        #mini_Text= mini_Text.numpy().tolist()
       # mini_Image = mini_Image.numpy().tolist()
        #mini_Text.append(mini_Text)#100个
        #print(mini_Text)
       # mini_Image.append(mini_Image)

    #if t % minibatch_size != 0:  # 还有剩余不够一个size的数据，剩下作为一个batch
    #    mini_Text = shuffled_Text[:, minibatch_size * num_minibatch:]
    #    #mini_Image = shuffled_Image[:, minibatch_size * num_minibatch:]
    #    mini_Text = mini_Text.numpy().tolist()
    #    #mini_Image = mini_Image.numpy().tolist()
    #    mini_Text.append(mini_Text)
        #mini_Image.append(mini_Image)

    #return mini_Text

if __name__ == '__main__':

    ################使用gpu###########################
    # parser = argparse.ArgumentParser(description='image and text mutual information')
    # Path Arguments
    # parser.add_argument('--cuda', dest='cuda', action='store_true',
    # help='use CUDA')
    # parser.add_argument('--device_id', type=str, default='0')
    # parser.add_argument('--seed', type=int, default=1111,
    # help='random seed')
    # args = parser.parse_args()
    # print(vars(args))

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    # if torch.cuda.is_available():
    # if not args.cuda:
    # print("WARNING: You have a CUDA device, "
    # "so you should probably run with --cuda")
    # else:
    # torch.cuda.manual_seed(args.seed)
    start_time = time.time()
    print("Training...")

    print("Loaded data!")
    Text, Image = Generate_data()
    result = []
    num = []
    loss_fn = KLLoss()             # loss_fn可选KLLoss()或者 DIMLoss()
    loss_fn.zero_grad()
    optimizer = SGD(loss_fn.parameters(), lr=1e-4)
    #seed=0
    for epoch in range(500):
#######制作minibatch  从100个文本样本和100个图像样本中各随机抽minibatch_size个——>minibatch_size个文本向量，minibatch_size个图像向量
        #seed=seed + 1
        #mini_Text=random_mini_batches(Text,20,0)  #type list
        #mini_Image=random_mini_batches(Image,20,0)
        #print(mini_Text)
        #mini_Text = torch.tensor(mini_Text)
        #mini_Image = torch.tensor(mini_Image)
        #mini_Text.transpose(1, 0)
        #mini_Image.transpose(1, 0)
        #for batch in range(10):

        # 批量中第一个放在最后一位，其他的顺着向上移动一位. 相当于打乱了batch中text的顺序
           Text_prime = torch.cat((Text[1:], Text[0].unsqueeze(0)), dim=0)  # 打乱text向量样本  100X1024
           #print(Text_prime.size())
           #Image_prime = torch.cat((mini_Image[1:], mini_Image[0].unsqueeze(0)), dim=0)# 打乱Image向量样本
           loss1 = loss_fn(Text, Image, Text_prime)
           #loss2 = loss_fn(Text, Image, Image_prime)
           loss = loss1
           loss.backward()
           optimizer.step()

           if (epoch+1) % (100) == 0:

            print(('[%d]  mutual information: %.8f ' %(epoch, loss)))
            num.append(epoch + 1)  # type list /int
            result.append(float(str(loss.detach().numpy())))  # type list /str-->float


    # print(num) #[1,2,3...100]
    # print(result)
    time_end = time.time()
    print(('totally cost:%.4f' % (time_end - start_time)))
    plt.clf()
    print('开始画图......')
    #plt.figure(figsize=(10, 10))  # 画布大小
    plt.title('mutual information')
    plt.xlabel('epoch')
    plt.ylabel('MI')
    plt.plot(num, result, linewidth=1, color='b')  # 将列表传递给plot,并设置线宽，设置颜色，默认为蓝色

    plt.savefig("Mutual_Info.jpg", dpi=100)
    print("Mutual Information final:", result[-1])


