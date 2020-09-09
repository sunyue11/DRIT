# -*- coding:utf-8 -*-
"""
@author:Echo
@file:drawer.py
@time:2020/7/17 16:02
"""

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import gensim
import matplotlib as mpl

from matplotlib.pyplot import MultipleLocator
#从pyplot导入MultipleLocator类，这个类用于设置刻度间隔

def plot_with_labels(low_dim_embs, labels, filename):   # 绘制词向量图
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    print('绘制向量中......')
    plt.figure(figsize=(10, 10))  # 画布大小
    #plt.xlim(-5, 10)
    # 把x轴的刻度范围设置为-0.5到11
    #plt.ylim(-10, 10)
    # 把y轴的刻度范围设置为-6到10
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)	# 画点，对应low_dim_embs中每个词向量
        plt.annotate(label,	# 显示每个点对应哪个单词
                     xy=(x, y),
                     xytext=(5, 2),#文本显示的位置
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    x_major_locator = MultipleLocator(0.001)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.001)
    # 把y轴的刻度间隔设置为10，并存在变量里
    #ax = plt.gca()
    # ax为两条坐标轴的实例
    #ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    #ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数


    plt.savefig(filename)
    plt.show()

def load_txt_vec(file, threshold=0, dtype='float'):	# 读取txt文件格式的向量
    print('读取向量文件中......')
    words = []
    matrix = np.empty((200, 1024), dtype=dtype)
    i = 0
    for line in file.readlines():
        word = line.strip().split(" ")[0]
        vec = [float(i) for i in line.strip().split(" ")[1:]]
        words.append(word)
        #print(len(vec))
        matrix[i] = np.array(vec)
        i += 1
        if i==200:     #根据向量数目修改
            break
    return (words, matrix)



if __name__ == '__main__':
    try:
        # 若要载入txt文件格式的向量，则执行下面这两行
        file = open('imgtext.txt', 'r',encoding='utf-8', errors='surrogateescape')	# txt格式的向量文件
        words, vectors = load_txt_vec(file)	# 载入txt文件格式的向量
        tsne = TSNE(perplexity=10, n_components=2, init='pca', n_iter=2000, method='exact')#参数设置
        plot_only = 200	# 限定画点（词向量）的个数，只画词向量文件的前plot_only个词向量  #根据向量数目修改
        low_dim_embs = tsne.fit_transform(vectors[:plot_only])
        labels = [words[i] for i in range(plot_only)] # 要显示的点对应的标签列表
        plot_with_labels(low_dim_embs, labels, 'imgtext.png')

    except ImportError as ex:
        print('Please install gensim, sklearn, numpy, matplotlib, and scipy to show embeddings.')
        print(ex)























