---
title: 社区发现-Fast Unfolding算法
tags:
  - community detection
date: 2018-01-11 20:05:43
categories: ml
---

在社交网络中，有些用户之间联系较为紧密，而另外一些用户之间的关系则较为稀疏。在网络中，我们可以将联系较为紧密的部分用户看成一个社区，在这个社区内部，用户之间联系紧密，而在两个社区之间，联系较为稀疏。

<!-- more -->

#### 社区划分的评价标准

利用算法将整个网络划分成多个社区之后，需要一个评价指标来衡量这个划分结果的好坏。fast unfolding算法采用的是模块度（Modularity）Q值来衡量。

模块度最早是由Newman在文章[Finding community structure in very large networks](https://arxiv.org/pdf/cond-mat/0408187.pdf) 中提出。

首先，定义$A_{vw}$为网络中邻接矩阵中的一个元素：

$$
A_{vw}=
\begin{cases}
1& 点v和w是相连的\\\\
0& 其它
\end{cases}
$$

则整个网络中的边数为

$$
m=\frac{1}{2}\sum\_{vw}A\_{vw}
$$

社区内部的边数和网络的总边数的比例为：

$$
\frac{\sum\_{vw}A\_{vw}\delta(c\_v,c\_w)}{\sum\_{vw}A\_{vw}} = \frac{1}{2m}\sum\_{vw}A\_{vw}\delta(c\_v,c\_w)
$$

其中，$c_v$表示点$v$所属的社区。当$i$,$j$存在于同一个社区中时，$\delta(i,j) = 1$，否则为0。

定义$k_v$表示点$v$的度，即

$$
k\_v = \sum\_w A\_{vw}
$$

则将网络设定成随机网络，并进行相同的社区分配操作形成的社区内部的总边数和网络中总边数的比例的大小为$k_vk_w/2m$。

于是，社区的**模块度**可以定义为：社区内部的总边数和网络中总边数的比例减去一个期望值，该期望值是将网络设定为随机网络时同样的社区分配所形成的社区内部的总边数和网络中总边数的比例的大小。

$$
Q = \frac{1}{2m}\sum\_{vw}\left[A\_{vw}-\frac{k_v k_w}{2m}\right]\delta(c\_v,c\_w)
$$

定义$e_{ij}$为社区$i$与社区$j$之间的边数占网络中所有边数的占比，即

$$
e\_{ij} = \frac{1}{2m} \sum\_{vw}A\_{vw}\delta(c\_v,i)\delta(c\_w,j)
$$

定义$a_i$为连接到社区$i$的边数占网络中所有边数的占比，即

$$
a\_i = \frac{1}{2m} \sum\_{v}k\_{v}\delta(c\_v,i)
$$

同时，由于$\delta(c\_v,c\_w)=\sum\_i\delta(c\_v,i)\delta(c\_w,i)$. 则模块度的计算可以简化为：

$$
\begin{eqnarray\*}
Q & = & \frac{1}{2m}\sum\_{vw}\left[A\_{vw}-\frac{k_v k_w}{2m}\right]\sum\_i\delta(c\_v,i)\delta(c\_w,i) \\\\
& = & \sum\_i \left[\frac{1}{2m}\sum\_{vw}A\_{vw}\delta(c\_v,i)\delta(c\_w,i)-\frac{1}{2m}\sum\_v k\_v\delta(c\_v,i)\frac{1}{2m}\sum\_w k\_w\delta(c\_w,i)\right]\\\\
& = & \sum\_i (e\_{ii}-a\_i^2)
\end{eqnarray\*}
$$

#### Fast Unfolding算法

在社区发现问题中，以前的研究人员提出了许多的方法，例如标签传播算法（Label Propagation Algorithm）、Fast Unfolding等。考虑到现有数据的规模和算法的复杂度等因素，本文选用的是fast unfolding。

Fast Unfolding算法的主要目标是不断划分社区使得划分后的整个网络的模块度不断增大。算法主要包括两个过程，过程示例如下。

![fast unfolding示意图](/imgs/ML/CommunityDetection/fast_unfolding.png)

1. **Modularity Optimization**，这一过程主要讲节点与邻近的社区进行合并，使得网络的模块度不断变大。

	定义$\sum\_{in}$为社区$C$内所有边的权重和，$\sum\_{tot}$为与社区$C$内的点连接的边的权重和，$k\_i$为所有连接到节点$i$上的边的权重和，$k\_{i,in}$为节点$i$与社区$C$内的节点连接的边的权重和，$m$是网络中所有边的权重和。

	则将节点$i$划分到社区$C$中产生的模块度的变化$\Delta Q$可用下式计算
	$$
	\begin{eqnarray\*}
	\Delta Q & = & \left[\frac{\sum\_{in} + k\_{i,in}}{2m} - \left(\frac{\sum\_{tot}+k\_i}{2m}\right)^2\right]-\left[\frac{\sum\_{in}}{2m} - \left(\frac{\sum\_{tot}}{2m}\right)^2 - \left(\frac{k\_i}{2m}\right)^2\right]\\\\
	& = & \frac{k\_{i,in}}{2m} - \frac{k\_i\sum\_{tot}}{2m^2}
	\end{eqnarray\*}
	$$

	根据上式，我们只需要知道社区中与该节点连接的边的权重之和，以及社区中的点连接的边的权重和就可以计算模块度的变化量。

2. **Commnunity Aggregation**这一过程将第一步中的社区汇聚成一个点，重构网络结果。

	这一步中，将原来的两个社区之间的边的权重和作为新的节点之间的权重，将社区内的权重和作为新节点上的环向边的权重。

fast unfolding算法将重复迭代以上过程，直至网络的结构不变。

# 参考文献
>[模块度Q——复杂网络社区划分评价标准](http://blog.csdn.net/wangyibo0201/article/details/52048248)
>[Finding community structure in very large networks](http://ece-research.unm.edu/ifis/papers/community-moore.pdf)
>[Fast unfolding of communities in large networks](https://arxiv.org/pdf/0803.0476.pdf)