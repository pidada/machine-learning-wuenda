---
title: 吴恩达笔记7
top: false
cover: false
toc: true
mathjax: true
copyright: true
date: 2019-12-03 23:24:39
password:
summary:
tags:
  - SVM
categories:
  - Machine learning
  - 吴恩达
---



Week7-SVM

本周主要是讲解了支持向量机SVM的相关知识点

- 硬间隔
- 支持向量
- 软间隔
- 对偶问题

<!--MORE-->

### 优化目标Optimization Objectives

主要是讲解如何从逻辑回归慢慢的推导出本质上的支持向量机。逻辑回归的假设形式：

![](https://tva1.sinaimg.cn/large/006tNbRwgy1g9jb3l5iqsj30ws0h0jwr.jpg)

- 左边是假设函数
- 右边是`Sigmoid`激活函数

令$z=\theta^Tx$，如果满足：

1. 若$y=1$，希望$h(\theta)$约为1，将样本正确分类，那么z必须满足$z>>0$
2. 若$y=0$，希望$h(\theta)$约为0，将样本正确分类，那么z必须满足$z<<0$

> 样本正确分类指的是：假设函数h(x)得到的结果和真实值y是一致的

![](https://tva1.sinaimg.cn/large/006tNbRwgy1g9jbtyk7aej31440metmd.jpg)

总代价函数通常是对所有的训练样本进行求和，并且每个样本都会为总代价函数增加上式的最后一项（还有个系数$\frac{1}{m}$，系数忽略掉）

1. 如果$y=1$，目标函数中只有第一项起作用，得到了表达式
   $$
   y=1-log(1-\frac{1}{1+e^{-z}})
   $$

### 支持向量机

![](https://tva1.sinaimg.cn/large/006tNbRwly1g9jc3uucfbj313u0lmne1.jpg)

根据逻辑回归推导得到的支持向量机的公式
$$
min C\sum^m_{i=1}[y^{(i)}cost_1(\theta^Tx^{(i)})+(1-y^{(i)})cost_0(\theta^Tx^{(i)}]+\frac{1}{2}\sum^n_{i=1}\theta_{j}^2
$$
两个$cost$函数是上面提到的两条直线。

对于逻辑回归，在目标函数中有两项：第一个是训练样本的代价，第二个是正则化项

### 大边界的直观解释

下面是支持向量机的代价函数模型。

![](https://tva1.sinaimg.cn/large/006tNbRwly1g9jce0d8luj31500l8akx.jpg)

#### SVM决策边界

![](https://tva1.sinaimg.cn/large/006tNbRwgy1g9jch4a4r2j314e0ie493.jpg)

**SVM鲁棒性**：间隔最大化，是一种大间距分类器。

![](https://tva1.sinaimg.cn/large/006tNbRwgy1g9jck7m31aj31360iqdoz.jpg)



![](https://tva1.sinaimg.cn/large/006tNbRwly1g9jcntequoj30uo0f8q98.jpg)

关于上图的解释：

1. C太大的话，将是粉色的线
2. C不是过大的话，将是黑色的线

> 大间距分类器的描述，仅仅是从直观上给出了正则化参数C非常大的情形，C的作用类似于之前使用过的正则化参数$\frac{1}{\lambda}$

- C较大，可能导致过拟合，高方差
- C较小，可能导致低拟合，高偏差



#### 硬间隔模型

![](https://tva1.sinaimg.cn/large/006tNbRwly1g9jut9jb7tj30pg0oodpu.jpg)



### 间隔和支持向量

注释：本文中全部采用列向量$w=(w_1,w_2,…,w_n)$

给定一个样本训练集$D=\{(x_1,y_1),(x_2,y_2),…,(x_m,y_m)\}$，其中$y_i \in \{-1,+1\}$分类学习的基本思想就是：基于训练集D在样本空间上找到一个划分的超平面

![](https://tva1.sinaimg.cn/large/006tNbRwly1g9jdjknd3gj30zv0u0u0x.jpg)

上面红色的线是最好的。所产生的分类结果是最鲁棒的，最稳定的，泛化能力是最好的。

划分超平面的的线性描述：
$$
w \cdot x+b=0
$$
$w$称之为法向量（看做是列向量），决定平面的方向；$b$是位移项，决定了超平面和原点之间的距离。

空间中任意一点$x$到超平面$(w,b)$的距离是
$$
r=\frac{|w \cdot x + b|}{||w||}
$$
在$+$区域的点满足$y=+1$：
$$
w \cdot x_+ + b \geq1
$$
在$-$区域的点满足$y=-1$：
$$
w \cdot x_- + b \leq-1
$$
综合上面的两个式子有：
$$
y(w \cdot x_+ + b )-1 \geq0
$$

#### 支持向量

距离超平面最近的几个点（带上圆圈的几个点）称之为“支持向量support vector”，这个点到超平面到距离称之为“间隔margin”

刚好在决策边界上的点（下图中带上圆圈的点）满足上式中的等号成立：
$$
y_i(w \cdot x + b) -1=0
$$


![](https://tva1.sinaimg.cn/large/006tNbRwgy1g9jkhlqhotj30n80i23zp.jpg)



#### 间距margin

求解间距`margin`就是求解向量$(x_+-x_-)$在法向量上的投影
$$
\begin{align}margin& = (x_+-x_-)\cdot \frac{w}{||w||} \\& = \frac {(x_+ \cdot w-x_-\cdot w}{||w||}\end{align}
$$
决策边界上的正例表示为：
$$
y_i=+1 \rightarrow 1*(w\cdot x_+ +b) - 1 =0 \rightarrow w\cdot x_+ =1-b
$$
决策边界行的负例表示为：
$$
y_i=-1 \rightarrow -1*(w\cdot x_- +b) - 1 =0 \rightarrow w\cdot x_- =-1-b
$$
将两个结果带入`margin` 的表达式中
$$
margin=\frac {1-b-(-1-b)}{||w||}=\frac{2}{||w||}
$$


![](https://tva1.sinaimg.cn/large/006tNbRwly1g9jee5offyj313c0u0e0c.jpg)



#### SVM的基本模型

最大间隔化只需要将$||w||$最小化即可
$$
\min_{w,b}\frac{1}{2}||w||^2 \\s.t.   y_i(w\cdot x_i+b) \geq 1 ;i=1,2,...,m
$$

### SVM-对偶模型

#### 模型参数推导

希望求解上面基本模型对应超平面的模型
$$
f(x)= w\cdot x+b
$$
利用拉格朗日乘子$\alpha_i$，改成拉格朗日函数
$$
L(w,b,\alpha)=\frac{1}{2}||w||^2+\sum^m_{i=1}\alpha_i(1-y_i(w^Tx_i+b))
$$
分别对$w,b$求导，可以得到
$$
w = \sum^m_{i=1}\alpha_iy_ix_i  \\ 0 = \sum^m_{i=1}\alpha_iy_i
$$


#### 对偶模型

原始问题是极大
$$
\min_{w,b}\max_{\alpha}L(w,b,\alpha)\rightarrow\max_{\alpha}\min_{w,b}L(w,b,\alpha)
$$
最大值问题：带入拉格朗日函数中，得到对偶问题（全部是关于$\alpha$系数）
$$
\max _{\alpha}\sum^m_{i=1}\alpha_i-\frac{1}{2}\sum^m_{i=1}\sum^m_{j=1}\alpha_i\alpha_jy_i y_jx_i^Tx_j \\s.t. \sum^m_{i=1}\alpha_iy_i=0 \\\alpha_i \geq0;i=1,2,...,m
$$
转换一下，变成最小值问题（上面的式子加上负号）：
$$
\min _{\alpha}\frac{1}{2}\sum^m_{i=1}\sum^m_{j=1}\alpha_i\alpha_jy_i y_jx_i^Tx_j - \sum^m_{i=1}\alpha_i\\s.t. \sum^m_{i=1}\alpha_iy_i=0 \\\alpha_i \geq0;i=1,2,...,m
$$


那么超平面的模型
$$
\begin{align}f(x)& = w\cdot x+b \\& = \sum^m_{i=1}\alpha_iy_ix_i^Tx+b \end{align}
$$


### SMO算法

#### 思想

SMO算法指的是`Sequential Minimal Optimization`，序列最小优化算法。算法的根本思路是：

所有的$\alpha$满足$\sum^m_{i=1}\alpha_iy_i=0 $：

1. 先选取需要更新的变量$\alpha_i$和$\alpha_j$
2. 固定变量$\alpha_i$和$\alpha_j$以外的参数，求解更新后的变量$\alpha_i$和$\alpha_j$

$$
\alpha_iy_i+\alpha_jy_j=c
$$

其中$c$使得上式成立：
$$
c= \sum_{k \neq i,j}\alpha_ky_k
$$

3. 将变量$\alpha_i$和$\alpha_j$的其中一个用另一个来表示，得到关于$\alpha_i$的单变量二次规划问题，就可以求出来变量$\alpha_i$

### 软间隔最大化

上面的结论和推导都是针对的线性可分的数据。线性不可分数据意味着某些样本点$(x_i,y_i)$不再满足函数间隔大于等于1的约束条件，比如下图中的红圈中的点，故引入了松弛变量$\xi_i \geq0$，满足：
$$
y_i(w \cdot x_i +b) +\xi_i \geq 1 \\ y_i(w \cdot x_i +b) \geq 1-\xi_i
$$
![](https://tva1.sinaimg.cn/large/006tNbRwly1g9js3ob78mj316a0u0npd.jpg)

因此，**目标函数**由原来的$\frac{1}{2}||w||^2$变成了
$$
\frac{1}{2}||w||^2+C\sum^N_{i=1}\xi _i
$$
其中$C\geq0$是惩罚项参数，C值越大对误分类的越大，C越小对误分类的惩罚越小。
