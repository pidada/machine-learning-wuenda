---
title: 吴恩达笔记12
top: false
cover: false
toc: true
mathjax: true
copyright: true
date: 2019-12-09 16:11:28
password:
summary:
tags:
  - OCR
  - 滑动窗口
categories:
  - Machine learning
  - 吴恩达
---


本周主要是介绍了两个方面的内容，一个是如何进行大规模的机器学习，另一个是关于图片文字识别OCR 的案例

<!--MORE-->

### 大规模机器学习(Large Scale Machine Learning)

在低方差的模型中，增加数据集的规模可以帮助我们获取更好的结果。但是当数据集增加到100万条的大规模的时候，我们需要考虑：大规模的训练集是否真的有必要。获取1000个训练集也可以获得更好的效果，通过绘制学习曲线来进行判断。

![](https://tva1.sinaimg.cn/large/006tNbRwly1g9qgzhssuvj30y20bomze.jpg)



#### 随机梯度下降法Stochastic Gradient Descent

如果需要对大规模的数据集进行训练，可以尝试使用随机梯度下降法来代替批量梯度下降法。随机梯度下降法的代价函数是
$$
\operatorname{Cost}\left(\theta,\left(x^{(i)}, y^{(i)}\right)\right)=\frac{1}{2}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}
$$
具体算法的过程为

1. 先对训练集进行随机的洗牌操作，打乱数据的顺序
2. 重复如下过程：

![](https://tva1.sinaimg.cn/large/006tNbRwly1g9qhq5faazj30k0088760.jpg)

3. 随机梯度下降算法是每次计算之后更新参数$\theta$，不需要现将所有的训练集求和。

**算法可能存在的问题**

> 不是每一步都是朝着”正确”的方向迈出的。因此算法虽然会逐渐走向全 局最小值的位置，但是可能无法站到那个最小值的那一点，而是在最小值点附近徘徊。

![](https://tva1.sinaimg.cn/large/006tNbRwly1g9qhse97qij30ki0eyk2x.jpg)



#### 小批量梯度下降 Mini-Batch Gradient Descent

小批量梯度下降算法是介于批量梯度下降算法和梯度下降算法之间的算法。每计算常数b次训练实例，便更新一次参数$\theta$。参数b通常在2-100之间。

![](https://tva1.sinaimg.cn/large/006tNbRwly1g9qhwj2h0ij30re0de0wa.jpg)

#### 随机梯度下降收敛

随机梯度下降算法的调试和学习率$\alpha$的选取

- 在**批量梯度下降算法**中，可以令代价函数$J$为迭代次数的函数，绘制图表，根据图表来 判断梯度下降是否收敛；大规模的训练集情况下，此举不现实，计算代价太大
- 在随机梯度下降中，更新$\theta$之前都计算一次代价，然后迭代$X$后求出$X$对训练实例的计算代价的平均值，最后绘制次数$X$和代价平均值之间的图像

![](https://tva1.sinaimg.cn/large/006tNbRwly1g9qi6lrehcj30so0fatbq.jpg)



随着不断地靠近全局最小值，通过减小学习率，迫使算法收敛而非在最小值最近徘徊。

![](https://tva1.sinaimg.cn/large/006tNbRwly1g9qi9sj4kuj30kc0bc0z2.jpg)

#### 映射化简和数据并行Map Reduce and Data Parallelism

映射化简和数据并行对于大规模机器学习问题而言是非常重要的概念。如果我们能够将我们的数据集分配给不多台 计算机，让每一台计算机处理数据集的一个子集，然后我们将计所的结果汇总在求和。这样 的方法叫做**映射简化**。

如果任何学习算法能够表达为对训练集的函数求和，那么便能将这个任务分配给多台计算机（或者同台计算机的不同CPU核心），达到加速处理的目的。比如400个训练实例，分配给4台计算机进行处理：

![](https://tva1.sinaimg.cn/large/006tNbRwly1g9qihqa135j315a074tgk.jpg)

### 图片文字识别(Application Example: Photo OCR)

#### 问题描述和流程图

图像文字识别应用所作的事是从一张给定的图片中识别文字。

![](https://tva1.sinaimg.cn/large/006tNbRwly1g9qij5kqzej30jm0bstl5.jpg)

基本步骤包含：

1. **文字侦测(Text detection)**——将图片上的文字与其他环境对象分离开来
2. **字符切分(Character segmentation)**——将文字分割成一个个单一的字符
3. **字符分类(Characterclassification)**——确定每一个字符是什么 可以用任务流程图来表

每项任务可以有不同的团队来负责处理。

![](https://tva1.sinaimg.cn/large/006tNbRwly1g9qil9py53j30tk02sab3.jpg)

#### 滑动窗口Sliding windows

##### 图片识别

滑动窗口是一项用来从图像中抽取对象的技术。看一个栗子：

![](https://tva1.sinaimg.cn/large/006tNbRwly1g9qiqtaps3j30gq0betj2.jpg)

如果我们需要从上面的图形中提取出来行人：

- 用许多固定尺寸的图片来训练一个能够准确识别行人的**模型**
- 用上面训练识别行人的模型时所采用的**图片尺寸**在我们要进行行人识别的图片上进行**剪裁**
- 剪裁得到的切片交给模型，让**模型判断是否为行人**
- 重复循环上述的操作步骤，直至将图片全部检测完。

##### 文字识别

滑动窗口技术也被用于文字识别。

- 首先训练模型能够区分字符与非字符
- 然后运用滑动窗口技术识别字符
- 完成字符的识别，将识别得出的区域进行扩展
- 将重叠的区域进行合并，以宽高比作为过滤条件，过滤掉高度比宽度更大的区域

![](https://tva1.sinaimg.cn/large/006tNbRwly1g9qix7iehfj30g009uac5.jpg)

上述步骤是**文字侦察阶段**，接下来通过训练出一个模型来讲文字分割成一个个字符，需要的训练集由单个字符的图片和两个相连字符之间的图片来训练模型。

![](https://tva1.sinaimg.cn/large/006tNbRwly1g9qj0f93zgj30m207kmzt.jpg)

训练完成之后，可以通过滑动窗口技术来进行字符识别。该阶段属于**字符切分阶段**。

![](https://tva1.sinaimg.cn/large/006tNbRwly1g9qj1o802yj30ia02ydhz.jpg)

最后通过利用神经网络、支持向量机、或者逻辑回归等算法训练出一个分类器，属于是**字符分类阶段**。

#### 获取大量数据和人工数据

> 如果我们的模型是低方差的，那么获得更多的数据用于训练模型，是能够有更好的效果。

获取大量数据的方法有

- 人工数据合成
- 手动收集、标记数据
- 众包

#### 上限分析Ceiling Analysis

在机器学习的应用中，我们通常需要通过几个步骤才能进行最终的预测，我们如何能够 知道哪一部分最值得我们花时间和精力去改善呢?这个问题可以通过上限分析来回答。

![](https://tva1.sinaimg.cn/large/006tNbRwly1g9qil9py53j30tk02sab3.jpg)

回到文字识别的应用中，流程图如下：

我们发现每个部分的输出都是下个部分的输入。在上限分析中，我们选取其中的某个部分，手工提供100%争取的输出结果，然后看整体的效果提升了多少。

- 如果提升的比例比较明显，可以考虑在这个方向投入更过的时间和经历
- 如果提升的效果微乎其微，意味着某个部分已经做的足够好了

![](https://tva1.sinaimg.cn/large/006tNbRwly1g9qjd3icglj30n6076ad8.jpg)
