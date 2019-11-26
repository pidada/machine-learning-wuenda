吴恩达笔记-第一周

在第一周中讲解的内容包含：

- 监督学习和无监督学习
- 单变量线性回归问题
- 代价函数
- 梯度下降算法

<!--MORE-->

### 监督学习Supervised Learning

#### 利用监督学习预测波士顿房价（回归问题）

- 大多数情况下，可能会拟合直线
- 有时候用二次曲线去拟合效果可能会更好的

![MX0eoR.png](https://s2.ax1x.com/2019/11/25/MX0eoR.png)

在监督学习中，我们给学习算法一个数据集，比如一系列房子的数据，给定数据集中每个样本的正确价格，即它们实际的售价然后运用学习算法，算出更多的答案，我们需要估算一个连续值的结果，这属于**回归问题**



#### 利用监督学习来推测乳腺癌良性与否（分类问题）

![MX03Oe.png](https://s2.ax1x.com/2019/11/25/MX03Oe.png)

- 横轴表示肿瘤的大小
- 纵轴表示1表示恶性，0表示良性

机器学习的问题就在于，估算出肿瘤是恶性的或是良性的概率，属于**分类问题**。

分类指的是，我们试着推测出离散的输出值：0或1良性或恶性，而事实上在分类问题中，输出可能不止两个值。

比如说可能有三种乳腺癌，所以希望预测离散输出0、1、2、3。0 代表良性，1 表示第1类乳腺癌，2表示第2类癌症，3表示第3类，也是分类问题。

#### 应用

- 垃圾邮件问题
- 疾病分类问题

-----

### 无监督学习Unsupervised Learning

- 监督学习中，数据是有标签的
- 无监督学习中，数据是没有标签，主要提到了聚类算法

![MX0UYt.png](https://s2.ax1x.com/2019/11/25/MX0UYt.png)

#### 应用

- 基因学的理解应用
- 社交网络分析
- 组织大型计算机集群
- 细分市场
- 新闻事件分类

-----

### 单变量线性回归Linear Regression with One Variable

#### 房价问题

横轴是不同的房屋面积，纵轴是房屋的出售价格。

监督学习：对于每个数据来说，给出了正确的答案。在监督学习中，我们有一个给定的数据，叫做**训练集training set**

回归问题：根据之前的数据，预测出一个准确的输出值。

分类问题：预测离散的输出值，例如寻找癌症肿瘤，并想要确定肿瘤是良性的还是恶性的，属于0/1离散输出的问题

![MXgYGj.png](https://s2.ax1x.com/2019/11/25/MXgYGj.png)

#### 监督学习工作模式

![MXgRQ1.png](https://s2.ax1x.com/2019/11/25/MXgRQ1.png)

学习过程解释：

- 将训练集中的房屋价格喂给学习算法
- 学习算法工作，输出一个函数，用h表示
- h表示hypothesis，代表的是学习算法的解决方案或者函数。
- h根据输入的x值得到y值，因此h是x到的y的一个函数映射
- 可能的表达式：$h_{\theta}(x)=\theta_0+\theta_1x$，只有一个特征或者出入变量，称为**单变量线性回归问题**

### 代价函数cost function

代价函数也称之为**平方误差函数，平方误差代价函数**

#### 函数解释

- m：训练样本的个数
- $h_{\theta}(x)=\theta_0+\theta_1x$：假设函数
- $\theta_0$ 和$\theta_1$：表示两个模型参数，即直线的斜率和y轴上的截距

![MXgvef.png](https://s2.ax1x.com/2019/11/25/MXgvef.png)

#### 建模误差

##### 建模目标

1. 图中红色的点表示真实值$y_i$，真实的数据集
2. $h(x)$表示的是通过模型得到的预测值
3. 目标：选择出可以使得建模误差的平方和能够最小的模型参数

$$
J(\theta_0,\theta_1)=\frac{1}{2m}\sum^m_{i=1}(h_{\theta}(x^{(i)})-y^{(i)})^2
$$

![MX25Xq.png](https://s2.ax1x.com/2019/11/25/MX25Xq.png)

![MXRtuq.png](https://s2.ax1x.com/2019/11/25/MXRtuq.png)



#### 代价函数直观解释1

本例中是通过假设$\theta_0=0$来进行，假设函数$h(x)$是关于x的函数，代价函数$J(\theta_0,\theta_1)$是关于$\theta$的函数，使得代价函数最小化

![MXWzTK.png](https://s2.ax1x.com/2019/11/25/MXWzTK.png)

#### 代价函数直观解释2

通过等高线图来进行解释。通过绘制出等高线图可以看出来，必定存在某个点，使得代价函数最小

![MXWCMn.png](https://s2.ax1x.com/2019/11/25/MXWCMn.png)

![MXIzGT.png](https://s2.ax1x.com/2019/11/25/MXIzGT.png)

### 梯度下降Gradient Descent

#### 思想

> 梯度下降是一个用来求函数最小值的算法。

- 背后的思想：开始随机选取一个参数的组合$(\theta_0,\theta_1,…,\theta_n)$计算代价函数，然后我们寻找下一个能让代价函数值下降最多的参数组合。

- 持续这么做，直到一个局部最小值（**local minimum**），因为并没有尝试完所有的参数组合，所以不能确定得到的局部最小值是否是全局最小值（**global minimum**）
- ![MX7l5t.png](https://s2.ax1x.com/2019/11/25/MX7l5t.png)



![MXbBcT.png](https://s2.ax1x.com/2019/11/25/MXbBcT.png)



#### 批量梯度下降**batch gradient descent**

算法公式为

![MXH5wj.png](https://s2.ax1x.com/2019/11/25/MXH5wj.png)

**特点：需要同步更新两个参数**

#### 梯度下降直观解释

算法公式：$$\theta_j:=\theta_j-\alpha \frac {\partial J(\theta)}{\partial \theta_j}$$

具体描述：对$\theta$赋值，使得$J(\theta)$按照梯度下降最快的方向进行，一直迭代下去，最终得到局部最小值。

学习率：$\alpha$是学习率它决定了我们沿着能让代价函数下降程度最大的方向向下迈出的步子有多大。

- 学习率太小：收敛速度慢需要很长的时间才会到达全局最低点
- 学习率太大：可能越过最低点，甚至可能无法收敛

![MXLMy8.png](https://s2.ax1x.com/2019/11/25/MXLMy8.png)

![MXLL1P.png](https://s2.ax1x.com/2019/11/25/MXLL1P.png)

### 梯度下降的线性回归GradientDescent-For-LinearRegression

梯度下降是很常用的算法，它不仅被用在线性回归上和线性回归模型、平方误差代价函数。将梯度下降和代价函数相结合。

#### 梯度下降VS线性回归算法

![MXXAPA.png](https://s2.ax1x.com/2019/11/25/MXXAPA.png)

对之前的线性回归问题运用梯度下降法，关键在于求出代价函数的导数，即：
$$
\frac {\partial J(\theta_0,\theta_1)}{\partial \theta_j}=\frac{1}{2m}\frac {\partial \sum^m_{i=1}(h_{\theta}(x^{(i)})-y^{(i)})^2}{\partial \theta_j}
$$
$j=0$：
$$
\frac {\partial J(\theta_0,\theta_1)}{\partial \theta_0}=\frac{1}{m}\sum^m_{i=1}(h_\theta{(x^{(i)})-y^{(i)}})
$$
$j=1$:
$$
\frac {\partial J(\theta_0,\theta_1)}{\partial \theta_1}=\sum^{m}_{i=1}((h_\theta(x^{(i)})-y^{(i)})\cdot x^{(i)})
$$
因此将算法改成：
$$
\theta_0 := \theta_0- \alpha \frac{1}{m}\sum^m_{i=1}(h_\theta{(x^{(i)})-y^{(i)}})
$$

$$
\theta_1 := \theta_1- \alpha  \frac {1}{m}\sum^{m}_{i=1}((h_\theta(x^{(i)})-y^{(i)})\cdot x^{(i)})
$$

这种梯度下降的算法称之为“批量梯度下降算法”，主要特点：

- 在梯度下降的每一步中，我们都用到了所有的训练样本
- 在梯度下降中，在计算微分求导项时，我们需要进行求和运算,需要对所有m个训练样本求和