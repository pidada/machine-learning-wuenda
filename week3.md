**吴恩达笔记-第三周**

主要讲解的内容包含：
- 逻辑回归
- 代价函数
- 线性回归和逻辑回归的比较
- 正则化问题

<!--MORE-->



### 一、逻辑回归

#### 分类问题

假设预测的变量y是离散的值，需要使用逻辑回归（Logistic Regression，LR）的算法，**实际上它是一种分类算法**

##### 二元分类问题

将因变量(**dependent variable**)可能属于的两个类分别称为负向类（**negative class**）和正向类（**positive class**），因变量y的取值只能在0和1之间，其中0表示负类，1表示正类

![MxqRAg.png](https://s2.ax1x.com/2019/11/26/MxqRAg.png)

#### 假说表示Hypothesis Representation

分类器的输出值在0和1之间，因此，希望找出一个满足某个性质的假设函数，这个性质是它的预测值要在0和1之间。



![MxjlVK.png](https://s2.ax1x.com/2019/11/26/MxjlVK.png)





![Mxj7RJ.png](https://s2.ax1x.com/2019/11/26/Mxj7RJ.png)

引入一个新的模型：逻辑回归。该模型的输出变量范围始终在0和1之间。 逻辑回归模型的假设是：
$$
h(\theta) = g(\theta^TX)
$$
其中X代表的是特征向量g的逻辑函数，常用的S型函数（上图的右边，sigmoid function）公式为
$$
g(z)= \frac{1}{1+e^{-z}}
$$
Python代码实现sigmod激活函数：

```python
import numpy as np

def sigmod(z):
  return 1 / (1 + np.exp(-z))
```

$$
h_\theta(x)=g(z)= \frac{1}{1+e^{-\theta^TX}}
$$

$h_{\theta}(x)$作用是对于给定的输入变量，根据选择的参数计算输出变量=1的可能性，即：$h_{\theta}(x)=P(y=1|x;\theta)$

例如：对于给定的x，通过已经确定的参数计算得出$h_{\theta}(x)=0.7$，则表示有70%的几率y属于正类



#### 决策边界decision boundary

##### 解释逻辑回归

1. 在逻辑回归中$h \geq 0.5$预测$y=1$；反之y=0

2. 在激活函数$g(z)$中：

当$z \geq 0$则$g(z) \geq 0.5$

当$z < 0$则$g(z) < 0.5$

3. 又因为$z=\theta^Tx$，则$\theta^Tx \geq 0$则y=1；反之y=0

##### 实例demo

在下图的中实例中，参数$\theta$满足[-3,1,1]，当$-3+x_1+x_2 \geq0$，即$x_1+x_2\geq3$时，模型预测y=1；说明此时：直线$x_1+x_2=3$就是决策边界

![MzSvXd.png](https://s2.ax1x.com/2019/11/26/MzSvXd.png)

**复杂的模型边界问题**

![MzpIgg.png](https://s2.ax1x.com/2019/11/26/MzpIgg.png)



# 二、代价函数Cost Function

#### 如何拟合LR模型的参数$\theta$

![Mz9yGT.png](https://s2.ax1x.com/2019/11/26/Mz9yGT.png)

**1. 线性模型**中代价函数是模型误差的**平方和**
$$
J(\theta)=\frac{1}{2m}\sum^m_{i=1}(h_{\theta}(x^{(i)})-y^{(i)})^2
$$

​			

如果直接使用线性模型中的代价函数，即误差平方和，得到的代价函数是个**"非凸函数"**，但是实际上我们期望看的是凸函数（右边）

![MzkHrF.png](https://s2.ax1x.com/2019/11/26/MzkHrF.png)

2. **重新定义逻辑回归的代价函数**
   				
   $$
   h_\theta(x)=g(\theta^TX)= \frac{1}{1+e^{-\theta^TX}}
   $$

$$
J(\theta)=\frac{1}{m}\sum^m_{i=1}Cost(h_\theta(x^{(i)}),y^{(i)})
$$

$$
Cost(h_\theta(x), y) = 
\begin{cases}
-\log(h_\theta(x)), & \text{y=1} \\
-\log(1-h_\theta(x)), & \text{y=0} \\
\end{cases}
$$

将上面的两个式子进行合并：
$$
Cost(h_\theta(x), y)=-y\log(h_\theta(x))-(1-y)\log(1-h_\theta(x))
$$
![MzVxGn.png](https://s2.ax1x.com/2019/11/26/MzVxGn.png)

3. $h_\theta(x)$和$Cost(h_\theta(x),y)$之间的关系

根据y的不同取值来进行分别判断，同时需要注意的是：假设函数h的取值只在[0,1]之间

![MzZFZF.png](https://s2.ax1x.com/2019/11/26/MzZFZF.png)

**y=1的情形**

![MzPSB9.png](https://s2.ax1x.com/2019/11/26/MzPSB9.png)					
						

**y=0的情形**

![MzP8gS.png](https://s2.ax1x.com/2019/11/26/MzP8gS.png)



#### Python代码实现代价函数

利用`Python`实现下面的代价函数

- `first` 表示的是右边第一项
- `second` 表示的是右边第二项

$$
Cost(h_\theta(x), y)=-y\log(h_\theta(x))-(1-y)\log(1-h_\theta(x))
$$



```python
import numpy as np

def cost(theta, X, y):
  # 实现代价函数
  
  theta=np.matrix(theta)
  X = np.matrix(X)
  y = np.matrxi(y)
  
  first = np.multiply(-y, np.log(sigmod(X * theta.T)))
  second = np.multiply((1 - y), np.log(1-sigmod(X * theta.T)))
  
  return np.sum(first - second) / (len(X))
```

#### 利用梯度下降来求解LR最小参数

**LR中的代价函数是**
$$
J(\theta)=-\frac{1}{m}\sum^m_{i=1}[-y^{(i)}\log(h_\theta(x^{(i)}))-(1-y^{(i)})\log(1-h_\theta(x^{i}))]
$$
**最终结果**
$$
\frac{\partial J(\theta)}{\partial \theta_j}=\frac{1}{m}\sum^m_{i=1}[h_\theta(x^{(i)})-y^{(i)}]x_j^{(i)}
$$
**具体过程**

![MzyOw6.png](https://s2.ax1x.com/2019/11/26/MzyOw6.png)

不断地迭代更新$\theta_{j}$
$$
\theta_{j} := \theta_j-\alpha\frac{\partial J(\theta)}{\partial \theta_j}
$$

$$
\theta_{j} := \theta_j-\alpha\frac{1}{m}\sum^m_{i=1}[h_\theta(x^{(i)})-y^{(i)}]x_j^{(i)}
$$

如果存在n个特征，也就是$\theta=[\theta_0,\theta_1,…,\theta_n]^T$。那么就需要根据上面的式子从0-n来更新所有的$\theta$

### 三、线性回归 VS 逻辑回归

1. 假设的定义规则发生变化

线性回归：
$$
h_{\theta}{(x)}=\theta^TX=\theta_0x_0+...+\theta_nx_n
$$
逻辑回归：
$$
h_\theta{(x)}= \frac{1}{1+e^{-\theta^TX}}
$$

> 因此，即使更新参数的规则看起来基本相同，但由于假设的定义发生了变化，所以逻辑函数的梯度下降，跟线性回归的梯度下降实际上是两个完全不同的东西。

#### 其他求解代价函数最小的算法

- 共轭梯度conjugate gradient
- 局部优化法**Broyden fletcher goldfarb shann,BFGS**
- **有限内存局部优化法(LBFGS)**

### 四、多类别分类one-vs-all

实际中的例子

> 假如现在需要一个学习算法能自动地将邮件归类到不同的文件夹里，或者说可以自动地加上标签，那么需要一些不同的文件夹，或者不同的标签来完成这件事，来区分开来自工作、朋友、家人或者有关兴趣爱好的邮件，那么，就有了这样一个分类问题：其类别有4个，分别用$y=1,2,3,4$ 来代表。

![Mz55HH.png](https://s2.ax1x.com/2019/11/26/Mz55HH.png)

### 五、正则化问题Regularization

#### 正则化基础

> 正则化技术主要是为了解决过拟合的问题。**过拟合指的是**：对样本数据具有很好的判断能力，但是对新的数据预测能力很差。

![MzINad.png](https://s2.ax1x.com/2019/11/26/MzINad.png)

- 第一个模型是一个线性模型，欠拟合，不能很好地适应我们的训练集

- 第三个模型是一个四次方的模型，过于强调拟合原始数据，而丢失了算法的本质：预测新数据
- 中间的模型似乎最合适

如果是多项式拟合，x的次数越高，拟合的效果越好，但是相应的预测能力就可能变差。**对于过拟合的处理：**

1. 丢弃一些不能正确预测的特征。可以是手工选择保留哪些特征，或者使用一些模型选择的算法，例如**PCA**
2. 正则化。 保留所有的特征，但是减少参数的大小（**magnitude**）

![Mz78MV.png](https://s2.ax1x.com/2019/11/26/Mz78MV.png)



#### 加入正则化参数

在模型$h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2+\theta_3x_3+\theta_4x_4$中，主要是高次项产生的过拟合问题：

![MzL71s.png](https://s2.ax1x.com/2019/11/26/MzL71s.png)

加入正则化参数后能够防止过拟合问题，其中$\lambda$是正则化参数**Regularization Parameter**
$$
J(\theta)=\frac{1}{2m}\sum^m_{i=1}(h_{\theta}(x^{(i)})-y^{(i)})^2+\lambda \sum^n_{j=1}\theta^2_{j}
$$
**Attention**：一般地，不对$\theta_0$进行惩罚；加上正则化参数实际上是对参数$\theta$进行惩罚。经过正则化处理后的模型和原模型的对比：

![MzHJSI.png](https://s2.ax1x.com/2019/11/26/MzHJSI.png)

- 如果$\lambda$过大，所有的参数最小化，模型变成了$h_\theta(x)=\theta_0$，造成了过拟合

#### 正则化线性回归Regularized Linear Regression

正则化线性回归的代价函数
$$
J(\theta)=\frac{1}{2m}\sum^m_{i=1}[(h_{\theta}(x^{(i)})-y^{(i)})^2+\lambda \sum^n_{j=1}\theta^2_{j}]
$$


**Attention：在线性回归中，不对$\theta_0$进行正则化**
$$
{\theta_0}:={\theta_0}-a\frac{1}{m}\sum\limits_{i=1}^{m}{(({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{0}^{(i)}}
$$
当$j=1,2,…,n$时
$$
{\theta_j}:={\theta_j}-a[\frac{1}{m}\sum\limits_{i=1}^{m}{(({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{j}^{\left( i \right)}}+\frac{\lambda }{m}{\theta_j}]   ；j=1,2,...,n
$$
**调整下变成：**
$$
{\theta_j}:={\theta_j}(1-\frac{\lambda }{m})-a\frac{1}{m}\sum\limits_{i=1}^{m}{(({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{j}^{\left( i \right)}}
$$


#### 正则化逻辑回归Regularized Logistic Regression

LR问题两种优化方法：

- 梯度下降法
- 更高级优化算法

**加上正则惩罚项后的代价函数为：**
$$
J(\theta)=\frac{1}{m}\sum^m_{i=1}[-y^{(i)}\log(h_\theta(x^{(i)}))-(1-y^{(i)})\log(1-h_\theta(x^{i}))]+\frac{\lambda}{2m}\sum^n_{j=1}\theta^2_j
$$

#### python代码实现

```python
import numpy as np

# 实现代价函数
def costReg(theta, X, y, lr):
  theta= np.matrix(theta)
  X = np.matrix(X)
  y = np.matrix(y)
  
  first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
  second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
  reg = (lr / (2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))   # theta[:, 1:theta.shape[1]] 代表的是 \theta_j 
  return np.sum(first - second) / len((X)) + reg
```



通过求导，得到梯度下降算法，本质上就是对$\theta$的不断更新：
$$
{\theta_0}:={\theta_0}-a\frac{1}{m}\sum\limits_{i=1}^{m}{(({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{0}^{(i)}})
$$

$$
{\theta_j}:={\theta_j}-a[\frac{1}{m}\sum\limits_{i=1}^{m}{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{j}^{\left( i \right)}}+\frac{\lambda }{m}{\theta_j}]  ； j=1,2,...,n
$$

