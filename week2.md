吴恩达笔记-第二周

本周主要讲解的内容是：

- 多维特征
- 多变量梯度下降
- 梯度下降法实践
- 正规方程

<!--MORE-->

----

### 多维特征Multiple Features

还是利用房价模型的例子，增加了更多的特征，比如：房间楼层、房间数量、地理位置等，构成了一个含有多个变量的模型

![MjiSfS.png](https://s2.ax1x.com/2019/11/25/MjiSfS.png)

n：代表的是特征的数量

$x^{(i)}$：代表第$i$个训练实例，是特征矩阵中的第$i$行，是一个**向量vector**

$x^{(i)}_{j}$：表示的是第$i$个训练实例的第$j$个特征；i表示行，j表示列

**支持多变量的假设$h$表示为：**
$$
h_{\theta}(x)=\theta_0+\theta_1x_1+…+\theta_nx_n
$$
为了简化公式，引入$x_0=1$，公式转化为：
$$
h_{\theta}(x)=\theta_0x_0+\theta_1x_1+…+\theta_nx_n
$$
特征矩阵X 的维度是$m*(n+1)$，公式简化为：
$$
h_{\theta}{(x)}=\theta^{T}X
$$

----

### 多变量梯度下降

#### 算法目标

与单变量线性回归类似，在多变量线性回归中，构建一个代价函数，则这个代价函数是所有**建模误差的平方和**，即：
$$
J(\theta_0,\theta_1,...,\theta_n)=\frac{1}{2m}\sum^m_{i=1}(h_{\theta}(x^{(i)})-y^{(i)})^2
$$
其中
$$
h_\theta(x)=\theta^T{X}=\theta_0+\theta_1x_1+…+\theta_nx_n
$$

#### 算法过程

**原始形式：**
$$
\theta_j:=\theta_j-\alpha \frac {\partial J(\theta_0,\theta_1,...,\theta_n)}{\partial \theta_j}
$$
**将代价函数$J$带进去：**
$$
\theta_j:=\theta_j-\frac{1}{2m} \alpha \frac {\partial \sum^m_{i=1}(h_{\theta}(x^{(i)})-y^{(i)})^2}{\partial \theta_j}
$$
**求导数之后：**
$$
\theta_j:=\theta_j-\frac{1}{m} \alpha \sum^m_{i=1}((h_{\theta}(x^{(i)})-y^{(i)})\cdot x_j^{(i)}
$$
其中$j\in{(0,1,2,…,n)}$

当$n \geq 1$时：
$$
\theta_0:=\theta_0-\frac{1}{m} \alpha \sum^m_{i=1}((h_{\theta}(x^{(i)})-y^{(i)})\cdot x_0^{(i)}
$$

$$
\theta_1:=\theta_1-\frac{1}{m} \alpha \sum^m_{i=1}((h_{\theta}(x^{(i)})-y^{(i)})\cdot x_1^{(i)}
$$

$$
\theta_2:=\theta_2-\frac{1}{m} \alpha \sum^m_{i=1}((h_{\theta}(x^{(i)})-y^{(i)})\cdot x_2^{(i)}
$$

![MjKQwF.png](https://s2.ax1x.com/2019/11/25/MjKQwF.png)



**Python代码**

给定特征矩阵X，输出y，学习率$\theta$，求代价函数$J$

```python
import numpy as np

def computeCost(X,y,theta):
  inner = np.power(((X * theta.T) - y), 2)  # 求解每个平方项
  return np.sum(inner) / (2 / lne(X))   # 求和再除以2*len(X)
```

### 梯度下降法实践

#### 特征缩放

面对多维度特征问题，我们需要保证这些特征具有相近的尺度，帮助梯度下降算法更快地收敛。

以房价问题为例，假设仅用两个特征，房屋的尺寸和数量，以两个参数分别为横纵坐标，假设尺寸在0-2000平方英尺，数量在0-5之间。

绘制代价函数的等高线图能，看出图像会显得很扁，梯度下降算法需要非常多次的迭代才能收敛。

![MjKpsf.png](https://s2.ax1x.com/2019/11/25/MjKpsf.png)

**解决办法**：将所有的特征的尺度尽量缩放到-1到1之间，令：
$$
x_n=\frac{x_n-u_n}{s_n}
$$
其中$u_n$为平均值，$s_n$为标准差

![MvFKYV.png](https://s2.ax1x.com/2019/11/25/MvFKYV.png)

#### 均值归一化

![MvFOhV.png](https://s2.ax1x.com/2019/11/25/MvFOhV.png)

#### 学习率问题

梯度下降算法的每次迭代受到学习率的影响

- 如果学习率过小，则达到收敛所需的迭代次数会非常高，收敛速度非常慢
- 如果学习率过大，每次迭代可能不会减小代价函数，可能会越过局部最小值导致无法收敛

常用学习率包含：$\alpha=0.01, 0.03, 0.1, 0.31, 3,10$

### 特征和多项式回归

如房价预测问题，
$$
h_{\theta}{(x)} = \theta_0+\theta_1 \cdot宽度 + \theta_2 \cdot 深度
$$
同时房屋面积=宽度 * 深度

![MvArdA.png](https://s2.ax1x.com/2019/11/25/MvArdA.png)



在实际拟合数据的时候，可能会选择二次或者三次方模型；如果采用多项式回归模型，在运行梯度下降法之前，特征缩放很有必要。

![MvETne.png](https://s2.ax1x.com/2019/11/25/MvETne.png)

### 正规方程 Normal Equation

#### 梯度下降缺点

需要多次迭代才能达到局部最优解

![MvZFqe.png](https://s2.ax1x.com/2019/11/25/MvZFqe.png)

#### 正规方程demo

正规方程具有不可逆性

正规方程就是通过求解下面例子中的方程找出使得代价函数最小参数$\theta$
$$
\theta=(X^TX)^{-1}X^Ty
$$
![Mveew4.png](https://s2.ax1x.com/2019/11/25/Mveew4.png)

**不可逆矩阵不能使用正规方程求解**

#### Normal Equation VS Gradient Descent

| 梯度下降                    | 正规方程                                                     |
| :-------------------------- | :----------------------------------------------------------- |
| 需要选择学习率$\theta$      | 不需要                                                       |
| 需要多次迭代                | 一次运算得出                                                 |
| 当特征数量n大时也能较好适用 | 需要计算 $(X^TX)^{-1}$如果特征数量n较大则运算代价大<br />矩阵逆的计算时间复杂度为$O(n^3)$，通常小于10000建议用正规方程 |
| 适用于各种类型的模型        | 只适用于线性模型，不适合逻辑回归模型等其他模型               |

![MvGlLR.png](https://s2.ax1x.com/2019/11/25/MvGlLR.png)

#### 参数$\theta$求解过程

**目标任务**
$$
\theta=(X^TX)^{-1}X^Ty
$$
其中：
$$
J(\theta)=\frac{1}{2m}\sum^m_{i=1}(h_{\theta}(x^{(i)})-y^{(i)})^2
$$

$$
h_\theta(x)=\theta^T{X}=\theta_0x_0+\theta_1x_1+…+\theta_nx_n
$$

**具体过程：**

1. 将代价函数$J(\theta)$的向量表达式转成矩阵表达式

$$
J(\theta) = \frac{1}{2}(X\theta-y)^2
$$

		- ​	X为m行n列（m个样本个数，n个特征个数）
		- $\theta$为n行1列的矩阵
		- y为m行1列的矩阵

2. $J(\theta)$做变换：

$$
\begin{align}
J(\theta)
& = \frac{1}{2}{(X\theta-y)}^T(X\theta-y) \\
& = \frac {1}{2}{(\theta^TX^T-y^T)(X\theta-y)} \\ 
& = \frac {1}{2}{(\theta^TX^TX\theta-\theta^TX^Ty-y^TX\theta+y^Ty)}
\end{align}
$$

3. 在进行求解偏导的过程中会用到的公式

$$
\frac {\partial AB}{\partial B} = A^T
$$

$$
\frac {\partial X^TAX}{\partial X}=2AX
$$

4. 求导

$$
\begin{align}
\frac{\partial J(\theta)}{\partial \theta}
& =\frac{1}{2}(2X^TX\theta-X^Ty-(y^TX)^T-0) \\
& = \frac{1}{2}(2X^TX\theta-X^Ty-X^Ty-0) \\
& = X^TX\theta-X^Ty \\

\end{align}
$$

令上面的导数等于0，得到$\theta$



#### Python实现

```python 
import numpy as np

def normalEquation(X, y):
  theta = np.linalg.inv(X.T@X)@X.T@Y   # X.T@X等价于X.T.dot(X)  @等价于.dot
  return theta
```

