# Intro
  基于opencv的svm做的图像分类
  数据集使用[The CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html)
### SVM参数简介
```
- SVM类型
(C_SVC、NU_SVC、ONE_CLASS、EPS_SVR、NU_SVR)
  -   C-SVC：C-支持向量分类机；参数C为惩罚系数，C越大表示对错误分类的惩罚越大，适当的参数C对分类Accuracy很关键。
  -   v-SVC：v-支持向量分类机；由于C的选取比较困难，用另一个参数v代替C。C是“无意义”的，v是有意义的。（与C_SVC其实采用的模型相同，但是它们的参数C的范围不同,C_SVC采用的是0到正无穷，该类型是[0,1]。）
  -   一类SVM：​单类别-支持向量机，不需要类标号,用于支持向量的密度估计和聚类。
  -   e-SVR：ε-支持向量回归机，不敏感损失函数，对样本点来说，存在着一个不为目标函数提供任何损失值的区域。
  -   v-SVR：n-支持向量回归机，由于EPSILON_SVR需要事先确定参数，然而在某些情况下选择合适的参数却不是一件容易的事情。而NU_SVR能够自动计算参数。
- kernel_type(LINEAR POLY RBF SIGMOID INTER CHI2)
  -  linear: u'*v  线性核函数   不需要参数
  -  polynomial: (gamma*u'*v + coef0)^degree   
   多项式核函数，三个参数，（gamma、coef0、degree）
  -  radial basis function: exp(-gamma*|u-v|^2)
   径向基函数(高斯核函数)，一个参数（gamma）
  - sigmoid: tanh(gamma*u'*v + coef0)    
   sigmoid核函数（多层感知机核函数），两个参数（gamma、coef0） 
  - precomputed kernel (kernel values in training_set_file)
 ```