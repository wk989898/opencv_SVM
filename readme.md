# SVM参数简介
```
- SVM类型
(C_SVC、NU_SVC、ONE_CLASS、EPS_SVR、NU_SVR)
  -   C-SVC：C-支持向量分类机；参数C为惩罚系数，C越大表示对错误分类的惩罚越大，适当的参数C对分类Accuracy很关键。
  -  v-SVC：v-支持向量分类机；由于C的选取比较困难，用另一个参数v代替C。C是“无意义”的，v是有意义的。（与C_SVC其实采用的模型相同，但是它们的参数C的范围不同,C_SVC采用的是0到正无穷，该类型是[0,1]。）
  -   一类SVM：​单类别-支持向量机，不需要类标号,用于支持向量的密度估计和聚类。
  -   e -SVR：ε-支持向量回归机，不敏感损失函数，对样本点来说，存在着一个不为目标函数提供任何损失值的区域。
  -   v-SVR：n-支持向量回归机，由于EPSILON_SVR需要事先确定参数，然而在某些情况下选择合适的参数却不是一件容易的事情。而NU_SVR能够自动计算参数。
- kernel_type
  -  linear: u'*v  线性核函数   不需要参数
  -  polynomial: (gamma*u'*v + coef0)^degree   
   多项式核函数，三个参数，（gamma、coef0、degree）
  -  radial basis function: exp(-gamma*|u-v|^2)
   径向基函数(高斯核函数)，一个参数（gamma）
  - sigmoid: tanh(gamma*u'*v + coef0)    
   sigmoid核函数（多层感知机核函数），两个参数（gamma、coef0） 
  - precomputed kernel (kernel values in training_set_file)
-  gamma : set gamma in kernel function 
 多项式核函数、RBF核函数、sigmoid核函数。默认是1/num_features，因为特征往往比较多，所以使用时为0
- set the parameter C of C-SVC, epsilon-SVR, and nu-SVR 
 C-SVC使用的惩罚因子C，epsilon-SVR, and nu-SVR中也使用 ，默认值为1
 ```

--- 
 (1)、cv::ml::SVM类：继承自cv::ml::StateModel，而cv::ml::StateModel又继承自cv::Algorithm;

  (2)、create函数：为static，new一个SVMImpl用来创建一个SVM对象；

 (3)、setType/getType函数：设置/获取SVM公式类型，包括C_SVC、NU_SVC、ONE_CLASS、EPS_SVR、NU_SVR，用于指定分类、回归等，默认为C_SVC；

 (4)、setGamma/getGamma函数：设置/获取核函数的γ参数，默认值为1；

 (5)、setCoef0/getCoef0函数：设置/获取核函数的coef0参数，默认值为0；

 (6)、setDegree/getDegree函数：设置/获取核函数的degreee参数，默认值为0；

 (7)、setC/getC函数：设置/获取SVM优化问题的C参数，默认值为0；

 (8)、setNu/getNu函数：设置/获取SVM优化问题的υ参数，默认值为0；

 (9)、setP/getP函数：设置/获取SVM优化问题的ε参数，默认值为0；

 (10)、setClassWeights/getClassWeights函数：应用在SVM::C_SVC,设置/获取weights，默认值是空cv::Mat；

 (11)、setTermCriteria/getTermCriteria函数：设置/获取SVM训练时迭代终止条件，默认值是cv::TermCriteria(cv::TermCriteria::MAX_ITER + TermCriteria::EPS,1000, FLT_EPSILON)；

 (12)、setKernel/getKernelType函数：设置/获取SVM核函数类型，包括CUSTOM、LINEAR、POLY、RBF、SIGMOID、CHI2、INTER，默认值为RBF；

 (13)、setCustomKernel函数：初始化CUSTOM核函数；

 (14)、trainAuto函数：用最优参数训练SVM；

 (15)、getSupportVectors/getUncompressedSupportVectors函数：获取所有的支持向量；

 (16)、getDecisionFunction函数：决策函数；

 (17)、getDefaultGrid/getDefaultGridPtr函数：生成SVM参数网格；

 (18)、save/load函数：保存/载入已训练好的model，支持xml,yaml,json格式；

 (19)、train/predict函数：用于训练/预测，均使用基类StatModel中的。