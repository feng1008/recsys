Deep & Cross Network for Ad Click Prediction

预测模型通常需要手动进行特征工程或遍历搜索。google 2017 提出的dcn能对sparse和dense的输入自动学习特征交叉，有效地捕获有限阶上的有效特征交叉，无需进行人工特征工程或者暴力搜索。

论文主要贡献：

1. 提出一种新的交叉网络，每层有效地应用特征交叉
2. 跨网络简单而有效，每层的多项式级数最高，并由层深度决定
3. 跨网络内存高效，易于实现
4. DNN相比少了近一个量级的参数量


### 原理

dcn的模型结构如下：

![image](https://cdn.kesci.com/images/batch_upload/4155986-aa7d9d209c6cf3d5.png)

首先dense特征和sparse特征embedding之后的特征拼接起来作为输入`$x_0$`, 然后是两个平行的网络深度网络和交叉网络，最后结合两者的输出。

输入层：sparse特征转化成embedding向量，与dense特征拼接起来作为输入

深度网络：全连接的前馈神经网络  `$h_{l+1} = f(w_lh_l + b_l)$`

交叉网络：

交叉网络是本文的核心，其每一层计算公式如下：

`$x_{l+1} = x_0x_l^Tw_l + b_l + x_l = f(x_l, w_l, b_l) + x_l$`

![image](https://cdn.kesci.com/images/batch_upload/4155986-8de32b6dab68c108.png)

交叉特征的程度随网络深度的增加而增大

交叉网络的参数： 输入维度d * 交叉层数 `$L_c$` * 2 (w, b)

Combination 层：

`$p = \sigma([x_{L_1}^T, h_{L_2}^T]W_logits)$`

损失函数：

![image](https://cdn.kesci.com/images/batch_upload/4155986-6a3cad235da5dd61.png)



### 参考文献
1. https://arxiv.org/abs/1708.05123
2. https://blog.csdn.net/roguesir/article/details/79763204
3. https://www.kesci.com/home/project/5d176e081951a9002c7d2cee/code

