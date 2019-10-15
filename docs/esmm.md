Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate


essm是阿里妈妈提出，利用多目标学习思想优化淘宝cvr预估的一篇文章。

## 一、背景

电商系统中，用户的行为一般遵循曝光->点击->转化的链路行为。曝光到点击的比例叫做ctr，点击到转化的比例叫做cvr。传统的cvr预估存在两个重要问题：样本选择偏差以及数据稀疏。

##### 样本选择偏差（ssb）：
传统的cvr预估使用点击后的样本来训练，但模型却是在整个样本空间（曝光数据）做推断的。点击样本只是曝光样本的一个很小的子集，从中提取的特征是有偏差的。这种训练样本从整个样本空间的一个子集提取，而模型却要对整个样本空间的数据做预测的现象，我们称之为样本偏差。样本偏差一定程度上违背了机器学习独立同分布的假设，影响模型泛化能力。


##### 数据稀疏（ds）：
用户点击的商品数，远远小于用展示给用户的商品数；有点击行为的用户只占所有用户的一小部分。这种数据稀疏问题加大了模型训练的困难程度。


## 二、essm模型结构：

阿里妈妈借鉴多任务学习，提出了esmm模型，引入两个辅助的学习任务，分别拟合pctr和pctcvr，模型结构如图：

![image](https://cdn.kesci.com/images/batch_upload/4155986-e583e6dbf39b38d5.png)

essm又两个子网络组成，左边的用来拟合pctr，右边的拟合pcvr，两个子网络的输出相乘得到pctcvr。

`$pctcvr = pctr * pcvr$`

`pcvr = pctcvr/pctr`

假如用x表示样本，y表示点击，z表示转化，则：

`$p(y = 1, z = 1 | x) = p(y = 1 | x) * p(z = 1 | y = 1 , x)$`

esmm 主要有以下两个特点：

1. 在整个样本空间建模

由上式可知，pcvr可由pctcvr和pctr得出，ctr和ctcvr是在整个样本空间上训练的，一定程度上消除了样本选择偏差。我们将曝光点击数据作为正样本，曝光未点击作为副样本来训练ctr模型；将点击且转化作为正样本，其他作为副样本来训练ctcvr模型。模型的损失函数为：

![image](https://cdn.kesci.com/images/batch_upload/4155986-3543355f3bed04dc.png)

其中， l是交叉熵损失函数。

2. 共享特征表示

esmm借鉴迁移学习思路，两个子网络共享特征表示。网络的embedding层把大规模稀疏数据映射到低维的特征表示，这些embedding需要大量的数据才能得到充分训练。esmm这种共享特征表示，使得cvr任务的特征表示也可以从整个样本空间训练，极大地缓解来数据稀疏问题。


总结：esmm借鉴多目标学习思路，同时优化pctr和pctcvr，迂回地学习cvr，很好地解决了数据稀疏和样本选择偏差问题。另外，esmm的子网络可以轻松集成其他模型，吸取其他模型的有点。


### 参考文献
1. https://arxiv.org/abs/1804.07931 
2. http://xudongyang.coding.me/esmm/