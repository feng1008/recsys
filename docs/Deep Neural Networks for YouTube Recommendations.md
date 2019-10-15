# Deep Neural Networks for YouTube Recommendations

Youtube采取了两层深度网络完成整个推荐过程：


第一层是Candidate Generation Model 完成候选视频的快速筛选，这一步候选视频集合由百万降低到了百的量级。

第二层是用Ranking Model完成几百个候选视频的精排，作为最后的输出结果。


### 一、召回

![image](https://cdn.kesci.com/images/batch_upload/4155986-1104f822da782760.png)
- 从左到右，特征分别为用户观看过的视频的embedding，用户搜索过的视频embedding，地理位置信息，人口统计信息

召回层可以看成是拿用户历史观看、搜索的视频，预测下一个观看的视频，并得出user 与item 的向量。与word2vec有点类似，不同的是，youtube Dnn还可以加入用户侧的side imformation，可以将任意的连续特征和离散特征引入来做用户的embedding。

##### 基本思想：
利用深度神经网络，结合用户基础属性与上下文特征，以用户特征为输入，经过多层神经网络表达成user vector，然后以nce_weight作为item向量，通过一个softmax函数，预测下一个观看的视频，并以此得到user vector和item vector。在召回阶段，利用最近邻搜索，从数百万个视频中挑选与用户最近的几百个视频作为召回结果。

1. 采样：
    a. 不仅仅使用推荐数据，还使用其他页面观看数据
    b. 每个用户生成固定数量训练样本，防止部分活跃用户对模型的主导
    c. 抛弃序列特征
    d. 不对称的共同浏览问题，预测用户的下一次观看比传统的留一预测更有效。
2. 特征：包括用户观看过、搜索过的视频id，经过embedding之后再进行average pool拼接起来，再引入一些人口统计学特征，Example Age等。
3. 网络：三层relu，没有什么不同。
4. softmax层与user vector和item vector：文章使用relu的最后一层作为user vector, 因此输入也都是与用户侧的特征，不加入item侧特征。以softmax的nce_weight作为item的特征，通过user_vector与item vector的内积，得出用户对每个item的概率。
5. 预测目标与损失函数：如4所述，神经网络预测的是用户对每个item的得分，因此预测目标为用户感兴趣的item的id。损失函数是nce_loss。

召回层最有新意的是引入来Example age这个特征。作者发现，用户更倾向于观看那些相关度不高但比较新的视频。假如一个视频是十天前上传的，那在当天会有很多sample log，而在之后的时间sample log 会比较少。原文没有明确解释Example age是如何得到的，但一般理解成 训练时间-sample log 时间。而在线上服务，这个特征被赋值为0.


线上服务：

离线保存user vector 和 item vector，通过最近邻搜索得到每个用户的top n的结果。

### 二、排序

![image](https://cdn.kesci.com/images/batch_upload/4155986-2e46cf2dccca4760.png)
---
- 从左至右的特征一次为：当前要计算的视频的embedding，用户观看过的最后N个视频embedding的average pooling， 用户语言的embedding和当前视频语言的embedding， 自上次观看同channel视频的时间， 该视频已经被曝光给该用户的次数

##### 排序层和召回层区别：
1. 特征上排序层可以引入视频侧特征，且特征更加精细化。而召回层特征最后是要表达成user vector，不要用视频侧的特征
2. 预测目标不再是下一个要观看的视频id，而是拟合一个回归，预测观看时长，使用weighted LR作为排序网络的输出层。


##### 特征工程：

Embedding Categorical Features：神经网络更适合处理连续稠密特征，id类特征embedding之后再输入到模型；相同id空间下的embedding可以共享。

Normalizing Continuous Features：神经网络对特征的scaling以及输入分布很敏感，连续特征最好归一化；也可以通过取根号、平方等方法扩展特征。

##### 预测观看时长：

为什么使用weighted LR作为输出层，而且没有使用sigmoid来预测正样本的概率，使用`$e^{wx+b}$` 来预测用户的观看时长？？

传统的神经网络架构，输出层一般为LR或者softmax，label为0或者1，预测过程是计算该样本为正样本的概率。youtube 采用Weighted LR， 使用`$e^{wx+b}$`来预测用户观看时长。

Weighted LR：
weight对于正样本来说，就是观看时长Ti，负样本则为1。Weighted LR的特点是正样本权重w的加入会让正样本发生的几率变成原来的w倍

`$Odds(i) = \frac{w_ip}{1 - w_ip}  \approx w_ip = T_ip $ `

由于YouTube采用了用户观看时长Ti作为权重， 即`$w_i = T_ip$` ，因此`$odds$` 即为用户观看某一视频的期望。因此，YouTube采用`$e^{wx+b}$` 这一指数形式预测的就是曝光这个视频时，用户观看这个视频的时长的期望。

训练Weighted LR：
1. 正样本按weight做重复sampling，输入模型进行训练；
2. 梯度下降过程中，通过改变梯度的weight来得到Weighted LR。

---
### 问题：

###### 1. 预测next watch，总共分类数百万，如何解决效率？

负采样并用importance weighting 进行校准。


###### 2. serving过程为什么不采用训练的model进行预测，而使用最近邻搜索的方法？

工程与学术的trade-off，在得到user和item 的vector之后，使用最近邻搜索的方法效率高很多。

user vector 和video vector 怎么产生？
如上图，relu的最后一层为user vector, nce_veight 为video vector。


###### 3. 用户对新视频的偏好，如何引入这个特征？
example age，把sample log距今的时间作为example age.

###### 4. 对每个用户提取等量的训练样本的原因？

减少高度活跃用户对于loss的过度影响。

###### 5. 完全摒弃用户观看历史的时序特征的原因？
过多考虑时序特征，推荐结果将过多受最近观看或搜索的一个视频的影响。youtube将用户近期的访问记录等同看待，属于业务上的问题。

###### 6. 处理测试集的时候，为什么不采取经典的随机留一法，而把最近的一次观看行为作为测试集？
避免引入feature imformation，造成数据穿越。

###### 7.为什么不采用CTR或者播放率，而使用预期播放时间作为优化目标？

商业问题，watch time更能反应用户的真实兴趣，越长收益越多。

###### 8. video embedding 为什么把大量长尾video用0代替

节省online serving的资源浪费，低频embedding的准确性也难以保障。

###### 9. previous impressions 为什么进行开方和平方处理

引入特征非线性。

###### 10. 为什么使用weighted LR?

将watch time作为正样本的weight，在线上serving中使用`$e^{wx+b}$`做预测可以直接得到expected watch time的近似。在上节已经重点介绍。


##### YouTube Ranking Model的Serving过程要点：

1. 线上serving e wx+b预测的是weight LR的odds
2. Weighted LR使用用户观看时长作为权重，使得对应的Odds表示的就是用户观看时长的期望
3. 使用观看时长加权后，预测的就是观看市场的期望




### #### Reference
1. https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf
2. https://zhuanlan.zhihu.com/p/51440316
3. https://zhuanlan.zhihu.com/p/25343518
4. https://zhuanlan.zhihu.com/p/52504407
5. https://zhuanlan.zhihu.com/p/52169807

