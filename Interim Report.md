# 人工智能导论中期报告

**成员**： 钟昊东， 朱文轩， 张知行

**项目设计**：

​	实现一个生成对抗网络来处理FashionMNIST数据集并通过训练实现高效的同类图片生成。目前的神经网络设计如下图![pytorch_cGAN](/Users/zhixingzhang/Documents/大三下/AI-Final-Project/pytorch_cGAN.png)



**当前进度**：

​	目前已经完成基本的GAN网络设计与实现，程序可以正常运转，但是参数设定仍然需要进一步调整，使得生成的图片更为细致，当前训练结果如下图：

![FashionMNIST_200](/Users/zhixingzhang/Documents/大三下/AI-Final-Project/FashionMNIST_200.png)

​	上图是经过200次迭代之后生成的图片，可以看出其中第六行图片仍然不够清晰，而且整体上看来图片较为粗糙，和原始数据集的差别较大。在经过500迭代之后，训练结果为分辨器的BCELoss为0.13。

**后续计划**：

​	首先，我们计划进一步改善项目中的超参数设定，通过对学习速率的调整来进一步改进我们产生的图片的质量。同时，受到Progressive Growing of GANs for Improved Quality, Stability, and Variation的启发，我们计划在后续参数调整结束后，如果时间充足，我们会进一步改进我们的神经网络模型，借鉴该研究组的渐进式生成对抗网络设置思路，从而进一步改善我们的模型表现。