#与Progressive Growing of GANs 的对比

##Progressive Growing of GANs

产生思想：观察到从潜图像到高分辨率的图像的复杂映射更容易通过逐步学习来实现。与此相关的方法有在不同空间分辨率上使用多重的discriminator（Wang, 2017）。在这种方法的启发下，使用单一的GAN替代多层GAN。同时推迟预配置层的引入，从某种意义上该方法类似于自动编码器的分层训练。

核心思想：渐进性生成generator和discriminator。从低的分辨率的图片开始，在训练的过程中通过逐渐增加新的、更加精细的细节来逐渐增加图片的分辨率。在增加并稳定学习速度的同时，产生高质量的图片。不同于同时学习所有尺度的图片，在这种增加机制下，学习从最初分辨图片分布的大范围结构开始，然后再把注意力放在越来越精细的细节上。

Generator 和 discriminator网络是彼此的镜像图片并且同步生长。在加入新的、分辨率更高的层次到网络中时，将分辨率低的图片逐渐平滑地删去可以避免对已经训练好的低分辨率的层次产生冲击。示意图如下

![image1](/Users/zhixingzhang/Documents/大三下/AI-Final-Project/media/image1.png)

如图为将分辨率从16×16提高到32×32的过程，把更高分辨率的层次当做是残差块，逐渐将比重从α从0线性增加到1。在训练discriminator时，输入符合现在网络分辨率的，缩小后的图片。在变换分辨率时与generator输出时如何组合两个分辨率类似，在真实图片的两个分辨率之间进行插值。

具体训练如下图所示

![1](/Users/zhixingzhang/Documents/大三下/AI-Final-Project/media/image2.png){width="5.763194444444444in" height="2.5027777777777778in"}

训练从generator（G）和discriminator（D）都具有低的分辨率，4×4像素开始。随着训练的进行，递增地向G与D增加层次细节，即增加生成图片的空间分辨率。通过对N×N空间分辨的图片进行卷积操作得到低分辨度的图片。这保证了高分辨率图片的稳定生成，同时大大加快了训练的速度。

Progressive Growing GANs 从低分辨率开始进行训练，有更少的信息和模式，生成上更加稳定。另一个优势是可以减少训练的时间，应为使用PGGANs时大部分的迭代都是在低分辨率的条件下进行的。同时在得到相近质量的结果的前提下，根据不同的分辨率，可以加速2到6倍。

##CGAN与PGGAN的对比

PGGAN与其他GAN模型相似，都属于纯无监督的GAN模型。而CGAN在生成模型和判别模型的建设中引入了条件变量y，类似于鉴别标签，从而将纯无监督模型转变为半监督模型。CGAN在训练方式上与PGGAN的渐进方式有所不同，但通过引入监督也可以起到稳定训练和加快训练速度的作用。

在同一数据集FashionMNIST下，分别使用PGGAN和CGAN实验结果对比如下。

![22](/Users/zhixingzhang/Documents/大三下/AI-Final-Project/media/image3.png){width="3.8881944444444443in" height="3.8881944444444443in"}

PGGAN训练结果

![33](/Users/zhixingzhang/Documents/大三下/AI-Final-Project/media/image4.png){width="5.0in" height="5.0in"}

CGAN训练结果

分别截取PGGAN与CGAN训练结果并放大进行对比。低分辨率图片下的训练信息量和模式都更少，使得训练稳定性更高。渐进提高训练分辨率的PGGAN通过在低分辨率下的训练对于图片的轮廓有很好的把握，训练结果中图片的外形结构明显好于CGAN的训练结果。但由于在低分辨率下的训练，对于后加入的高分辨率的细节并没有足够敏感。CGAN与之不同，与其他GAN模型相同直接使用实际分辨率图片进行训练。虽然在外轮廓上的处理不如PGGAN好，但通过增加了类似于标签的条件变量，转化成半监督学习，在细节上的处理更加清晰，对于高分辨率的细节更加敏感。

![](/Users/zhixingzhang/Documents/大三下/AI-Final-Project/media/image5.png){width="1.2722222222222221in" height="1.261111111111111in"}![](/Users/zhixingzhang/Documents/大三下/AI-Final-Project/media/image6.png){width="1.2597222222222222in" height="1.2597222222222222in"}

![](/Users/zhixingzhang/Documents/大三下/AI-Final-Project/media/image7.png){width="1.2597222222222222in" height="1.2597222222222222in"}![](/Users/zhixingzhang/Documents/大三下/AI-Final-Project/media/image8.png){width="1.2597222222222222in" height="1.2597222222222222in"}

(a) （c）PGGAN训练结果，（b）（d）CGAN训练结果