超参数：
--model_name : name of the model to run
--dataset : name of the processed dataset to use
--optimizer : name of optimizer to use
--initializer : name of initializer to use
--lr : learning rate
--dropout : dropout rate in the training procedure
--l2reg : L2 regularization
--num_epoch : number of epoch to run
--batch_size
--log_step : number of step for logging
--patience : patience of epoch for early stop
--device : GPU to use
--seed : random seed
--macro_f1 : whether to use Macro-F1
--pre : whether to perform predicting


get_box:https://github.com/peteanderson80/bottom-up-attention
但是工具包caffe文件配置太老旧了，踩坑无数换个机器不一定成功，因此推荐使用的docker环境直接进行的配置。
Bottom-up的docker环境地址：https://hub.docker.com/r/airsplay/bottom-up-attention
此环境的使用说明：https://github.com/airsplay/lxmert
Docker使用说明（快速入门即可）：
https://zhuanlan.zhihu.com/p/344035406

VIT：https://github.com/lukemelas/PyTorch-Pretrained-ViT
这个配置比较简单，跟resnet的话都直接embedding一下就可以了

dataset：https://github.com/headacheboy/data-of-multimodal-sarcasm-detection

image_caption：
两者方案：1、https://github.com/ruotianluo/ImageCaptioning.pytorch#generate-image-captions
2、https://github.com/rmokady/CLIP_prefix_caption
推荐使用第二个方式来生成，第二个的效果较好，具体原因请看对应paper。不过之后大模型的效果可能会成为主流。