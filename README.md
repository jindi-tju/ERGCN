# ERGCN
Source code of Multi-Modal Sarcasm Detection Based on Cross-Modal Composition of Inscribed Entity Relations (ICTAI23)

get_box:
https://github.com/peteanderson80/bottom-up-attention
但是工具包caffe文件配置太老旧了，踩坑无数换个机器不一定成功，因此推荐使用的docker环境直接进行的配置。

Bottom-up的docker环境地址：
https://hub.docker.com/r/airsplay/bottom-up-attention
此环境的使用说明：https://github.com/airsplay/lxmert
Docker使用说明（快速入门即可）：
https://zhuanlan.zhihu.com/p/344035406

VIT：
https://github.com/lukemelas/PyTorch-Pretrained-ViT
这个配置比较简单，跟resnet的话都直接embedding一下就可以了

dataset：
https://github.com/headacheboy/data-of-multimodal-sarcasm-detection

image_caption：
两者方案:
1、https://github.com/ruotianluo/ImageCaptioning.pytorch#generate-image-captions
2、https://github.com/rmokady/CLIP_prefix_caption

推荐使用第二个方式来生成，第二个的效果较好，具体原因请看对应paper。不过之后大模型的效果可能会成为主流。
