
# Awesome of Computer Vision Resources

  A curated list of resources dedicated to Face Recognition & Detection, OCR, Objection Detection, Gan, 3D, Motion Track & Pose Estimation, ReID, NAS, Recommentation, Model Scaling. Any suggestions and pull requests are welcome.


Table of Contents
=================

   * [Awesome of Computer Vision Resources](#awesome-of-computer-vision-resources)
   * [Table of Contents](#table-of-contents)
         * [ReID](#reid)
         * [Gan](#gan)
         * [NAS](#nas)
         * [Classification](#classification)
         * [Recommendation &amp; CTR](#recommendation--ctr)
            * [CTR](#ctr)
            * [Recommendation](#recommendation)
         * [Video Processing](#video-processing)
            * [Classification](#classification-1)
            * [Augumentation](#augumentation)
         * [SLAM](#slam)
         * [Building and Training](#building-and-training)
            * [Optimizing](#optimizing)
            * [Constructure](#constructure)
            * [Strategy](#strategy)
            * [Evaluation](#evaluation)
         * [Body Related](#body-related)
            * [Face Detection](#face-detection)
            * [Face Alignment](#face-alignment)
            * [Head Detection](#head-detection)
            * [Liveness Detection](#liveness-detection)
            * [3D Face](#3d-face)
         * [Data Processing](#data-processing)
            * [Super resolution](#super-resolution)
            * [Synthesis](#synthesis)
            * [Image Translation](#image-translation)
            * [Date augmentaiton](#date-augmentaiton)
         * [Objection Detection &amp; Semantic](#objection-detection--semantic)
            * [Objection Detection](#objection-detection)
            * [Salient Object Detecion](#salient-object-detecion)
            * [Segmentation](#segmentation)
         * [Model Compress and Accelerate](#model-compress-and-accelerate)
            * [Pruning](#pruning)
            * [Accelerating](#accelerating)
         * [Motion &amp; Pose](#motion--pose)
            * [Pose Estimation](#pose-estimation)
            * [Pose Transfer](#pose-transfer)
            * [Motion Track](#motion-track)
            * [Action Recognition](#action-recognition)
            * [Keypoint Detection](#keypoint-detection)
         * [Text Detection &amp; Recognition](#text-detection--recognition)
            * [Detection](#detection)
            * [Recogination](#recogination)
 

### ReID
- [2019-CVPR] MAR: Unsupervised Person Re-identification by Soft Multilabel Learning [`paper`](https://arxiv.org/abs/1903.06325) [`code`](https://github.com/KovenYu/MAR)
- [2019-CVPR] Bags of Tricks and A Strong Baseline for Deep Person Re-identification(Baseline) [`paper`](https://arxiv.org/abs/1903.07071) [`code`](https://github.com/michuanhaohao/reid-strong-baseline) [`paper`](https://arxiv.org/abs/1901.06140) 
- [2019-CVPR] Backbone Can Not be Trained at Once: Rolling Back to Pre-trained Network for Person Re-IdentificationRolling Back to Pre-trained Network for Person Re-Identification [`paper`](https://arxiv.org/abs/1901.06140) [`code`](https://github.com/youngminPIL/rollback)
- [2019-CVPR] Invariance Matters: Exemplar Memory for Domain Adaptive Person Re-identification [`paper`](https://arxiv.org/abs/1904.01990) [`code`](https://github.com/zhunzhong07/ECN)
- [2019-CVPR] An Adaptive Training-less System for Anomaly Detection in Crowd Scenes [`paper`](https://arxiv.org/abs/1906.00705)
- [2019-CVPR] DBC: Dispersion based Clustering for Unsupervised Person Re-identification [`paper`](https://arxiv.org/abs/1906.01308) [`code`](https://github.com/gddingcs/Dispersion-based-Clustering)
- [2019-CVPR] SSA-CNN: Semantic Self-Attention CNN for Pedestrian Detection(SOTA) [`paper`](https://arxiv.org/abs/1902.09080)
- [2018-CVPR] Attention-Aware Compositional Network for Person Re-identification(with pose Information) [`paper`](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Attention-Aware_Compositional_Network_CVPR_2018_paper.pdf)

### Gan 
- [`collection`] Awesome Generative Adversarial Networks with tensorflow**[`code`](https://github.com/kozistr/Awesome-GANs)
- [`framework`] Implementations of a number of generative models GAN, VAE, Seq2Seq, VAEGAN, GAIA, Spectrogram Inversion in Tensorflow** [`code`](https://github.com/timsainb/tensorflow2-generative-models)
- [2019-CVPR] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [`paper`](https://arxiv.org/abs/1810.04805) [https://github.com/google-research/bert](https://github.com/google-research/bert) [`code-pytorch`](https://github.com/huggingface/pytorch-pretrained-BERT)
- [2019-CVPR] StyleGan: Generator Inversion for Image Enhancement and Animation [`paper`](https://arxiv.org/abs/1906.11880)[`code`](https://github.com/avivga/style-image-prior)
- [2018-ICLR] Progressive Growing of GANs for Improved Quality, Stability, and Variation [`paper`](https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf) [`code`](https://github.com/tkarras/progressive_growing_of_gans#progressive-growing-of-gans-for-improved-quality-stability-and-variation-official-tensorflow-implementation-of-the-iclr-2018-paper)) 

### NAS 
- [`framework`] An open source AutoML toolkit for neural architecture search and hyper-parameter tuning [`code`](https://github.com/microsoft/nni)
- [2019-CVPR] AutoGrow: Automatic Layer Growing in Deep Convolutional Networks [`paper`](https://arxiv.org/abs/1906.02909) [`code`](https://github.com/wenwei202/autogrow)
- [2019-ar Xiv] MDENAS: Multinomial Distribution Learning for Effective Neural Architecture Search [`paper`](https://arxiv.org/abs/1905.07529) [`code`](https://github.com/tanglang96/MDENAS)
- [2019-CVPR] MnasNet: Platform-Aware Neural Architecture Search for Mobile [`paper`](https://arxiv.org/pdf/1807.11626.pdf) [`code`](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet)
- [2019-CVPR] Searching for A Robust Neural Architecture in Four GPU Hours [`paper`](http://https://github.com/D-X-Y/GDAS/blob/master/data/GDAS.pdf) [`code`](https://github.com/D-X-Y/GDAS)
- [2019-arXiv] Single-Path Mobile AutoML: Efficient ConvNet Design and NAS Hyperparameter Optimization [`paper`](https://arxiv.org/abs/1907.00959) [`code`](https://github.com/dstamoulis/single-path-nas)
- [2019-CVPR] Dynamic Distribution Pruning for Efficient Network Architecture Search [`paper`](https://arxiv.org/abs/1905.13543) [`code`](https://github.com/tanglang96/DDPNAS)

### Classification
- [`ToolBox`] Sandbox for training convolutional networks for computer vision (VGG,ResNet,PreResNet,ResNeXt,SENet,ResAttNet,SKNet,PyramidNet,
- DenseNet,BagNet,MSDNet,FishNet,SqueezeNet,SqueezeResNet,SqueezeNext,ShuffleNet,ShuffleNetV2,MENet,MobileNet,FD-MobileNet,MobileNetV2,MobileNetV3,
Xception,InceptionV3,InceptionV4,InceptionResNetV2,PolyNet,NASNet-Mobile,PNASNet-Large,EfficientNet) [`code`](https://github.com/osmr/imgclsmob)
- [2019-CVPR] RepMet: Representative-based metric learning for classification and one-shot object detection [`paper`](https://arxiv.org/abs/1806.04728)
- [2018-CVPR] SENet: Squeeze-and-Excitation Networks(champion for imageNet) [`paper`](https://arxiv.org/abs/1709.01507) [`code`](https://github.com/moskomule/senet.pytorch) [`code-caffe`](https://github.com/hujie-frank/SENet)
- [2018-CVPR] FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction [`paper`](http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf) [`code`](https://github.com/kevin-ssy/FishNet)

### Recommendation & CTR
- [`ToolBox`] Implementation of Deep Learning based Recommender Algorithms with Tensorflow [`code`](https://github.com/cheungdaven/DeepRec)
- [`ToolBox`] A framework for training and evaluating AI models on a variety of openly available dialogue datasets [`code`](https://github.com/facebookresearch/ParlAI)
- [`ToolBox`] StarSpace: Embed All The Things! [`paper`](https://arxiv.org/abs/1709.03856) [`code`](https://github.com/facebookresearch/StarSpace)
- [`ToolBox`] Modular and Extendible package of deep-learning based CTR models [`code`](https://github.com/shenweichen/DeepCTR)
- [`collection`] Classic papers and resources on recommendation [`papers`](https://github.com/wzhe06/Reco-papers)
- [`collection`] A collection of resources for Recommender Systems [`papers`](https://github.com/chihming/competitive-recsys)
- [`collection`] papers,datas,outline for recommendation [`code`](https://github.com/zhaozhiyong19890102/Recommender-System)
#### CTR
- [2019-arXiv] Deep Learning Recommendation Model for Personalization and Recommendation Systems(***CTR) [`paper`](https://arxiv.org/abs/1906.00091)[`code`](https://github.com/facebookresearch/dlrm)
#### Recommendation
- [2019-arXiv] Generative Adversarial User Model for Reinforcement Learning Based Recommendation System [`paper`](https://arxiv.org/pdf/1812.10613.pdf)
- [2019-arXiv] Recent Advances in Diversified Recommendation [`paper`](https://arxiv.org/pdf/1905.06589.pdf)
- [2017-arXiv] Training Deep AutoEncoders for Collaborative Filtering(***SOTA) [`paper`](https://arxiv.org/abs/1708.01715) [`code`](https://github.com/NVIDIA/DeepRecommender)

### Video Processing

#### Classification
- [2019-CVPR] Video Classification [`paper`](https://arxiv.org/pdf/1703.10593.pdf) [`code`](https://github.com/HHTseng/video-classification
- [2019-CVPR] FastDVDnet: Towards Real-Time Video denoising Without Explicit Motion Estimation(denoising) [`paper`](https://arxiv.org/abs/1907.01361) [`code`](https://github.com/m-tassano/fastdvdnet)
- [2019-CVPR] Hallucinating Optical Flow Features for Video Classification [`paper`](https://arxiv.org/abs/1905.11799v1 ") [`code`](https://github.com/YongyiTang92/MoNet-Features)

#### Augumentation
- [2019-CVPR] DAVANet: Stereo Deblurring with View Aggregation(debluring) [`paper`](https://arxiv.org/pdf/1904.05065.pd) [`code`](https://github.com/sczhou/DAVANet)
- [2019-CVPR] DVDnet: A Simple and Fast Network for Deep Video Denoising(***SOTA) [`paper`](https://arxiv.org/abs/1906.11890) [`code`](https://github.com/m-tassano/dvdnet)
- [2019-CVPR] Deep Flow-Guided Video Inpainting [`paper`](https://arxiv.org/pdf/1905.02884.pdf) [`code`](https://nbei.github.io/video-inpainting.html)
- [2019-CVPR] EDVR: Video Restoration with Enhanced Deformable Convolutional Networks [`paper`](https://arxiv.org/abs/1905.02716v) [`code`](https://github.com/xinntao/EDVR)
- [2019-CVPR] FastDVDnet: Towards Real-Time Video denoising Without Explicit Motion Estimation(denoising) [`paper`](https://arxiv.org/abs/1907.01361) [`code`](https://github.com/m-tassano/fastdvdnet)
- [2019-CVPR] TecoGAN: Temporally Coherent GANs for Video Super-Resolution [`paper`]( https://arxiv.org/pdf/1811.09393.pdf) [`code`](https://github.com/thunil/TecoGAN)
- [2018-XXXX] A Deep Learning based project for colorizing and restoring old images and video!(***) [`code`](https://github.com/jantic/DeOldify)

### SLAM

- [2019-CVPR] AdaptForStereo: Learning to Adapt for Stereo [`paper`](https://arxiv.org/abs/1904.02957) [`code`](https://github.com/CVLAB-Unibo/Learning2AdaptForStereo)
- [2019-arXiv] DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction [`paper`](https://arxiv.org/abs/1905.10711) [`code`](https://github.com/laughtervv/DISN)
- [2019-CVPR] Detailed Human Shape Estimation from a Single Image by Hierarchical Mesh Deformation [`paper`](https://arxiv.org/abs/1904.10506) [`code`](https://github.com/zhuhao-nju/hmd)
- [2019-CVPR] Defusr: Learning Non-volumetric Depth Fusion using Successive Reprojections [`code`](https://github.com/simon-donne/defusr)
- [2019-IEEE] FANTrack: 3D Multi-Object Tracking with Feature Association Network [`paper`](https://arxiv.org/abs/1905.02843) [`code`](https://git.uwaterloo.ca/wise-lab/fantrack)
- [2019-CVPR] GA-Net: Guided Aggregation Net for End-to-end Stereo Matching [`paper`](https://arxiv.org/abs/1904.06587) [`code`](https://github.com/feihuzhang/GANet)
- [2019-CVPR] Joint Monocular 3D Vehicle Detection and Tracking(***) [`paper`](https://arxiv.org/abs/1811.10742) [`code`](https://github.com/ucbdrive/3d-vehicle-tracking)
- [2019-CVPR] Leveraging Shape Completion for 3D Siamese Tracking [`paper`](https://arxiv.org/pdf/1903.01784.pdf) [`code`](https://github.com/SilvioGiancola/ShapeCompletion3DTracking)
- [2019-CVPR] Neural Rerendering in the Wild [`paper`](https://arxiv.org/abs/1904.04290) [`code`](https://github.com/google/neural_rerendering_in_the_wild)
- [2019-CVPR] PyRobot: An Open-source Robotics Framework for Research and Benchmarking [`paper`](https://arxiv.org/abs/1906.08236) [`code`](https://github.com/facebookresearch/pyrobot)
- [2019-CVPR] Pixel-Accurate Depth Evaluation in Realistic Driving Scenarios [`paper`](https://arxiv.org/abs/1906.08953)
- [2019-CVPR] Pose from Shape: Deep Pose Estimation for Arbitrary 3D Objects [`paper`](https://arxiv.org/abs/1906.05105)
- [2019-CVPR] Robust Point Cloud Based Reconstruction of Large-Scale Outdoor Scenes(3D reconstruction) [`paper`](https://arxiv.org/abs/1905.09634) [`code`](https://github.com/ziquan111/RobustPCLReconstruction)
- [2019-CVPR] SGANVO: Unsupervised Deep Visual Odometry and Depth Estimation with Stacked Generative Adversarial Networks [`paper`](https://arxiv.org/abs/1906.08889)
- [2019-CVPR] Taking a Deeper Look at the Inverse Compositional Algorithm(image alignment) [`paper`](https://arxiv.org/pdf/1812.06861.pdf) [`code`](https://github.com/lvzhaoyang/DeeperInverseCompositionalAlgorithm) 

### Building and Training
- [`ToolBox`] Pretrained EfficientNet, MobileNetV3 V2 and V1, MNASNet A1 and B1, FBNet, ChamNet, Single-Path NAS [`code`](https://github.com/rwightman/gen-efficientnet-pytorch)

#### Optimizing
- [2019-CVPR] Aggregation Cross-Entropy for Sequence Recognition (The ACE loss function exhibits competitive performance to CTC)  [`paper`](https://arxiv.org/abs/1904.08364) [`code`](https://github.com/summerlvsong/Aggregation-Cross-Entropy) 
- [2019-CVPR] KL-Loss: Bounding Box Regression with Uncertainty for Accurate Object Detection [`paper`](https://arxiv.org/abs/1809.08545) [`code`](https://github.com/yihui-he/KL-Loss) 

#### Constructure
- [2019-CVPR] Pacnet: Pixel-Adaptive Convolutional Neural Networks(new net constructure) [`paper`](https://arxiv.org/abs/1904.05373) [`code`](https://github.com/NVlabs/pacnet)
- [2019-CVPR] ViP: Virtual Pooling for Accelerating CNN-based Image Classification and Object Detection [`paper`](https://arxiv.org/abs/1906.07912)

#### Strategy
- [`Toolbox`] A Python Package to Tackle the Curse of Imbalanced Datasets in Machine Learning [`code`](https://github.com/scikit-learn-contrib/imbalanced-learn)
- [2019-CVPR]mixup: Bag of Freebies for Training Object Detection Neural Networks [`paper`](https://arxiv.org/abs/1902.04103) [`code`](https://github.com/dmlc/gluon-cv)
- [2019-CVPR] Improving Transferability of Adversarial Examples with Input Diversity [`paper`](https://arxiv.org/abs/1803.06978) [`code`](https://github.com/cihangxie/DI-2-FGSM)
- [2018-CVPR] Fd-mobilenet: Improved mobilenet with a fast downsampling strategy [`paper`](https://arxiv.org/abs/1802.03750?context=cs.CV) [`code`](https://github.com/osmr/imgclsmob)

#### Evaluation
- [2019-CVPR] TedEval: A Fair Evaluation Metric for Scene Text Detectors(***) [`paper`](https://arxiv.org/abs/1907.01227) [`code`](https://github.com/clovaai/TedEval) 
- [2019-CVPR] Tools for evaluating and visualizing results for the Multi Object Tracking and Segmentation (MOTS) [`paper`](https://www.vision.rwth-aachen.de/media/papers/mots-multi-object-tracking-and-segmentation/MOTS.pdf) [`code`](https://github.com/VisualComputingInstitute/mots_tools)
 
### Body Related 
- [`collection`] A curated list of related resources for hand pose estimation** [`code`](https://github.com/xinghaochen/awesome-hand-pose-estimation) 
- [`collection`] Face Benchmark and Dataset [`code`](https://github.com/becauseofAI/HelloFace)
- [`ToolBox`]  A face recognition solution on mobile device [`code`](https://github.com/becauseofAI/MobileFace)

#### Face Detection 
- [2019-CVPR] Dense 3D Face Decoding over 2500FPS: Joint Texture & Shape Convolutional Mesh Decoders [`paper`](https://arxiv.org/abs/1904.03525?context=cs.CV)
- [2019-CVPR] DSFD: Dual Shot Face Detector [`paper`](https://arxiv.org/abs/1810.10220) [`code`](https://github.com/TencentYoutuResearch/FaceDetection-DSFD)
- [2019-CVPR] SRN: Improved Selective Refinement Network for Face Detection(SOTA) [`paepr`](https://arxiv.org/abs/1901.06651) [`code`](https://github.com/ChiCheng123/SRN)

#### Face Alignment
- [2018-arXiv] Face Alignment: How far are we from solving the 2D & 3D Face Alignment problem [`paper`](https://arxiv.org/pdf/1703.07332.pdf) [`code`](https://github.com/1adrianb/face-alignment)
- [2018-ECCV] Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network [`code`](https://github.com/YadiraF/PRNet)

#### Head Detection
- [2018-arXiv] FCHD: Fast and accurate head detection in crowded scenes [`paper`](https://arxiv.org/abs/1809.08766) [`code`](https://github.com/aditya-vora/FCHD-Fully-Convolutional-Head-Detector)

#### Liveness Detection
- [2019-CVPR] A Non-Intrusive Method of Face Liveness Detection Using Specular Reflection and Local Binary Patterns(Liveness Detection) [`paper`](https://arxiv.org/abs/1905.06540)
- [2019-CVPR] FeatherNets: Convolutional Neural Networks as Light as Feather for Face Anti-spoofing(***Anti-spoofing) [`paper`](https://arxiv.org/abs/1904.09290) [`code`](https://github.com/SoftwareGift/FeatherNets_Face-Anti-spoofing-Attack-Detection-Challenge-CVPR2019 )
- [2019-CVPR] Liveness Detection Using Implicit 3D Features [`paper`](https://arxiv.org/abs/1804.06702)


#### 3D Face
- [2019-CVPR] Disentangled Representation Learning for 3D Face Shape(3D face) [`paper`](https://arxiv.org/abs/1902.09887) [`code`](https://github.com/zihangJiang/DR-Learning-for-3D-Face)
- [2019-CVPR] Expressive Body Capture: 3D Hands, Face, and Body From a Single Image [`paper`](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/497/SMPL-X.pdf) [`code`](https://github.com/vchoutas/smplify-x)
- [2019-CVPR] Learning to Regress 3D Face Shape and Expression From an Image Without 3D Supervision [`paper`](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/509/paper_camera_ready.pdf) [`code`](https://github.com/soubhiksanyal/RingNet)
- [2019-CVPR] Monocular Total Capture: Posing Face, Body and Hands in the Wild [`paper`](https://arxiv.org/abs/1812.01598) [`code`](https://github.com/CMU-Perceptual-Computing-Lab/MonocularTotalCapture)
- [2019-CVPR] MVF-Net: Multi-View 3D Face Morphable Model Regression(face reconstructing) [`code`](https://github.com/Fanziapril/mvfnet)


### Data Processing

#### Super resolution
- [2019-CVPR] AdaFM: Modulating Image Restoration with Continual Levels via Adaptive Feature Modification Layers(denoising) [`paper`](https://arxiv.org/abs/1904.08118) [`code`](https://github.com/hejingwenhejingwen/AdaFM)
- [2019-arXiv] AWSRN: Lightweight Image Super-Resolution with Adaptive Weighted Learning Network [`paper`](https://arxiv.org/abs/1904.02358) [`code`](https://github.com/ChaofWang/AWSRN)
- [2019-CVPR] Deep Learning for Image Super-resolution: A Survey [`paper`](https://arxiv.org/abs/1902.06068)
- [2019-CVPR] DPSR: Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels [`paper`](https://arxiv.org/pdf/1903.12529.pdf) [`code`](https://github.com/cszn/DPSR)
- [2019-CVPR] Meta-SR: A Magnification-Arbitrary Network for Super-Resolution [`paper`](https://arxiv.org/pdf/1903.00875.pdf) [`code`](https://github.com/XuecaiHu/Meta-SR-Pytorch)
- [2019-arXiv] PASSRnet: Learning Parallax Attention for Stereo Image Super-Resolution [`paper`](https://arxiv.org/pdf/1903.05784.pdf) [`code`](https://github.com/LongguangWang/PASSRnet)
- [2019-CVPR] SRNTT: Image Super-Resolution by Neural Texture Transfer [`paper`](http://web.eecs.utk.edu/~zzhang61/project_page/SRNTT/cvpr2019_final.pdf)[`code`](https://github.com/ZZUTK/SRNTT)
- [2019-CVPR] Towards Real Scene Super-Resolution with Raw Images [`paper`](https://arxiv.org/abs/1905.12156) 
- [2018-CVPR] RCAN: Image Super-Resolution Using Very Deep Residual Channel Attention Networks [`paper`](https://arxiv.org/abs/1807.02758) [`code`](https://github.com/yulunzhang/RCAN)

#### Synthesis
- [`collection`] Awesome Generative Adversarial Networks with tensorflow**[`code`](https://github.com/kozistr/Awesome-GANs)
- [`framework`] Implementations of a number of generative models GAN, VAE, Seq2Seq, VAEGAN, GAIA, Spectrogram Inversion in Tensorflow** [`code`](https://github.com/timsainb/tensorflow2-generative-models)
- [2019-CVPR] DM-GAN: Dynamic Memory Generative Adversarial Networks for Text-to-Image Synthesis [`paper`](https://arxiv.org/abs/1904.01310)
github.com/NVlabs/SPADE)
- [2019-CVPR oral] GauGAN: Semantic Image Synthesis with Spatially-Adaptive Normalization [`paper`](https://arxiv.org/abs/1903.07291)    [`code`](https://github.com/NVlabs/SPADE)
- [2019-CVPR] MSGAN: Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis [`paper`](https://arxiv.org/abs/1903.05628) [`code`](https://github.com/HelenMao/MSGAN)
- [2019-arXiv] MSG-GAN: Multi-Scale Gradients GAN for more stable and synchronized multi-scale image synthesis [`paper`](https://arxiv.org/abs/1903.06048) [`code`](https://github.com/akanimax/BMSG-GAN)
- [2019-argXiv] Self-Attention Generative Adversarial Networks [`paper`](https://arxiv.org/abs/1805.08318) [`code`](https://github.com/brain-research/self-attention-gan)
- [2019-CVPR] Shapes and Context: In-the-wild Image Synthesis & Manipulation(Image Synthesis) [`code`](http://www.cs.cmu.edu/~aayushb/OpenShapes/OpenShapes.pdf) [`code`](http://www.cs.cmu.edu/~aayushb/OpenShapes/)
- [2019-CVPR] STGAN: A Unified Selective Transfer Network for Arbitrary Image Attribute Editing [`paper`](https://arxiv.org/abs/1904.09709) [`code`](https://github.com/csmliu/STGAN)
- [2018-CVPR] High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs [`paper`](https://arxiv.org/pdf/1711.11585.pdf) [`code`](https://github.com/NVIDIA/pix2pixHD)

#### Image Translation
- [2019-CVPR] Image-to-Image Translation via Group-wise Deep Whitening-and-Coloring Transformation(   ) [`paper`](https://arxiv.org/abs/1812.09912) [`code`](https://github.com/taki0112/GDWCT-Tensorflow)
- [2018-CVPR] CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks [`paper`](https://arxiv.org/pdf/1703.10593.pdf) 
- [2018-CVPR] Pix2pix: Image-to-Image Translation with Conditional Adversarial Networks [`paper`](https://arxiv.org/pdf/1611.07004.pdf) [`code`](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

#### Date augmentaiton
- [2019-CVPR] A Preliminary Study on Data Augmentation of Deep Learning for Image Classification [`paper`](https://arxiv.org/abs/1906.11887)
- [2019-CVPR] Further advantages of data augmentation on convolutional neural networks [`paper`](https://arxiv.org/abs/1906.11052)
- [2019-CVPR] Learning Data Augmentation Strategies for Object Detection [`paper`](https://arxiv.org/abs/1906.11172)
- [2019-CVPR] PSIS: Data Augmentation for Object Detection via Progressive and Selective Instance-Switching [`paper`](https://arxiv.org/abs/1906.00358) [`code`](https://github.com/Hwang64/PSIS)
- [2019-CVPR] Wide-Context Semantic Image Extrapolation(expand image) [`paper`](http://jiaya.me/papers/imgextrapolation_cvpr19.pdf) [`code`](https://github.com/shepnerd/outpainting_srn)

 
### Objection Detection & Semantic
- [`ToolBox`] A Simple and Versatile Framework for Object Detection and Instance Recognition [`code`](https://github.com/TuSimple/simpledet)
- [`ToolBox`] ObjectionDetection by yolov2, tiny yolov3, mobilenet, mobilenetv2, shufflenet(g2), shufflenetv2(1x), squeezenext(1.0-SqNxt-23v5), light xception, xception [`code`](https://github.com/TencentYoutuResearch/ObjectDetection-OneStageDet/tree/master/yolo)
- [`ToolBox`] MMDetection: Open MMLab Detection Toolbox and Benchmark [`paper`](https://arxiv.org/abs/1906.07155) [`code`](https://github.com/open-mmlab/mmdetection)
- [`ToolBox`] Semantic Segmentation on PyTorch (include FCN, PSPNet, Deeplabv3, DANet, DenseASPP, BiSeNet, EncNet, DUNet, ICNet, ENet, OCNet, CCNet, PSANet, CGNet, ESPNet, LEDNet) [`code`](https://github.com/Tramac/awesome-semantic-segmentation-pytorch) [`code`](https://github.com/hszhao/semseg) 
- [`ToolBox`] Segmentation models with pretrained backbones [`code`](https://github.com/qubvel/segmentation_models)

#### Objection Detection 
- [2019-CVPR] Activity Driven Weakly Supervised Object Detection [`code`](https://github.com/zhenheny/Activity-Driven-Weakly-Supervised-Object-)
- [2019-CVPR] CenterNet: Objects as Points [`paper`](https://arxiv.org/abs/1904.07850)(***) [`code`](https://github.com/xingyizhou/CenterNet)
- [2019-CVPR] Cascade R-CNN：High Quality Object Detection and Instance Segmentation(***SOTA) [`paper`](https://arxiv.org/abs/1906.09756) [`code`](https://github.com/zhaoweicai/Detectron-Cascade-RCNN) [`code-Caffe`](https://github.com/zhaoweicai/cascade-rcnn)
- [2019-CVPR] CornerNet-Lite: Efficient Keypoint Based Object Detection(SOTA) [`paper`](https://arxiv.org/abs/1904.08900) [`code`](https://github.com/princeton-vl/CornerNet-Lite)
- [2019-CVPR] DFPN: Efficient Object Detection Model for Real-Time UAV Applications [`paper`](https://arxiv.org/abs/1906.00786) [`code`](https://github.com/zhaoweicai/Detectron-Cascade-RCNN) [`code-Caffe`](https://github.com/zhaoweicai/cascade-rcnn)
- [2019-CVPR] Distilling Object Detectors with Fine-grained Feature Imitation [`code`](https://github.com/twangnh/Distilling-Object-Detectors)
- [2019-CVPR] ExtremeNet: Bottom-up Object Detection by Grouping Extreme and Center Points(***) [`paper`](https://arxiv.org/abs/1901.08043) [`code`](https://github.com/xingyizhou/ExtremeNet)
- [2019-CVPR] FSAF: Feature Selective Anchor-Free Module for Single-Shot Object Detection(SOTA) [`paper`](https://arxiv.org/abs/1903.00621)
- [2019-CVPR] FoveaBox: Beyond Anchor-based Object Detector(SOTA) [`paper`](https://arxiv.org/pdf/1904.03797v1.pdf)
- [2019-CVPR] FCOS: Fully Convolutional One-Stage Object Detection（***） [`paper`](https://arxiv.org/abs/1904.01355) [`paper`](https://github.com/tianzhi0549/FCOS/)
- [2019-CVPR] Grid R-CNN Plus: Faster and Better [`paper`](https://arxiv.org/abs/1906.05688) [`code`](https://github.com/STVIR/Grid-R-CNN)
- [2019-CVPR] Hybrid Task Cascade for Instance Segmentation [`paper`](https://arxiv.org/pdf/1901.07518v2.pdf) [`code`](https://github.com/open-mmlab/mmdetection)
- [2019-CVPR] Locating Objects Without Bounding Boxes [`paper`](https://arxiv.org/pdf/1806.07564.pdf)(***crowd count) [`code`](https://github.com/javiribera/locating-objects-without-bboxes)(https://github.com/xingyizhou/ExtremeNet)
- [2019-CVPR] Learning Data Augmentation Strategies for Object Detection [`paper`](https://arxiv.org/pdf/1906.11172v1.pdf) [`code`](https://github.com/tensorflow/tpu/tree/master/models/official/detection)
- [2019-CVPR] LightTrack: A Generic Framework for Online Top-Down Human Pose Tracking [`paper`](https://arxiv.org/pdf/1905.02822.pdf) [`code`](https://github.com/Guanghan/lighttrack)
- [2019-CVPR] Locating Objects Without Bounding Boxes [`paper`](https://arxiv.org/pdf/1806.07564.pdf)(***crowd count) [`code`](https://github.com/javiribera/locating-objects-without-bboxes)(https://github.com/xingyizhou/ExtremeNet)
- [2019-CVPR] TridentNet: Scale-Aware Trident Networks for Object Detection(***SOTA) [`paper`](https://arxiv.org/pdf/1901.01892.pdf) [`code`](https://github.com/TuSimple/simpledet/tree/master/models/tridentnet)
- [2019-CVPR] NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection [`paper`](https://arxiv.org/abs/1904.07392) [`code`](https://github.com/tensorflow/tpu/tree/master/models/official/detection)
- [2019-CVPR] Region Proposal by Guided Anchoring [`paper`](https://arxiv.org/abs/1901.03278) [`code`](https://github.com/open-mmlab/mmdetection)
- [2019-CVPR] SNIPER: Efficient Multi-Scale Training [`paper`](https://arxiv.org/abs/1805.09300) [`code`](https://github.com/MahyarNajibi/SNIPER/)
- [2019-CVPR] SkyNet: A Champion Model for DAC-SDC on Low Power Object Detection(fast and low power)  [`paper`](https://arxiv.org/abs/1906.10327)
- [2019-CVPR] ScratchDet: Training Single-Shot Object Detectors from Scratch [`paper`](https://arxiv.org/abs/1810.08425) [`code`](https://github.com/KimSoybean/ScratchDet)
- [2019-CVPR] Video Instance Segmentation [`paper`](https://arxiv.org/abs/1905.04804) [`code`](https://github.com/youtubevos/MaskTrackRCNN)
- [2019-CVPR] YOLOv3+: Assisted Excitation of Activations: A Learning Technique to Improve Object Detectors [`paper`](https://arxiv.org/abs/1906.05388) [`code`](https://github.com/ultralytics/yolov3)
- [2018-ECCV] Acquisition of Localization Confidence for Accurate Object Detection [`paper`](https://arxiv.org/abs/1807.11590) [`code`](https://github.com/vacancy/PreciseRoIPooling)

#### Salient Object Detecion 
- [`Survey`] Salient Object Detection: A Survey [`paper`](https://arxiv.org/pdf/1411.5878.pdf)
- [2019-CVPR] A Mutual Learning Method for Salient Object Detection with intertwined Multi-Supervision [`code`](https://github.com/JosephineRabbit/MLMSNet)
- [2019-CVPR] AFNet: Attentive Feedback Network for Boundary-aware Salient Object Detection [`code`](https://github.com/ArcherFMY/AFNet)
- [2019-CVPR] A Simple Pooling-Based Design for Real-Time Salient Object Detection [`code`](https://github.com/backseason/PoolNet)
- [2019-CVPR] BASNet: Boundary-Aware Salient Object Detection [`paper`](http://39.137.36.61:6310/openaccess.thecvf.com/content_CVPR_2019/papers/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.pdf) [`code`](https://github.com/NathanUA/BASNet)
- [2019-CVPR] Contrast Prior and Fluid Pyramid Integration for RGBD Salient Object Detection [`paper`](http://mftp.mmcheng.net/Papers/19cvprRrbdSOD.pdf) [`code`](https://github.com/JXingZhao/ContrastPrior) 
- [2019-CVPR] CapSal: Leveraging Captioning to Boost Semantics for Salient Object Detection [`paper`](https://drive.google.com/open?id=1JcZMHBXEX-7AR1P010OXg_wCCC5HukeZ) [`code`](https://github.com/zhangludl/code-and-dataset-for-CapSal)
- [2019-CVPR] Cascaded Partial Decoder for Fast and Accurate Salient Object Detection(***) [`code`](https://github.com/wuzhe71/CPD) 
- [2019-CVPR] LFNet: Light Field Saliency Detection with Deep Convolutional Networks [`paper`](https://arxiv.org/abs/1906.08331) [`code`](https://github.com/pencilzhang/LFNet-light-field-saliency-net)
- [2019-CVPR] Pyramid Feature Attention Network for Saliency detection(***) [`paper`](https://arxiv.org/abs/1903.00179) [`code`](https://github.com/CaitinZhao/cvpr2019_Pyramid-Feature-Attention-Network-for-Saliency-detection)
- [2019-CVPR] Shifting More Attention to Video Salient Objection Detection [`paper`](http://dpfan.net/DAVSOD/) [`code`](https://github.com/DengPingFan/DAVSOD)

#### Segmentation
- [2019-CVPR oral] CLAN: Category-level Adversaries for Semantics Consistent [`paper`](https://arxiv.org/abs/1809.09478?context=cs) [`code`](https://github.com/RoyalVane/CLAN)
- [2019-CVPR] BRS: Interactive Image Segmentation via Backpropagating Refinement Scheme(***) [`paper`](https://vcg.seas.harvard.edu/publications/interactive-image-segmentation-via-backpropagating-refinement-scheme/paper) [`code`](https://github.com/wdjang/BRS-Interactive_segmentation)
- [2019-CVPR] DFANet：Deep Feature Aggregation for Real-Time Semantic Segmentation(used in camera) [`paper`](https://share.weiyun.com/5NgHbWH) [`code`](https://github.com/j-a-lin/DFANet_PyTorch)
- [2019-CVPR] Domain Adaptation(reducing the domain shif) [`paper`](http://openaccess.thecvf.com/content_CVPR_2019/papers/Luo_Taking_a_Closer_Look_at_Domain_Shift_Category-Level_Adversaries_for_CVPR_2019_paper.pdf) 
- [2019-CVPR] ELKPPNet: An Edge-aware Neural Network with Large Kernel Pyramid Pooling for Learning Discriminative Features in Semantic Segmentation [`paper`](https://arxiv.org/abs/1906.11428) [`code`](https://github.com/XianweiZheng104/ELKPPNet)
- [2019-CVPR oral] GLNet: Collaborative Global-Local Networks for Memory-Efficient Segmentation of Ultra-High Resolution Images[`paper`](https://arxiv.org/abs/1905.06368) [`code`](https://github.com/chenwydj/ultra_high_resolution_segmentation)
- [2019-CVPR] Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth(***SOTA) [`paper`](https://arxiv.org/abs/1906.11109) [`code`](https://github.com/davyneven/SpatialEmbeddings)
- [2019-ECCV] ICNet: Real-Time Semantic Segmentation on High-Resolution Images [`paper`](https://arxiv.org/abs/1704.08545) [`code`](https://github.com/oandrienko/fast-semantic-segmentation
- [2019-CVPR] LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation(***SOTA) [`paper`](https://arxiv.org/abs/1905.02423) [`code`](https://github.com/xiaoyufenfei/LEDNet)
- [2019-arXiv] LightNet++: Boosted Light-weighted Networks for Real-time Semantic Segmentation [`paper`](http://arxiv.org/abs/1605.02766) [`code`](https://github.com/ansleliu/LightNetPlusPlus)
- [2019-CVPR] PTSNet: A Cascaded Network for Video Object Segmentation [`paper`](https://arxiv.org/abs/1907.01203) [`code`](https://github.com/sydney0zq/PTSNet)
- [2019-CVPR] PPGNet: Learning Point-Pair Graph for Line Segment Detection [`paper`](https://www.aiyoggle.me/publication/ppgnet-cvpr19/ppgnet-cvpr19.pdf) [`code`](https://github.com/svip-lab/PPGNet)
- [2019-CVPR] Show, Match and Segment: Joint Learning of Semantic Matching and Object Co-segmentation [`paper`](https://arxiv.org/abs/1906.05857) [`code`](https://github.com/YunChunChen/MaCoSNet-pytorch)
- [2018-ECCV] BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation [`paper`](https://arxiv.org/abs/1808.00897v1) [`code`](https://

### Model Compress and Accelerate
- [`collection`] Collection of recent methods on DNN compression and acceleration [https://github.com/MingSun-Tse/EfficientDNNs](https://github.com/MingSun-Tse/EfficientDNNs)
- [`collection`] A curated list of neural network pruning resources [https://github.com/he-y/Awesome-Pruning](https://github.com/he-y/Awesome-Pruning)
- [`collection`] model compression and acceleration research papers [https://github.com/cedrickchee/awesome-ml-model-compression](https://github.com/cedrickchee/awesome-ml-model-compression)
- [`TollBox`] Neural Network Distiller by Intel AI Lab: a Python package for neural network compression research [`code`](https://github.com/NervanaSystems/distiller)

#### Pruning
- [2019-CVPR] An Improved Trade-off Between Accuracy and Complexity with Progressive Gradient Pruning(Prune) [`paepr`](https://arxiv.org/abs/1906.08746) [`code`](https://github.com/Anon6627/Pruning-PGP)
- [2019-ICML] EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks [`paper`](https://arxiv.org/abs/1905.11946) [`code`](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) [`code`](https://github.com/lukemelas/EfficientNet-PyTorch)
- [2019-CVPR] FPGM: Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration [`paper`](https://arxiv.org/abs/1811.00250) [`code`](https://github.com/he-y/filter-pruning-geometric-median)
- [2019-CVPR] Importance Estimation for Neural Network Pruning [`code`](https://github.com/NVlabs/Taylor_pruning)
#### Accelerating
- [2019-CVPR] SKNet: Selective Kernel Networks [`paper`](https://arxiv.org/abs/1903.06586?context=cs) [`code`](https://github.com/implus/SKNet)
- [2019-CVPR] SENet: Squeeze-and-Excitation Networks[`paper`](https://arxiv.org/abs/1709.01507) [`code`](https://github.com/hujie-frank/SENet)
- [2019-CVPR] ViP: Virtual Pooling for Accelerating CNN-based Image Classification and Object Detection [`paper`](https://arxiv.org/abs/1906.07912)
 


### Motion & Pose

#### Pose Estimation
- [2019-CVPR] AlphaPose: Real-Time and Accurate Multi-Person Pose Estimation&Tracking System [`paper`](https://www.mvig.org/research/alphapose.html) [`code`](https://github.com/MVIG-SJTU/AlphaPose/tree/pytorch)
- [2019-CVPR] CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark [`paper`](https://arxiv.org/abs/1812.00324) [`code`](https://github.com/Jeff-sjtu/CrowdPose) 
- [2019-CVPR] Efficient Online Multi-Person 2D Pose Tracking with Recurrent Spatio-Temporal Affinity Fields(Oral) [`paper`](https://arxiv.org/abs/1811.11975) [`code`](https://www.gineshidalgo.com/)
- [2019-CVPR] EpipolarPose: Self-Supervised Learning of 3D Human Pose using Multi-view Geometry [`paper`](https://arxiv.org/abs/1903.02330) [`code`](https://github.com/mkocabas/EpipolarPose)
- [2019-CVPR] Exploiting Temporal Context for 3D Human Pose Estimation in the Wild [`paper`](http://arxiv.org/abs/1905.04266) [`code`](https://github.com/deepmind/Temporal-3D-Pose-Kinetics)
- [2019-CVPR] Generating Multiple Hypotheses for 3D Human Pose Estimation With Mixture Density Network(SOTA) [`paper`](https://arxiv.org/pdf/1904.05547.pdf) [`code`](https://github.com/chaneyddtt/Generating-Multiple-Hypotheses-for-3D-Human-Pose-Estimation-with-Mixture-Density-Network)
- [2019-CVPR] Fast Human Pose Estimation(pytorch) [`paper`](https://arxiv.org/abs/1811.05419) [`code`](https://github.com/yuanyuanli85/Fast_Human_Pose_Estimation_Pytorch)
- [2019-CVPR] High-Resolution Representation Learning for Human Pose Estimation(SOTA) [`paper`](https://arxiv.org/abs/1904.04514) [`code`](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
- [2019-CVPR] Hand Shape and Pose Estimation from a Single RGB Image [`paper`](https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxnZWxpdWhhb250dXxneDo3ZjE0ZjY3OWUzYjJkYjA2) [`code`](https://github.com/3d-hand-shape/hand-graph-cnn)
- [2019-CVPR] In the Wild Human Pose Estimation Using Explicit 2D Features and Intermediate 3D Representations [`paper`](https://arxiv.org/pdf/1904.03289v1.pdf)
- [2019-CVPR] VideoPose3D: 3D Human Pose Estimation in Video With Temporal Convolutions and Semi-Supervised Training [`code`](https://github.com/facebookresearch/VideoPose3D)
- [2019-CVPR] XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera [`paper`](https://arxiv.org/abs/1907.00837)

#### Pose Transfer

- [2019-CVPR] Dense Intrinsic Appearance Flow for Human Pose Transfer [`paper`](http://mmlab.ie.cuhk.edu.hk/projects/pose-transfer/) [`code`](https://github.com/ly015/intrinsic_flow)

#### Motion Track
- [2019-CVPR] ATOM: Accurate Tracking by Overlap Maximization(***SOTA) [`paper`](https://arxiv.org/pdf/1811.07628.pdf) [`code`](https://github.com/visionml/pytracking)
- [2019-CVPR Oral] Graph Convolutional Tracking(SOTA) [`code`](http://nlpr-web.ia.ac.cn/mmc/homepage/jygao/gct_cvpr2019.html#)
- [2019-arXiv] Instance-Aware Representation Learning and Association for Online Multi-Person Tracking [`paper`](https://arxiv.org/abs/1905.12409)
- [2019-Github] multi-people tracking (centerNet based person detector + deep sort algorithm with pytorch)(SOTA) [`code`](https://github.com/kimyoon-young/centerNet-deep-sort)
- [2019-CVPR] PoseFix: Model-agnostic General Human Pose Refinement Network [`paper`](https://arxiv.org/abs/1812.03595) [`code`](https://github.com/mks0601/PoseFix_RELEASE)
- [2019-CVPR Oral] Progressive Pose Attention Transfer for Person Image Generation [`paper`](https://arxiv.org/abs/1904.03349) [`code`](https://github.com/tengteng95/Pose-Transfer)
- [2019-CVPR] PifPaf: Composite Fields for Human Pose Estimation [`paper`](https://arxiv.org/abs/1903.06593) [`code`](https://github.com/vita-epfl/openpifpaf) [`code`](https://github.com/vita-epfl/openpifpaf)
- [2019-CVPR] SemGCN: Semantic Graph Convolutional Networks for 3D Human Pose Regression [`paper`](https://arxiv.org/abs/1904.03345) [`code`](https://github.com/garyzhao/SemGCN)
- [2019-CVPR] MVPOSE: Fast and Robust Multi-Person 3D Pose Estimation from Multiple Views(multi-person) [`paper`](https://arxiv.org/pdf/1901.04111.pdf) [`code`](https://github.com/zju3dv/mvpose)
- [2019-CVPR] SiamMask: Fast Online Object Tracking and Segmentation: A Unifying Approach(***SOTA) [`paper`](https://arxiv.org/pdf/1812.05050.pdf) [`code`](https://github.com/foolwood/SiamMask)
- [2019-CVPR] SiamRPN++: Evolution of Siamese Visual Tracking With Very Deep Networks(***SOTA) [`paper`](https://arxiv.org/pdf/1812.11703.pdf) [`code`](https://github.com/PengBoXiangShang/SiamRPN_plus_plus_PyTorch)
#### Action Recognition
- [2019-arXiv] VTN:Lightweight Network Architecture for Real-Time Action Recognition[`paper`](https://arxiv.org/abs/1905.08711) [`code`](https://github.com/opencv/openvino_training_extensions/tree/develop/pytorch_toolkit/action_recognition)
#### Keypoint Detection
- [2018-CVPR] OpenPose: Real-time multi-person keypoint detection library for body, face, hands, and foot estimation(***) [`code`](https://github.com/CMU-Perceptual-Computing-Lab/openpose)



### Text Detection & Recognition

#### Detection 
- [2019-CVPR] Arbitrary Shape Scene Text Detection with Adaptive Text Region Representation [`paper`](https://arxiv.org/abs/1905.05980)
- [2019-CVPR] A Multitask Network for Localization and Recognition of Text in Images(end-to-end) [`paper`](https://arxiv.org/abs/1906.09266)
- [2019-CVPR] AFDM: Handwriting Recognition in Low-resource Scripts using Adversarial Learning(data augmentation) [`paper`](https://arxiv.org/abs/1811.01396) [`code`](https://github.com/AyanKumarBhunia/Handwriting_Recogition_using_Adversarial_Learning)
- [2019-CVPR] CRAFT: Character Region Awareness for Text Detection [`paper`](https://arxiv.org/abs/1904.01941) [`code`](https://github.com/clovaai/CRAFT-pytorch
- [2019-CVPR] Data Extraction from Charts via Single Deep Neural Network(*) [`paper`](https://arxiv.org/abs/1906.11906?from=singlemessage&isappinstalled=0)
- [2019-CVPR] E2E-MLT - an Unconstrained End-to-End Method for Multi-Language Scene Text [`paper`](https://arxiv.org/abs/1801.09919)
- [2019-arXiv] FACLSTM: ConvLSTM with Focused Attention for Scene Text Recognition [`paper`](https://arxiv.org/abs/1904.09405)
- [2019-CVPR] Look More Than Once: An Accurate Detector for Text of Arbitrary Shapes [`paper`](https://arxiv.org/abs/1904.06535)
- [2019-CVPR] PSENET: Shape Robust Text Detection with Progressive Scale Expansion Network [`paper`](https://arxiv.org/abs/1903.12473)
- [2019-CVPR] PMTD: Pyramid Mask Text Detector [`paper`](https://arxiv.org/abs/1903.11800) [`code`](https://github.com/STVIR/PMTD)
- [2019-CVPR] Spatial Fusion GAN for Image Synthesis (word Synthesis) [`paper`](https://arxiv.org/abs/1812.05840 [`code`](https://github.com/Sunshine352/SF-GAN)
- [2019-CVPR] Scene Text Detection with Supervised Pyramid Context Network [`paper`](https://arxiv.org/abs/1811.08605)
- [2019-arXiv] TextField: Learning A Deep Direction Field for Irregular Scene Text Detection [`paper`](https://arxiv.org/abs/1812.01393) [`code`](https://github.com/YukangWang/TextField)
- [2019-CVPR] Typography with Decor: Intelligent Text Style Transfer [`paper`](https://github.com/daooshee/Typography2019/blob/master/3159.pdf) [`code`](https://github.com/daooshee/Typography-with-Decor)
- [2019-CVPR] TIOU: Tightness-aware Evaluation Protocol for Scene Text Detection(new Evalution tool)[`paper`](https://arxiv.org/abs/1904.00813) [`code`](https://github.com/Yuliang-Liu/TIoU-metric)
- [2019-arXiv] MORAN: A Multi-Object Rectified Attention Network for Scene Text Recognition [`paper`](https://arxiv.org/abs/1901.03003) [`code`](https://github.com/Canjie-Luo/MORAN_v2)
- [2019-CVPR] Scene Text Magnifier [`paper`](https://arxiv.org/abs/1907.00693)
- [2018-CVPR] Pixel-Anchor: A Fast Oriented Scene Text Detector with Combined Networks [`paper`](https://arxiv.org/abs/1811.07432)
- [2018-ECCV] Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes [`paper`](https://arxiv.org/abs/1807.02242) [`code`](https://github.com/lvpengyuan/masktextspotter.caffe2)
- [2018-AAAI] PixelLink: Detecting Scene Text via Instance Segmentation [`paper`](https://arxiv.org/abs/1801.01315) [`code`](https://github.com/ZJULearning/pixel_link)
- [2018-CVPR] RRPN: Arbitrary-Oriented Scene Text Detection via Rotation Proposals [`paper`](https://arxiv.org/pdf/1703.01086.pdf) [`code`](http://https://github.com/DetectionTeamUCAS/RRPN_Faster-RCNN_Tensorflow) 

#### Recogination
- [2019-CVPR] ESIR: End-to-end Scene Text Recognition via Iterative Image Rectification [`paper`](https://arxiv.org/abs/1812.05824) [`code`](https://github.com/fnzhan/ESIR) [`code`](https://github.com/MichalBusta/E2E-MLT)
- [2019-CVPR] E2E-MLT: an Unconstrained End-to-End Method for Multi-Language Scene Text [`paper`](https://arxiv.org/abs/1801.09919)
- [2018-CVPR] FOTS: Fast Oriented Text Spotting With a Unified Network [`paper`](http://openaccess.thecvf.com/content_cvpr_2018/html/Liu_FOTS_Fast_Oriented_CVPR_2018_paper.html) [`code`](https://github.com/Vipermdl/OCR_detection_IC15)