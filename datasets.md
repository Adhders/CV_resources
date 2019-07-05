
# Awesome of Datesets for Computer CV

a list of datasets dedicated to the Face Recognition & Detection , OCR , Objection Detection, Gan , SLAM, Motion Track & Pose Estimation , ReID, etc. Any suggestions and pull request are welcome.


## repository
- [`Kaggle`] the datasets are used for kaggle competition [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)
- [`google`] the datasets search engine [https://toolbox.google.com/datasetsearch](https://toolbox.google.com/datasetsearch)
- [`AWS`] the datasets is diversity,contains transportation,satellite picture, with description and tutor material[https://registry.opendata.aws/](https://registry.opendata.aws/)
- [`UCI`] The UCI Machine Learning Repository is a collection of databases, domain theories, and data generators that are used by the machine learning community for the empirical analysis of machine learning algorithms [http://archive.ics.uci.edu/ml/about.html](http://archive.ics.uci.edu/ml/about.html)
- [`AwesomeData`] The vision of the AwesomeData community is contributing a pure list of high quality datasets for open communities such as academia, research, education etc [https://github.com/awesomedata/awesome-public-datasets](https://github.com/awesomedata/awesome-public-datasets)
- [`LSP`] [http://sam.johnson.io/research/lsp.html](http://sam.johnson.io/research/lsp.html) 
- [`FLIC`] [https://bensapp.github.io/flic-dataset.html](https://bensapp.github.io/flic-dataset.html)
- [`MPII`] [https://bensapp.github.io/flic-dataset.html](https://bensapp.github.io/flic-dataset.html)

     
# Computer Vision

## picture

### general
- [`MSCOCO`] [http://cocodataset.org/#download](http://cocodataset.org/#download)
- [`AI Challenge`] [https://challenger.ai/competition/keypoint/subject](https://challenger.ai/competition/keypoint/subject)
- [`Visual Tracker Benchmark`] [http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html)
- [`visualdata`] the datasets is used for classification, objcetion detection and semantic, automatic,OCR,etc[https://www.visualdata.io/](https://www.visualdata.io/)
- [`VOC-360`] VOC-360 is the first dataset for object detection, segmentation, and classification in fisheye images, which contains 39,575 fisheye images[https://researchdata.sfu.ca/islandora/object/sfu%3A2724](https://researchdata.sfu.ca/islandora/object/sfu%3A2724)

#### Traffic
- [`tusimple`] Lane Detection application [`LANE DETECTION`](https://github.com/TuSimple/tusimple-benchmark/issues/3)

### Surgery
 [`lesions`] a large collection of multi-source dermatoscopic images of pigmented lesions [`classification`](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)


#### cloth
- [`DeepFashion2`]: DeepFashion2 is a comprehensive fashion dataset. It contains 491K diverse images of 13 popular clothing categories from both commercial shopping stores and consumers. It totally has 801K clothing clothing items, where each item in an image is labeled with scale, occlusion, zoom-in, viewpoint, category, style, bounding box, dense landmarks and per-pixel mask.There are also 873K Commercial-Consumer clothes pairs [`cloth classification & detection`](https://github.com/switchablenorms/DeepFashion2)

#### people
- [`CrowdPose`]	CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark [`Pose Estimation`](https://drive.google.com/file/d/1VprytECcLtU4tKP32SYi_7oDRbw7yUTL/view?usp=sharing)
- [`face`] https://github.com/becauseofAI/HelloFace 
- [`CASIA-SURF`] A Dataset and Benchmark for Large-scale Multi-modal Face Anti-spoofing [`Anti-spoofing`](https://sites.google.com/qq.com/chalearnfacespoofingattackdete)




#### Face Recognition
- **DiF**: Diversity in Faces [[project]](https://www.research.ibm.com/artificial-intelligence/trusted-ai/diversity-in-faces/) [[blog]](https://www.ibm.com/blogs/research/2019/01/diversity-in-faces/)
- **FRVT**: Face Recognition Vendor Test [[project]](https://www.nist.gov/programs-projects/face-recognition-vendor-test-frvt) [[leaderboard]](https://www.nist.gov/programs-projects/face-recognition-vendor-test-frvt-ongoing)
- **IMDb-Face**: The Devil of Face Recognition is in the Noise(**59k people in 1.7M images**) [[paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Liren_Chen_The_Devil_of_ECCV_2018_paper.pdf "ECCV2018") [[dataset]](https://github.com/fwang91/IMDb-Face)
- **Trillion Pairs**: Challenge 3: Face Feature Test/Trillion Pairs(**MS-Celeb-1M-v1c with 86,876 ids/3,923,399 aligned images  + Asian-Celeb 93,979 ids/2,830,146 aligned images**) [[benckmark]](http://trillionpairs.deepglint.com/overview "DeepGlint") [[dataset]](http://trillionpairs.deepglint.com/data) [[result]](http://trillionpairs.deepglint.com/results)
- **MF2**: Level Playing Field for Million Scale Face Recognition(**672K people in 4.7M images**) [[paper]](https://homes.cs.washington.edu/~kemelmi/ms.pdf "CVPR2017") [[dataset]](http://megaface.cs.washington.edu/dataset/download_training.html) [[result]](http://megaface.cs.washington.edu/results/facescrub_challenge2.html) [[benckmark]](http://megaface.cs.washington.edu/)
- **MegaFace**: The MegaFace Benchmark: 1 Million Faces for Recognition at Scale(**690k people in 1M images**) [[paper]](http://megaface.cs.washington.edu/KemelmacherMegaFaceCVPR16.pdf "CVPR2016") [[dataset]](http://megaface.cs.washington.edu/participate/challenge.html) [[result]](http://megaface.cs.washington.edu/results/facescrub.html) [[benckmark]](http://megaface.cs.washington.edu/)
- **UMDFaces**: An Annotated Face Dataset for Training Deep Networks(**8k people in 367k images with pose, 21 key-points and gender**) [[paper]](https://arxiv.org/pdf/1611.01484.pdf "arXiv2016") [[dataset]](http://www.umdfaces.io/)
- **MS-Celeb-1M**: A Dataset and Benchmark for Large Scale Face Recognition(**100K people in 10M images**) [[paper]](https://arxiv.org/pdf/1607.08221.pdf "ECCV2016") [[dataset]](http://www.msceleb.org/download/sampleset) [[result]](http://www.msceleb.org/leaderboard/iccvworkshop-c1) [[benchmark]](http://www.msceleb.org/) [[project]](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)
- **VGGFace2**: A dataset for recognising faces across pose and age(**9k people in 3.3M images**) [[paper]](https://arxiv.org/pdf/1710.08092.pdf "arXiv2017") [[dataset]](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
- **VGGFace**: Deep Face Recognition(**2.6k people in 2.6M images**) [[paper]](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf "BMVC2015") [[dataset]](http://www.robots.ox.ac.uk/~vgg/data/vgg_face/)
- **CASIA-WebFace**: Learning Face Representation from Scratch(**10k people in 500k images**) [[paper]](https://arxiv.org/pdf/1411.7923.pdf "arXiv2014") [[dataset]](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html)
- **LFW**: Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments(**5.7k people in 13k images**) [[report]](http://vis-www.cs.umass.edu/lfw/lfw.pdf "UMASS2007") [[dataset]](http://vis-www.cs.umass.edu/lfw/#download) [[result]](http://vis-www.cs.umass.edu/lfw/results.html) [[benchmark]](http://vis-www.cs.umass.edu/lfw/)

#### Face Detection
- **WiderFace**: WIDER FACE: A Face Detection Benchmark(**400k people in 32k images with a high degree of variability in scale, pose and occlusion**) [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_WIDER_FACE_A_CVPR_2016_paper.pdf "CVPR2016") [[dataset]](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) [[result]](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html) [[benchmark]](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)
- **FDDB**: A Benchmark for Face Detection in Unconstrained Settings(**5k faces in 2.8k images**) [[report]](https://people.cs.umass.edu/~elm/papers/fddb.pdf "UMASS2010") [[dataset]](http://vis-www.cs.umass.edu/fddb/index.html#download) [[result]](http://vis-www.cs.umass.edu/fddb/results.html) [[benchmark]](http://vis-www.cs.umass.edu/fddb/) 

#### Face Landmark
- **LS3D-W**: A large-scale 3D face alignment dataset constructed by annotating the images from AFLW, 300VW, 300W and FDDB in a consistent manner with 68 points using the automatic method [[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Bulat_How_Far_Are_ICCV_2017_paper.pdf "ICCV2017") [[dataset]](https://adrianbulat.com/face-alignment)
- **AFLW**: Annotated Facial Landmarks in the Wild: A Large-scale, Real-world Database for Facial Landmark Localization(**25k faces with 21 landmarks**) [[paper]](https://files.icg.tugraz.at/seafhttp/files/460c7623-c919-4d35-b24e-6abaeacb6f31/koestinger_befit_11.pdf "BeFIT2011") [[benchmark]](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)

#### Face Attribute
- **CelebA**: Deep Learning Face Attributes in the Wild(**10k people in 202k images with 5 landmarks and 40 binary attributes per image**) [[paper]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Liu_Deep_Learning_Face_ICCV_2015_paper.pdf "ICCV2015") [[dataset]](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
































#### text

- [`ICDAR 2015`](http://rrc.cvc.uab.es/) 1000 training images and 500 testing images
- [`ICDAR 2017`](https://rrc.cvc.uab.es/?ch=8&com=downloads) Competition on Multi-lingual scene text detection and script identification
- [`MLT 2017`](http://rrc.cvc.uab.es/?ch=8&com=introduction) 7200 training, 1800 validation images
- [`COCO-Text (Computer Vision Group, Cornell)`](http://vision.cornell.edu/se3/coco-text/) 63,686 images, 173,589 text instances, 3 fine-grained text attributes.
- [`Synthetic Word Dataset (Oxford, VGG)`](http://www.robots.ox.ac.uk/~vgg/data/text/) 9 million images covering 90k English words
- [`IIIT`](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html) 5000 images from Scene Texts and born-digital (2k training and 3k testing images) Each image is a cropped word image of scene text with case-insensitive labels
- [`StanfordSynth`](http://cs.stanford.edu/people/twangcat/#research) Small single-character images of 62 characters (0-9, a-z, A-Z)
- [`(MSRA-TD500)`](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500))
- [`Street View Text (SVT)`](http://tc11.cvc.uab.es/datasets/SVT_1) 100 images for training and 250 images for testing
- [`KAIST Scene_Text`](http://www.iapr-tc11.org/mediawiki/index.php/KAIST_Scene_Text_Database) 3000 images of indoor and outdoor scenes containing text
- [`Chars74k`](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) Small single-character images of 62 characters (0-9, a-z, A-Z) Over 74K images from natural images, as well as a set of synthetically generated characters 



## video

#### gerneral
- [`LaSOT`] A High-quality Benchmark for Large-scale Single Object Tracking [`Object Tracking`](https://cis.temple.edu/lasot/index.html)
- [`Moments in Time`] Moments in Time: one million videos for event understanding） [`videos understanding`](http://moments.csail.mit.edu/)
- [`UCF101`] action recognition data set of realistic action videos, collected from YouTube, having 101 action categories. This data set is an extension of UCF50 data set which has 50 action categories [`action recognition`](https://www.crcv.ucf.edu/data/UCF101.php)
- [`DAVIS`] DAVIS Challenge on Video Object Segmentation [`Video Object Segmentation`](https://davischallenge.org/davis2017/code.html)
#### sports
- [`Sports1M`] contains 1,133,158 video URLs which have been annotated automatically with 487 Sports labels using the YouTube Topics API [`video classification`](https://cs.stanford.edu/people/karpathy/deepvideo)
- [`Kinetics`] Kinetics consists of approximately 650,000 video clips, and covers 700 human action classes with at least 600 video clips for each action class. Each clip lasts around 10 seconds and is labeled with a single class. [`video understanding`](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)
#### car
- [`CityFlow`] A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking and Re-Identification [`Vehicle ReId`](https://www.aicitychallenge.org/)
- [`Argoverse`] 3D Tracking and Forecasting With Rich Maps [`Object Tracking`](https://www.argoverse.org/)

#### people
- [`CrowdPose`]	CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark [`Pose Estimation`](https://drive.google.com/file/d/1VprytECcLtU4tKP32SYi_7oDRbw7yUTL/view?usp=sharing)
- [`JHMDB`] J-HMDB is, however, more than a dataset of human actions; it could also serve as a benchmark for pose estimation
and human detection[`motion understand`](http://jhmdb.is.tue.mpg.de/dataset
- [`Kinetics`] Kinetics consists of approximately 650,000 video clips, and covers 700 human action classes with at least 600 video clips for each action class. Each clip lasts around 10 seconds and is labeled with a single class. [`video understanding`](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)


## Recommend


