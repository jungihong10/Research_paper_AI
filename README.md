# Research_paper_AI

연구 논문을 읽고 분석

## Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks 
7/19
### Abstract:
State-of-the-art object detection 문제: 물체가 어디에 있는지, 그리고 그 물체가 어떤 class 인지 구별하는 문제

R-CNN 이 나온뒤로 SPPnet, Fast R-CNN이 속도를 많이 끌어 올림 하지만 Region Proposal 에서 많은 시간이 걸린다는게 단점.

Region Proposal Network - RPM 은 어떠한 위치에 존재를 하는가, 안하는가를 식별함, 또 점수도 부여함.

Region Proposal Network - CPU 가 아닌 GPU 로 돌아갈수 있기 때문에 기존에 제안된 fast R-CNN 보다 빠르다. -> Faster R-CNN

별도에 Region Proposal Network 를 추가하여 더 빠르게 돌아감. 이는 그냥 object의 위치를 알려주고, 나머지는 기존에 Fast R-CNN 모델을 따라서 classification 을 진행함.

End-to-end 방법으로 학습이 가능함 -gpu 상이기 때문에 

VGG 모델 사용시 초당 5장의 이미지를 처리 가능함.  - 물론 자율주행은 더 많이 필요하겠지만 하지만 2015년도에서는 엄청 빠른속도

state-of-the-art object detection accuracy 도 올라감을 볼 수 있다. 

속도 + 정확도도 증가했다.


### Introduction:

기존 CNN
Fast CNN - CPU 에서 region proposal 을 함으로서 연산이 느리다. - bottleneck.
-Selective Search - CPU 에서 진행됨 -  fps 1도 안나옴
-EdgeBoxes - 더 빨라졌지만 region proposal 단계에서 시간을 뺏김 - time consuming

그러므로 region proposal 만 수행하는 network 를 별도로 만듬. 하지만 그러면 computation 을 어떻게 sharing 할 수 있을 것이냐가 문제.

본 논문에서는 획기적인 방법으로 region proposal 을 gpu 로 올려서 cost-free, 즉 region proposal을 구하는데 시간이 하나도 안걸림.
뿐만 아니라 detection network 와 feature map 자체를 sharing 하기 때문에 성능도 좋아짐. 시간적인 이점도 가짐. 

RPN은 두가지 기능이 있음: simultaneously regress region bounds + objectness score
물체가 있을 법만한 위치를 적절하게 조정하고 점수를 매기는 기능. 

RPN은 다양한 scale 와 ratio 를 통해서 예측을 할수있다. 1x1, 2x1, 1x2, 다양한 anchor box.

기존 논문들에서 pyramids of filters 사용을 했지만 본 논문은 anchor box 를 사용했다. 다양한 크기와 다양한 비율을 가진 anchor box 로 시간 단축.

사진을 보면 pyramids of images 는 다양한 scale 인 image 를 각각 따로 만들어서 예측을 진행한다. 하지만 anchor box 은 각각의 위치마다 다양한 anchor box 를 사용하여서 pyramid 보다 빠르다. 

RPN 그리고 object detection network fine tuning 시에 번갈아가면서 fine-tuning 을 진행해 학습을 진행했다. - convolutional features 가 공유됨으로서 빠르게 진행되는것을 볼수 있었다. 
- 이 방법은 selective search 를 사용하는 fast r-cnn 보다 빨랐다. 
- GPU 를 사용하여 5fps
- 속도보다 성능도 좋아진것을 볼수 있었다. (다양한 대회)



### Deep Networks for Object  Detection 
물체가 있을 법만한 위치를 찾는건 오직 CPU상에서 selective search 를 통해 구해왔다. 
Object 의 bounding box 를 구하는 연구는 게속되어 왔다. 
예를 들어 overheat method, multibox 가 있다. 

Multibox 의 단점: does not share features between proposal and detection networks

Fast R-CNN, selective search 를 CPU 에서 돌린다는 단점이 있지만, 아무튼 이미지를 단 한번만 cnn 모델을 forwarding 하면 되기 때문에 feature 을 sharing 한다는 장점.

즉 faster R-CNN 은 RPN 을 gpu로 돌린다는 장점.

이미지를 보면 feature 2: 
하나의 이미지가 주어졌을때,  compression layer 을 거쳐 feature map 을 뽑아낸다. 이 feature map 은 공유가 된다. 즉 RPN, classification network 둘다 들어간다. 
RPN - 어떤 위치에 물체가 있는지 예측을 하고, 만약 있다면 어떤 바운딩 박스에 있는지 예측을 함.
RPN에서 나온 결과를 classification network 에 넣어서 각 위치에 있는 물체가 무슨 class 인지 예측을 해줌. 
즉 fast R-cnn 을 따라감. 
RPN 은 일종의 ‘attention’ module- 즉 이런 부분을 조금 더 중심적으로 확인해봐- 하는 역할. 

### Faster R-CNN
1. RPN
2. Fast R-CNN detector
위 두가지 모듈을 합한게 Faster R-CNN

### 3.1 Region Proposal Network

입력을 이미지로 받아서 사각형 바운딩 박스 형태로 아웃풋. 
그 박스에 물체가 존재 하는지 존재하지 않는지 여부 - objectness score

Fast R-CNN 와 feature map 을 공유할 수 있도록 만들어짐.

### Sliding windows: 
그림 figure 3: 한장의 이미지를 conv layer 에 넣어 feature map 을 뽑는다. 
이 feature map 에 다양한 크기와 비율을 가진 anchor box 를 쭉 sliding 해 각 위치에 prediction 을 진행한다. 
왼쪽 위부터 쭉 슬라이딩. Forwarding 을 통해 mapping 을 해 reg layer, cls layer 을 거쳐서 물체가 존재하는지 존재하지 않는지, 만약에 존재하면 정확히 어떤 위치, 그리고 x, y 좌표, height, width 을 구한다. 
총 k 개의 anchor box 를 사용한다. 

### Translation-Invariant 

이미지가 이동을 가해진다고 해도 똑같이 sliding windows 를 하기 때문에 translation invariant 함. 
Figure 1 의 3번이 본 논문에서 사용하는 pyramids of anchor 방식임. 

Loss Function 
기본적으로 RPN 에서 훈련시 binary classification 을 통해 물체가 있는지 없는지를 판별함. 
이때 있다라고 판별하기 위해 두가지 방법이 있음.
1. Highest IOU
2. IOU over 0.7
두개의 바운딩 박스가 있으면 교집합/ 합집합 얼마나 겹쳐있는지

-답이 여러개 있을수 있다. 
-IOU 0.7 이상인게 없을 수도 있다. -> highest IOU 채택

IOU 0.3 이하면 없다고 본다

둘다 아니면 training 에서 배제 시킨다.

### Loss function:
RPN cls, reg 두가지 기능. 
ground truth = 1 
P 는 1 혹은 0

물체가 있을때만 밑에 식도 사용됨. 왜냐면 물체가 없으면 필요 없기 때문. 
Tstar 은 네가지 정보를 가지는 튜플 형태.

1/Ncls, Nreg는 weight parameter. Normalization 을 진행함. 
Lambda 는 가중치 parameter 로 큰 영향을 안끼침.


##You Only Look Once: Unified, Real-Time Object Detection 
7/25

Abstract 

###1. Introduction 

Reframe object detection as a single regression problem

A single convolutional network predicts
-multiple bounding boxes
-class probabilities for those boxes

Real-time speed

###Method

Each grid predicts 1) B 개의 bounding boxes 2) confidence score

Each bounding box consists 5 predictions:
x, y, w, h ,confidence

x, y: center of the box
W, h  the width and height

Confidence score
Pr(object)
= 박스 안에 있을 확률
IOU
=  

Class conditional class probability C = Pr(class|object)
Class-specific 

###Sum-Squared Error
1. Weights localization error equally  with classification error
2. Overpowering the gradient from cells that do contain objects



###Limitations of YOLO

1. Spatial constraint limits the number of nearby objects that our model can predict
    1. 무리지어 나타내는 small objects 못잡음 (ex: flock of birds)
2. Hard to generalize to objects in new of unusual aspect ratios or configurations
3. Treats errors the same in small bounding boxes vs large bounding boxes
    1. Sum-Squared Error

###Comparison to other detection systems

DPM

Sliding window approach
-일정한 크기를 갖는 window 를 이미지 왼쪽 위부터 모든 영억을 탐색함
-real time 불가능 정확도도 yolo 보다 낮음

R-CNN

Region proposal
Selective search algorithm
-후보 영역을 구하는데 사용되는 방법
	1. 초기 영역 생성, 2. 작은 영역의 통합 (비슷한 영역끼리 greedy algorithm) 3. 후보 영역 생성 (2단계 통합된 이미지들을 기반으로 바운딩 박스 추출)
	2000개 받아옴

Differences - YOLO -> spatial constaraaints on the grid cell proposals far fewer bounding boxes (98 vs 2000)

Fast R-CNN, Faster R-CNN
-여전히 real-time 어려움
-focus on speeding up the R-CNN network

###Experiments:
High mAP (mean average precision) in real-time detectors (여러 class)

Vs Fast R-CNN
- Higher Localization error
- Lower background error
Combining Fast R-CNN and YOLO
- YOLO 와 Fast R-CNN이 유사한 bounding box를 예측하는지 확인
- Gives the prediction aa boost
- 

Generalizability: Person detection in Artwork
 - good performance


##U-Net: Convolutional Networks for Biomedical Image Segmentation

###Basic summary:image segmentation for small datasets

###특징:
No padding -> smaller output
Cropping -> loss of border pixels

Mirroring Extrapolate
- 현미경에 세포 이미지를 다룰때 사용됨. 
- 원본 이미지를 좌우 대칭으로 확장시킨다.
    - Output 이 input보다 작기 때문에 크기를 맞추기 위해

Overlap - tile strategy

Weighted Loss
D1: distance to the border of the nearest cell
D2: to the second nearest cell

No batch, one big image
High momentum (0.99)

Data Augmentation: ( rotation, shift, deformation, …)

Random elastic deformation:


