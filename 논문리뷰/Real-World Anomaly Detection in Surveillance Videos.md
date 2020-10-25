# Real-World Anomaly Detection in Surveillance Videos
### CVPR 2018

논문링크 : https://arxiv.org/abs/1801.04264

코드는 버전 안맞는게 많은듯하다. 최근에 re-implemented 된 코드를 보는게 나을수도

# Abstract
- 정상과 비정상 비디오를 둘다 활용하여 이상현상을 학습하는 방식을 제안
- Deep MIL(Multiple Instance Learning)을 활용한 학습을 제안
- 각 비디오를 하나의 bag으로, video segments를 MIL의 하나의 instance로 둠
- 각 anomalous video segment에 높은 anomaly scores를 예측할 수 있게 Deep anomaly ranking model을 학습
- sparsity와 temporal smoothness constraint를 제시
- Large-scale dataset 소개
  - 이상/비이상의 이진 분류 혹은 13개의 이상행동 탐지의 다중 분류 task로 사용 가능
- Experiment: SOTA 달성

# 1. Introduction
- CCTV 영상을 항상 감시할 수 없으며, 인간의 모니터링에는 한계가 존재한다.
- 따라서, 이를 포착하는 모델을 제안한다. 목표는 적시에 이상 현상을 알리는 것이며 정상 패턴과 다른 videos를 탐지해낸다.
- Abnormal event는 다양하기 때문에 이전에 알고있던 Abnormal Event의 정보에 의지하는 것이 아닌 최소한의 supervised으로 수행되어야 함
- 기존 논문은 Sparse-coding 기반의 representative method를 제안했었으며, SOTA였다. video의 초기부분에만 normal event가 포함되어 있으므로 이를 기반으로 normal event dictoinary를 작성하고 사용했다.

#### Movitation and contributions.
- 기존의 접근 방식은 normal pattern을 학습하여 Abnormal을 탐지할 수 있다고 가정함
- 그러나 이 가정은 항상 맞는게 아니다. 이상과 비이상의 경계는 모호하며 그리고 실제 상황에서 경우에 따라 같은 행동이 이상/비이상이 되는 경우도 비일비재하기 때문이다.
- 본 논문에서는 weakly label된 training video를 활용한 anomaly detection 모델을 제안한다.
- 즉, 각 비디오는 normal과 anomaly를 가지고 있으나 어디 시점에 있는지는 모른다.
- 이렇게하면 비디오별로 labeling하는 것이 가능해져서 많은 양의 video를 쓸 수 있다.

#### Summary
- weakly-supervised learning 을 MIL 방식으로 진행
- ranking model로 학습
- test하면서, long-untrimmed video를 segment로 나누고 모델에 input으로 넣어줌
- MIL Solution 제시
- sparsity와 temporally smoothness constraints를 가진 MIL ranking loss는 DNN에서 각 video segment별 anomaly scores를 계산
- large dataset도 소개함
- Experiment : SOTA 달성
- baseline, C3D, TCNN의 결과를 제공

# 2. Related Work
## Anomaly Detection
킵킵

## Ranking
- Rank 학습 : 머신러닝의 한 연구 분야
- 이는 각각의 점수가 아닌 각 instance별로 서로 연관된 scores를 증가시키는데 집중
- 절대적(X) 상대적(O)인 개념
- 검색 엔진에서 검색(retrieval) quality를 증진시키기 위해 rank-SVM이 나오기도 했고 successive linear programming을 사용해 multiple instance ranking problems을 푸는 알고리즘이 나오기도 했다.
- 근래에 Deep Ranking Networks는 많이 사용되고 있고 SOTA 를 달성하고 있다.
- 모든 deep ranking methods는 positive, negative samples에 대한 라벨링이 아주 많이 필요하다.
- 하지만 본 논문에서는 anomaly detection을 regression 문제로 풀었다. (feature vector를 anomaly score(0-1)로 매핑)
- 학습에서 segment-level label의 어려움을 덜기 위해, MIL 방식을 활용

# 3. Proposed Anomaly Detection Method

![image](https://user-images.githubusercontent.com/61506233/97106076-d4a6b280-1702-11eb-85ac-d8114d654aa1.png)

- 비디오 데이터를 특정 숫자의 segment로 분할하는 접근 방식이며, 이 segment는 bag의 instance가 된다.
- positive(abnormal)과 negative(normal) bags를 사용하여 deep MIL ranking loss를 사용한 anomaly detection model을 학습시킨다.

## 3.1 Multiple Instance Learning
- SVM을 활용한 기존 지도학습에서 사용한 optimization function 대신 새롭게 만들었다.
![image](https://user-images.githubusercontent.com/61506233/97106006-6235d280-1702-11eb-8db3-bf868775e360.png)


## 3.2 Deep MIL Ranking Model

- 최종적으로 제안한 MIL ranking objective function

![image](https://user-images.githubusercontent.com/61506233/97106009-6bbf3a80-1702-11eb-9e45-b2497e4e6c4f.png)

![image](https://user-images.githubusercontent.com/61506233/97106011-724db200-1702-11eb-8a41-565d4c1f5d76.png)

#### Bags Formations
- 각 video를 같은 크기의 겹치지 않는 temporal segment로 나누고 이를 bag instance로 사용
- 주어진 video segment에 대해, 3D convolution features를 추출
- 이를 1. computational efficiency와 2. video 행동 인식에서 capturing appearance와 motion dynamics의 evident capability 때문에 3D feature representation을 사용

# 4. Dataset
## 4.1 Previous datasets
## 4-2. Our datasets
- Abuse, Arrest, Arson, Assault, Accident, Burglary, Explosion, Fighting, Robbery, Shooting, Stealing, Shoplifting, and Vandalism의 13개 이상 행동을 포함하는 long untrimmed surveillance videos

# 5. Experiemtns
## 5.1 Implementation Details
- C3D의 FC6 layer에서 visual features를 추출
- features를 계산하기 앞서, video frame을 240X320 pixels로 바꾸고 frame rate를 30fps(frames per second)로 고정
- 모든 16 frame video clip(l2 normalization)에 대해 C3D features를 계산
- video segment의 features를 얻기 위해 해당 segment에 대해 모든 16-frame clip features의 평균을 취함
- 위에서 계산한 features (4,096D)를 3-layer FC neural network에 넣어줌
- First FC layer는 512 units, Second는 32 units, 마지막 layer는 1 units를 가짐
- 각 FC layer에 60%의 dropout regularizaion이 사용됨
- 1st FC layer엔 ReLU, 3rd FC layer엔 sigmoid activation function을 사용
- 초기 learning rate=1e-3으로 Adagrad optimizer를 사용
- MIL ranking loss의 sparsity와 smoothness constraints의 parameter는  for the best performance
- 각 video를 32개의 non-overlapping segments로 나누고 각 video segment를 bag의 instance로 고려
- The number of segments 32 는 경험적으로 구해낸것
- multi-scale overlapping temporal segments에 대해서도 연구했으나 detection accuracy에 큰 도움이 되지 않음
- mini-batch로 30 positive, 30 negative bags를 random하게 선택
- Theano을 활용하여 automatic differentiation으로 gradient를 계산
- 그 다음 MIL ranking objective function을 계산, 모든 batch에 대해 loss를 역전파


#### Evaluation Metric
이전 연구와 동일하게 ROC(Receiver Operating Characteristic)과 AUC(Area Under the Curve)를 사용하여 methodology를 평가함


## 5.2 Comparison with the State-of-the-art

## 5.3 Analysis of the Proposed Method
#### Model training
- 본 논문의 assumption은 positive, negative videos가 video-level로 labeling 되어 있다는 것
- network는 자동적으로 video의 anomaly 위치를 예측하도록 학습이 됨
- 위 목표를 당성하기 위해 network는 training iteration 동안 anomalous video segments에 대해 높은 score를 토해내도록 학습해야 함
- 1,000 iteration에선 network는 이상/비이상 video segment에 대해 동일하게 높은 점수를 부여
- 3,000 iteration을 넘기고 나선 network가 normal segments에 대해 낮은 점수를 부여하고 anomalous segment에 대해서는 높은 점수를 유지하기 시작
- iteration을 높이고 network가 더 많은 video를 보게 만드니 모델이 자동적으로 localize anomaly를 정확하게 학습하게 됨
- 비록 본 논문에서 어떠한 segment level annotations을 사용하지 않았지만, network는 anomaly의 시간적 위치를 anomaly scores를 통하여 예측할 수 있게되었다.
- False alarm rate
- 실제 세계에 적용했을 때, 감시 카메라의 주요 부분은 normal이다.
- 강건한 anomaly detection method는 normal video에 대해 FAR을 적게 울려야 한다.
- 그러므로, 저자는 본 방법론의 성능을 normal video에 대해서만 평가했다.
- 타 방법론 대비 FAR이 현저하게 낮게 나왔다. ([18]:27.2, [28]:3.1, Proposed:1.9)
- 이는 훈련에 이상 비디오와 일반 비디오를 모두 사용하면 깊은 MIL 순위 모델이보다 일반적인 일반 패턴을 학습하는 데 도움이 된다는 것을 반증함

## 5.4 Anomalous Activity Recognition Experiments


# 6. Conclusion
- Surveillance video에서 실생활의 이상을 탐지하는 딥러닝 접근 방식을 제안
- 실제 이상 행동의 복잡함 때문에 normal data만 사용하는 것은 최적이 아님
- 이상/비이상 데이터 둘 다 사용하는 시도를 수행
- 라벨링의 비효율성을 피하기 위해 Deep MIL frameword with weakly labeled data를 제안
- 위 방식을 검증하기 위해 large-scale anomaly dataset을 구축
- 실험 결과는 baseline 보다 뛰어난 성능을 보임

