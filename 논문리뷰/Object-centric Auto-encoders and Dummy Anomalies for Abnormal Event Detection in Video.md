# Object-centric Auto-encoders and Dummy Anomalies for Abnormal Event Detection in Video
### CVPR 2019

논문링크 : https://arxiv.org/abs/1812.04960


# Abstract
- 대부분의 기존 접근 방식은 비정상적인 Train Data가 부족하기때문에 비정상적인 이벤트 탐지를 outlier detection 문제로 공식화하여 접근한다.
- 하지만, 비정상적인 문제가 너무 다양하고 사전 정보가 거의 없어 규칙을 만들 수 없기 때문에 이러한 전통적인 방법을 사용하기 어렵다.
- 따라서, 본 연구에서는 비정상적인 사건탐지를 one-versus-rest binary classification 문제로 접근한다.

- Contribution
  1. 모션 중심의 모양 정보를 인코딩하기 위해 객체 중심 컨볼루션 자동 인코더를 기반으로 하는 unsupervised 학습 프레임 워크를 제안
  2. 학습 데이터를 클러스터링 하기 위한 supervised classification 제안

- OVR(one-versus-rest) abnormal event classifier를 사용하여, 각 normal cluster를 다른 cluster와 분리한다. 
- 분류기를 학습시키기 위해, 다른 cluster들은 dummy anomalies로 둔다.
- Inference 동안에 test data에 대응하는 최고 classification 점수는 각 data의 정상 점수를 나타낸다. 점수가 음수인 경우 비정상으로 판단
- Avenue, ShanghaiTech, UCSD, UMN 데이터에 대해 실험했으며, 해당연도 기준 SOTA 달성

# 1. Introduction
- Anomaly Detection : 예상과는 다른 패턴을 보이는 개체나 데이터를 찾는 것이다. (군집형/시계열 접근 가능)
- 정상 이벤트와 비정상 이벤트의 구분이 문제에 따라 다양하기 때문에, 문제에 의존하는 경우가 많다.
  - EX) 정상: 거리에서 트럭을 운전하는 시나리오 / 비정상: 보행자 도보에서 트럭을 운전하는 시나리오
- Abnormal event보다 Normal Event가 상대적으로 많이 발생함(데이터 불균형)
- 문제가 다양해서 규칙을 만들 수 없기 때문에 전통적인 지도학습 방법을 사용하기 어렵다.
- 기존 이상탐지 방법은 정상 이벤트만 포함된 학습 비디오에서 정규 모델을 학습한다. 정규 모델에서 벗어나는 이벤트의 경우를 비정상으로 판단했었다.

- 여기서 제안하는 내용
  - 다중 클래스 분류 문제로 비정상 이벤트 감지
  - 학습 데이터에서는 정상 이벤트만 포함, 다양한 유형을 나타내는 군집을 찾기위해 K-Means Clustering 사용
  - 비정상 이벤트 감지를 OVR 바이너리 분류문제로 지정
  - OVR(One-versus-rest)란? K개의 클래스가 존재하는 경우, 각각의 클래스에 대해 표본이 속하는지(y=1) 속하지 않는지(y=0)의 이진 문제를 푸는 것

- Inference 동안에 test data에 대응하는 최고 classification 점수는 각 data의 정상 점수를 나타낸다. 점수가 음수인 경우 비정상으로 판단한다.
- 탐지된 객체 위에 CAE(Convolutional Auto-Encoder)를 사용하여 각 프레임에서의 unsupervised 학습
- 이를 통해 각 장면에 존재하는 객체에만 초점을 맞출 수 있으며, 각 프레임에서 이상을 정확하게 localize 할 수 있다. 
- 추론 단계에서 가장 높은 점수가 음수이면, 어떤 클래스에도 귀속되지 않기 때문에 비정상으로 판단함

# 2. Realted Work
킵킵

# 3. Method
#### Motivation
- 기존 이상 탐지를 위한 데이터셋이 다양하지 않다.
- 이상 탐지에서 Abnormal event을 정의하는 것이 어렵다. 
- 학습 데이터에는 정상 이벤트만 포함한다.
- 우수한 성능을 추출하기 위해, 객체의 다양한 정보를 이용하는 것이 좋다.

#### Object detection
- Single-shot object detection(FPN=Feature Pyramid Network)를 씀
 - 왜냐면,,
 - 1. 정확도와 속도 사이에 최적 균형
 - 2. 작은 객체를 정확하게 감지할 수 있으며, GPU에서 초당 13 프레임 처리 가능
- 각 프레임당 object detection, object 마다 bounding box로 crop
- 결과이미지는 gray scale로 변환함
- Appearance and Motion features
  - 자른 객체 이미지를 기준으로 t-3, t, t+3 프레임으로 gradients를 계산

 #### Feature learning
- 각 물체에 대한 특징 벡터를 얻기 위해, 3개의 Convolution Auto-Encoder(CAE)를 학습
- frame 1, Auto-Encoder는 객체를 포함하는 입력 이미지를 객체 기준으로 자르고, appearance features를 학습
- frame t-3 and frame t+3, 객체가 어떻게 이동하였는지를 확인하는 gradients 입력으로 사용, motion 학습
- frame t-3 and frame t+3 의 특징 dummy를 이용하여 학습 데이터 부족을 해결

#### Model Training
- multiclass classification 문제로 abnormal event를 공식화 하는 새로운 방식을 제안한다.
- 이 방식은 normal train sample의 하위 집합이 normal train sample의 다른 하위 집합과 관련하여 dummy animalies sample의 역할을 할 수 있는 상황을 구성함으로써 abnormal sample의 부족함을 해결하는 것을 목표로 한다.
- 이는 k-평균을 사용하여 정상 훈련 샘플을 k-means로 군집화함으로써 해결한다.
- k binary SVM models.

#### Inference
- 두 bbox가 overlap 되면, abnormality score를 최대치로
- 마지막으로 가우시안 필터 써서 frame-level anomaly score를 스무딩

![image](https://user-images.githubusercontent.com/61506233/97102685-ee3c0000-16ea-11eb-985e-5fb4cc8081d1.png)

# 4. Experiment
![image](https://user-images.githubusercontent.com/61506233/97102691-ff850c80-16ea-11eb-881f-347d54c57275.png)
![image](https://user-images.githubusercontent.com/61506233/97102693-06ac1a80-16eb-11eb-916b-f263b7e8be3c.png)
![image](https://user-images.githubusercontent.com/61506233/97102695-0d3a9200-16eb-11eb-8a1f-54d971b3fab2.png)
![image](https://user-images.githubusercontent.com/61506233/97102718-2b07f700-16eb-11eb-9bdb-7639aae57c5c.png)



# 5. Conclusion
- 이상 탐지를 위한 새로운 모델 제안하셨다.
1) 객체 중심의 컨볼루션 자동 인코더 학습
2) 이상 탐지를 다중 클래스 문제로 변환하여 해결
- 4가지 데이터셋(Avenue, ShanghaiTech, UCSD, UMN)에서 우수한 결과를 도출했다. 

