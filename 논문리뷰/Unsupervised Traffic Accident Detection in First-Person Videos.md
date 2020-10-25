# Unsupervised Traffic Accident Detection in First-Person Videos

논문링크 : https://arxiv.org/abs/1903.00618


# 0. Abstract
- 교통법 위반, 교통사고와 같은 Abnormal Event를 잘 탐지해내는 방법을 제안
- 이전까지의 Abnormal Event Detection의 한계점 제시
  - 1) 고정된 카메라로 촬영된 영상을 사용하기 때문에 프레임 내의 대부분의 객체가 고정되어있음. 자동차에 장착된 블랙박스와 같은 움직이는 카메라에는 활용되지 못함
  - 2) 각 이상 상황 별로 카테고리를 나누어 분류하는 방법으로 이상탐지를 했기 때문에, 라벨링 과정의 비효율성. (Supervised 방식이기에 시간/비용적 문제 + 새로운 데이터에 대한 이상탐지 어려움)
- 본 논문은 1인칭 시점의 블랙박스 영상을 활용한, 비지도 학습 기반의 교통 상황 이상탐지 모델을 제안
- 새로운 방식 : 영상 프레임 내의 객체들에 대한 미래 위치를 예측해서 이상 탐지를 진행 -> 시간/비용 최소화 

- 미래 위치 예측 방식
  - 1) 모델이 예측한 미래 위치가 실제 위치와 비교했을 때 얼마나 정확한지(Accuracy)
  - 2) 객체가 보이는 비정상적인 움직임이 일관적으로 관측되는지(Consistency)
  - 를 평가하며 학습이 진행되고, 잘 학습된 모델은 예측한 위치와 실제 위치의 차이가 큰 경우를 이상 상황으로 탐지 

- 사용한 DataSet : HEV-I 데이터셋, AnAn Accident Detection(A3D) 데이터셋을 이용해서 모델을 평가
- Experiment : SOTA (이전에 리뷰한 논문들에 비해 성능 많이 떨어지긴하는데.. 주행 영상에 적용했고 성격이 좀 달라서 그런거같음)

# 1. Instroduction
- 도로 위 상황에 대한 특징
  - Long-tailed Distribution
  - 데이터 불균형(전체 상황에 비해 abnormal Event는 너무 적음)
  - -> 가능한 모든 상황을 모델에 입력해서 분류하도록 하는 것은 한계가 있다.

- 제안
  - normal/abnormal 이진분류
  - 블랙박스 영상 내에 잡히는 특정 객체의 바로 다음 순간 위치를 예측하는 새로운 방식 제안
  - 만약 이상 상황이 발생한다면 모델이 예측한 위치에서 크게 벗어난 위치에 있을 것
  - 블랙박스가 장착된 1인칭의 차 (ego-vehicle) 의 움직임 또한 예측함으로써 스스로의 움직임에 대한 이상치 탐지도 함께 진행
  - 탐지된 이상치가 나 자신의 사고인지, 또는 다른 차량의 사고를 목격한 것인지를 구분할 수 있게 한다.
- large dataset 사용
- SOTA

# 2. Related Work
킵킵

# 3. Unsupervised Traffic Accident Detection In First-Person Videos

- 자율주행 자동차는 어떤 상황이 발생하더라도 반드시 그 상황을 인식하고, 이를 피하는 행동을 할 수 있어야 함
- 이전 연구에서는 전체 프레임에 대한 예측을 활용해 이상치를 탐지했지만, 움직이는 차량에서 촬영한 영상은 나 자신과 객체 모두가 움직이고 있으므로 이를 적용하기 어려움
- 또한, 주행 영상에는 건물이나 도로같이 움직이지 않는 객체도 많이 있는데 이에 대한 미래 위치는 예측할 필요가 없음
- 따라서 본 연구에서는 객체 단위로 위치를 예측하며, 그 예측값에서 크게 벗어날 경우 이상 상황이 일어났다고 간주하는 방식을 제안 
- 기존 연구처럼 모델을 대량의 정상 상태 데이터셋으로 먼저 학습
- 이러한 학습을 거치면 모델은 이 세상에 존재하는 모든 이상 상황에 대해 직접적으로 노출되지 않더라도 '정상 패턴'이라는 것이 무엇인지 학습할 수 있으며, 나아가 객체들이 정상적으로 움직일 위치까지 잘 예측할 수 있게 됨
- 스스로의 움직임(ego-motion)에 의한 영향을 고려하기 위해 입력값에는 ego-motion prediction을 추가
- 과거 프레임을 바탕으로 객체의 현재 위치를 예측했을 때, 모델에 대한 평가는 세 가지 전략을 이용해 진행

## A. Future Object Localization (FOL) 

![image](https://user-images.githubusercontent.com/61506233/97111763-efd5ea00-1723-11eb-8263-d256b5ab4605.png)

### 1) Bounding Box Prediction
- Bounding Box : Xt = [c_x,c_y,w,h]_t (box의 중심 점, 가로길이, 세로길이) -  Detection(Mask R-CNN) 사용
- Future Bounding box : Yt = {Y_t+1,Y_t+2 .... Y_t+sigma} (현재시간 t~t+sigma 까지 예측)
- Ot : 이미지 정보를 담은 벡터 (Image와 Optical flow의 정보를 RoIPool을 이용해 합친 것) - Dense Optical Flow(FlowNet) 사용

- Ot, Xt, Ht-1(과거정보)를 Input으로 하여 Output으로 Yt 예측

- GRU 기반 구조
- Location Encoder
  - 1) 객체의 현재 bounding box에 대한 정보인 Xt
  - 2) 시간 정보와 공간 정보를 합친 Spatiotemporal feature를 입력받아서 Feature Extraction.
  - Spatiotemporal feature : Optical flow를 RoIPool 과정을 통해 추출한 벡터로, 이미지 내의 움직임에 대한 추가적인 정보를 포함
- Location Decoder
  - 다음 순간의 위치를 빠르게 예측

### 2) Ego-Motion Cue
- 본인 차량의 Ego-Motion : Et = {촬영각도, x축방향으로 이동한 거리, y축 방향으로 이동한 거리}
- Odometry Encoder : Et - Et-1 / Decoder : 예측값 Et={Et+1-Et, Et+2-Et, ... Et+sigma-Et}
- 여기서 Et-Et-1을 쓰는 이유는? 축적된 에러를 제거하기 위함
-  즉, 지금까지 축적되어 있지만 겉으로 드러나지 않는(implicit) 에러를 무시하면서, 현재의 순간적인 motion 변화를 담기 위해 차이값을 사용 
- 이걸 다시 Location Decoder에 입력해서 Future Localization 예측의 정확성을 높임
 
### 3) Missed Objects
- 영상 프레임을 지나면서 순간순간 탐지되는 모든 객체들은 tracker 리스트에 담아둠.
- 왜냐면 어떤 순간 특정 객체가 다른 차량 등에 의해 가려져서 보이지 않게 되었을 때에도 그 객체의 현재 위치를 계속 추정하기위함
  - 엥 근데 이게 잘 작동될까...? 이전에 예측했던 값을 이용해 또다시 예측하는데.. 이전에 예측한게 엄청 정확하지않은데 이걸로 또 예측하면 오차가 넘 심하지 않나... 근데 이거아니면 다른방법이 없으셨겠지... 흠 나는 감자니까 일단 넘어가보쥬,,?^_^ 
- age를 둬서 없어진지 너무 오래되었다면, 삭제함

![image](https://user-images.githubusercontent.com/61506233/97111772-f9f7e880-1723-11eb-8893-7c39ae0ad965.png)

## B. Traffic Accident Detection

![image](https://user-images.githubusercontent.com/61506233/97111775-00866000-1724-11eb-9255-543c3755e390.png)

### 1) Predicted Bounding Boxes - Accuracy
- 객체의 다음 순간 위치로 예측한 bounding box와 실제 위치 box 가 얼마나 유사한지를 측정
- bounding box와 실제 box 간의 유사도는 IoU로 측정

![image](https://user-images.githubusercontent.com/61506233/97111788-0aa85e80-1724-11eb-8d00-90b288252287.png)

### 2) Predicted Box Mask - Accuracy
- 심각한 사고가 났을 때에는 객체들의 형상이 크게 꺾이거나 왜곡되기 때문에 부정확한 탐지가 일어남
- 강한 충격 등에 의해 ego-motion이 크게 변화할 때에도 영상 화면이 급격하게 변화하므로 False alarm이 일어날 가능성이 커짐
- 따라서, 이러한 한계점을 극복하기 위해 영상 화면을 Bounding box와 Background 두 가지로 나누어 0과 1로 masking 하고, masking 된 이미지로 IoU를 계산한 값을 또다른 손실함수에 이용

![image](https://user-images.githubusercontent.com/61506233/97111794-12680300-1724-11eb-9a7b-08718b1cbb05.png)

### 3) Predicted Bounding Boxes - Consistency
- 여러 객체가 서로를 가리는 등에 대한 상황이 발생하는 것을 고려
- Anomaly가 발생한다면, 이는 단 한 순간에 머무르지 않고 일정 기간에 걸쳐 쭉 현저하게 Anomalous 한 양상을 보인다는 가정을 더 사용함
- 일관성의 척도
- sigma 개의 미래를 예측하는데, Yt_hat, Yt-j_hat을 비교해서 {cx,cy,w,h}중에 가장 큰 차이를 보이는 값을 선택하고 모든 객체에 대한 평균값을 구함
- 어떤 값이 가장 큰 차를 보일지 모르기 떄문에 네 값에 대한 표준편차를 모두 구해서 이중에서 가장 큰 값을 선택함

![image](https://user-images.githubusercontent.com/61506233/97111803-1ac03e00-1724-11eb-9e0b-0e41aba4113d.png)

# 4. Experiments

![image](https://user-images.githubusercontent.com/61506233/97111816-2ca1e100-1724-11eb-9056-9c7c0bdb0bb1.png)


# 5. Conclusion
- 본 논문은 1인칭 동영상의 교통사고 감지를 위한 Unsupervised 학습 프레임워크를 제안
- Key Challenge는 본인 차량의 빠른 주행으로,  regular training data에서 현재 또는 미래의 RGB 프래임을 시각적으로 reconstruction 하는 것을 어렵게한다.
- traffic participant trajectories를 예측했을 뿐만아니라, 미래의 위치까지 예측했으며 Abnormal Event가 발생했을 수 있다는 신호로 정확도와 일관성을 활용했다.
- 도로에서의 다양한 실제 사고로 구성된 새로운 Data set을 적용했고 기존의 Dataset에 대해서도 Evaluate 해봤다.
- 실험결과 우리 모델은 짱짱맨이다.

