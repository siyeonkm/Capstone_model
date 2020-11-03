# Abnormal Event Detection 평가지표

- 비디오 영상에서 비정상적인 사건을 잘 평가했는지 모델을 평가하기 위해서 적당한 평가지표가 필요하다.
- 일반적인 딥러닝 모델에서 사용하는 것은 Accuracy(정확도) 이지만, Abnormal Event Detection에서 이를 사용하는 것은 한계가 있어 적절하지 않다.

### Accuracy의 한계점

![image](https://user-images.githubusercontent.com/61506233/97954976-5b851a80-1de8-11eb-95da-833fe4da771e.png)


- 전체 비디오 100이 있고 이 중 정상적인 사건 99개, 비정상적인 사건 1개가 있다고 하자. 만약 내가 만든 모델이 정상적인 사건을 100개로 비정상적인 사건을 0개로 예측해도 정확도는 99%가 된다.
99%라는 수치만 보면 잘 예측하는 모델처럼 보일 수 있지만, 예측값만 놓고보면 전혀 아닌 것이다. 이를 좋은 모델이라고 할 수 있을까? 아니다.
- 이런 상황을 정확도 역설(Accuracy Paradox)라고 하는데, 주로 데이터가 불균형한 상황에서 발생한다. 따라서 데이터가 균형구조일때에만 사용하는 경우가 일반적이다.

즉, 우리 데이터도 정상적인 사건이 대부분이고 비정사적인 사건이 매우 드문 불균형 데이터이기때문에 평가지표로 Accuracy를 사용하기 어렵다.

따라서 우리는 보다 정확한 진단을 하기위해 AUC라는 평가지표를 사용할 것이다. 

### AUC

- AUC를 이해하기 위해서는 우선 Confusion Matrix에 대한 이해가 필요하다.

![image](https://user-images.githubusercontent.com/61506233/97954821-eaddfe00-1de7-11eb-979a-e352dd383d9b.png)


- TP (True Positive) : 맞춘 경우. 실제 값이 1이고, 예측 값도 1인 경우.
- FN (False Negative) : 틀린 경우. 실제 값이 1인데, 예측 값은 0인 경우.
- FP (False Positive : 틀린 경우. 실제 값이 0인데, 예측 값은 1인 경우.
- TN (True Negatives) : 맞춘 경우. 실제 값이 0이고, 예측 값도 0인 경우.

- Accuracy
   - 전체 케이스 중, 맞춘 케이스의 비율
   - (TP + TN) / (TP + FN + FP + TN)

- TPR
   - 실제로 1인 것 중 내가 예측한 값이 1인 비율이며, Recall 하고 동일한 값이다. 즉 내가 잘 예측한 경우다. 이 값은 높을수록 좋다.
   
![image](https://user-images.githubusercontent.com/61506233/97956017-f5e65d80-1dea-11eb-91dc-68809e79c716.png)

- FPR
  - 실제로는 0인 것중, 내가 예측한 것이 1인 비율이다. 즉, 내가 잘못 예측한 경우다. 이 값은 낮을수록 좋다.

![image](https://user-images.githubusercontent.com/61506233/97956037-fd0d6b80-1dea-11eb-9a2a-d674c59bc69e.png)

![image](https://user-images.githubusercontent.com/61506233/97956344-bff5a900-1deb-11eb-8ded-e74326e90d70.png)


- 이렇게 점들을 연결한 곡선을 ROC 커브 (Receiver Operating Characteristic) 라고 한다.

![image](https://user-images.githubusercontent.com/61506233/97955832-807a8d00-1dea-11eb-854e-e5f604c5b0a3.png)

- 그리고 그 곡선 아래의 넓이를 AUC (Area Under Curve) 라고 하며, 이게 이 모델의 성능지표가 된다.

- AUC 값의 범위는 0~1이다. 예측이 100% 잘못된 모델의 AUC는 0.0이고 예측이 100% 정확한 모델의 AUC는 1.0이다.

- AUC는 다음 두가지 이유로 이상적이다.
   - AUC는 척도 불변이다. AUC는 절대값이 아니라 예측이 얼마나 잘 평가되는지 측정한다.
   - AUC는 분류 임계값 불변이다. 즉, AUC는 어떤 분류 임계값이 선택되었는지와 상관없이 모델의 예측 품질을 측정한다.
   
   
   
   
 ### 현재 Abnormal Event Detection에서의 AUC
 - 카메라가 고정된 상황에서 폭력, 폭발사고, 훼손, 도둑질 등을 탐지 (주차중 모델) : 0.9x AUC
 - 주행중인 영상에서 교통사고를 탐지 (주행중 모델) : 0.6 AUC
 
