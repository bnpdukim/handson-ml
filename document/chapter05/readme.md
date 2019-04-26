##CHAPTER 5 서포트 벡터 머신
* 선형/비선형 분류, 회귀, 이상치 탐색에 사용할 수 있는 다목적 머신러닝 모델
* 복잡한 분류 문제에 잘 맞음
* 작거나 중간 크기의 데이터셋에 적합

###5.1 선형 SVM 분류
* 그림 5-1 라지 마진 분류
  - 왼쪽 그래프의 결정 경계는 설명을 위해 임의의 직선을 그은 것
    - 점선은 제대로 분류 못함
    - 나머지 두 선은 결정 경계가 샘플에 너무 가까움
  - 오른쪽 그래프는 SVM 분류기의 결정 경계
    - SVM 분류기를 클래스 사이에 가장 폭이 넓은 도로를 찾음
    - 라지 마진 분류(large margin classification)라고 함
* SVM은 특성의 스케일에 민감함
  - 사이킷런의 StandardScaler를 사용하것을 권장
  - 그림 5-2 특성 스케일에 따른 민감성

####5.1.1 소프트 마진 분류
* 하드 마진 분류(hard margin classification)
 - 모든 샘플이 도로 바깥쪽에 분류
  - 그림 5-3 이상치에 민감한 하드 마진
  - 데이터가 선형적으로 구분될수 있어야 함
  - 이상치에 민감함
* 소프트 마진 분류(soft margin classification)
  - 도로의 폭을 가능한 한 넓게 유지하는 것과 마진 오류(margin violation)사이에 균형을 이뤄야 함
    - 마진 오류 : 샘플이  도로 중간이나 반대쪽에 있는 경우
  - 사이킷런 SVM 모델에서는 C하이퍼파라미터를 사용해 균형 조절
  - C값을 줄이면 도로의 폭이 넓어지지만 마진 오류도 커짐
  - 그림 5-4 좁은 마진과 넓은 마진
* e.g. Iris-Virginia 품종 감지
  ``` 
  iris = datasets.load_iris()
  X = iris["data"][:, (2,3)]
  y = (iris["target"] == 2).astype(np.float64)
  svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge"))
  ])
  svm_clf.fit(X,y)
  ```
  
###5.2 비선형 SVM 분류
* 4장에서처럼 다항 특성과 같은 특성을 추가
  ``` 
  X, y = make_moons(n_smaples=100, noise=0.15, random_state=42)
  polinomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge",max_iter=2000))
  ])
  polinomial_svm_clf.fit(X, y)
  ```
####5.2.1 다항식 커널
* 낮은 차수의 다항식은 매우 복잡한 데이터셋을 표현 못함
* 높은 차수의 다항식은 굉장히 많은 특성을 추가하므로 모델을 느리게 함
* SVM을 사용할 땐 커널 트릭(kernel trick)이라는 수학적 기교를 적용
  - 실제로는 특성을 추가하지 않으면서 다항식 특성을 많이 추가한 것과 같은 결과를 얻음
  ``` 
  poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
  ])
  polinomial_svm_clf.fit(X, y)
  ```
  - 매개변수 coef0는 모델이 높은 차수와 낮은 차수에 얼마나 영향을 받을지 조절
  - coef0 매개변수는 식5-10의 다항식 커널에 있는 상수항 r임

####5.2.2 유사도 특성 추가
* 각 샘플이 특정 랜드마크(landmark)와 얼마나 닮았는지 측정하는 유사도 함수(similarity function)로 계산한 특성을 추가
* e.g. 2개의 랜드마크 x1=-2, x1=1을 추가
  - r=0.3인 가우시안 방사 기저 함수(Radial Basis Function:RBF)를 유사도 함수로 정의
  - 식 5-1 가우시안 RBF
  - x1=-1 샘플의 경우 첫번째 랜드마크와 1만큼 떨어져있음, 두번째 랜드마크와 2만큼 떨어져 있음
  - x2=exp(-0.3*1^2) = 0.74, x3=exp(-0.3*2^2)=0.30
  - 그림 5-8 가우시안 RBF를 사용한 유사도 특성
* 랜드마크를 어떤게 선택하는가?
  - 데이터셋에 있는 모든 샘플 위치에 랜드마크를 설정하는 것
  - n개의 특성을 가진 m개의 샘플이 m개의 특성을 가진 m개의 샘플로 변환
  - 훈련 세트가 매우 클 경우 동일한 크기의 아주 많은 특성이 만들어짐

####5.2.3 가우시안 RBF 커널
* 커널 트릭으로 실제 특성을 추가하지 않고 유사도 특성을 많이 추가한 것과 같은 결과를 얻음
  ``` 
  rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
  ])
  rbf_svm_clf.fit(X, y)
  ```
  - 그림 5-9 RBF커널을 이용한 SVM 분류기
  - gamma를 증가시키면 각 샘플을 따라 구불구불하게 휘어짐
  - gamma를 감소시키면 결정 경계가 부드러워짐
  - 하이퍼파라미터 r가 규제의 역할
    - 과대적합일 경우엔 감소
    - 과소적합일 경우엔 증가
     
####5.2.4 계산 복잡도
* 표5-1 SVM 분류를 위한 사이킷런 파이썬 클래스 비교

###5.3 SVM 회귀
* 제한된 마진 오류 안에서 도로 안에 가능한 한 많은 샘플이 들어가도록 학습
  - 도로의 폭은 하이퍼파라미터 ε로 조절
  - 그림 5-10 svm 회귀
* 사이킷런의 LinearSVR을 사용해 선형 SVM 회귀 적용
  ``` 
  svm_reg = LinearSVR(epsilon=1.5)
  svm_reg.fit(X, y)
  ```
* 비선형 회귀
  - 그림 5-11 2차 다항 커널을 사용한 SVM 회귀
  - 사이킷런의 SVR을 사용
  - SVR은 SVC의 회귀 버전
  ```
  svm_poly_reg = SVR(kernel="poly", gamma='auto', degree=2, C=100, epsilon=0.1)
  svm_poly_reg.fit(X, y)
  
###5.4 SVM 이론
* SVM의 예측은 어떻게 이뤄지는가?
* SVM의 훈련 알고리즘이 어떻게 작동하는가?
####5.4.1 결정 함수와 예측
* 식 5-2 선형 SVM 분류기의 예측


####5.4.2 목적 함수

####5.4.3 콰드라틱 프로그래밍

####5.4.4 쌍대 문제

####5.4.5 커널 SVM

####5.4.6 온라인 SVM

###5.5 연습문제