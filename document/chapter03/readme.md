##CHAPTER 3 분류
###3.1 MNIST
* 다운받기
  ```
  from sklearn.datasets import fetch_mldata
  mnist = fetch_openml('mnist_784', version=1)
  ```
  - 데이터셋을 설명하는 DESCR 키
  - 샘플이 하나의 행, 특성이 하나의 열로 구성된 배열을 가진 data 키
  - 레이블 배열을 담고 있는 target 키
  - 이미지가 70,000개, 각 이미지에 784(28*28)개의 특성
  - 훈련과 테스트 세트가 나뉘어져 있음
    - 훈련 세트 : 앞쪽 60,000개
    - 테스트 세트 : 뒤쪽 10,000개
  - 훈련 세트를 섞어서 모든 교차 검증 폴드가 비슷해지도록 구성
  ```
  import numpy as np
  shuffle_index = np.random.permutation(60000)
  X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
  ```
  
###3.2 이진 분류기 훈련
* 확률적 경사 하강법(stochastic Gradient Descent:SGD) 분류기
  - 사이킷런의 SGDClassifier 클래스
    - 매우 큰 데이터셋을 효율적으로 처리
    ```
    from sklearn.linear_model import SGDClassifier
    sgd_clf = SGDClassifier(max_iter=5, random_state=42)
    sgd_clf.fit(X_train, y_train_5)
    ```
###3.3 성능 측정
####3.3.1 교차 검증을 사용한 정확도 측정
####3.3.2 오차 행렬
####3.3.3 정밀도와 재현율
####3.3.4 정밀도/재현율 트레이드오프
####3.3.5 ROC 곡선
###3.4 다중 분류
###3.5 에러 분석
###3.6 다중 레이블 분류
###3.7 다중 출력 분류
###3.8 연습문제
   