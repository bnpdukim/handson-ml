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
    sgd_clf.predict([some_digit])
    ```
###3.3 성능 측정
####3.3.1 교차 검증을 사용한 정확도 측정
* 폴드가 3개인 K-겹 교차 검증
  ```
  from sklearn.model_selection import cross_val_score
  cross_val_score(sgd_clf, X_train, y_trin_5, cv=3, scoring="accuracy")
  ```
  - 정확도(accurarcy) : 95%???
* 더미 분류기
  ```
  from sklearn.base import BaseEstimator
  class Never5Classifier(BaseEstimator):
    def fit(self, X, y=none):
      pass
     def predict(self, X):
      return np.zeros((len(X), 1), dtype=bool)
  never_5_clf = Never5Classfier()
  cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
  ```
  - 정확도 : 90%???
* 정확도를 분류기의 성능 측정 지표로 선호하지 않음

####3.3.2 오차 행렬
* 오차 행렬(confusion matrix)
  - 클래스 A의 샘플이 클래스 B로 분류된 횟수를 세는 것
    - e.g. 숫자 5의 이미지를 3으로 분류한 횟수 -> 5행 3열 확인
  - 예측값 만들기
    ```
    from sklearn.model_selection import_val_predict
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    ```
    - cross_val_predict() 함수는 K-겹 교차 검증을 수행
      - 각 테스트 폴드에서 얻은 예측을 반환
  - 오차 행렬 만들기
    ```
    from sklearn,matrics import confusion_matrix
    confusion_matrix(y_train_5, y_train_pred)
    ```
    - 그림 3-2 오차 행렬
    - true negative(tn)
      - 5가 아닌 수를 5가 아니라가 판별
    - false positive(fp)
      - 5가 아닌 수를 5라고 판별
    - false negative(tn)
      - 5인 수를 5가 아니라고 판별
    - true positive(tp)
      - 5인 수를 5라고 판별
  - 정밀도(precision)
    - 식 3-1
  - 재현률
    - 식 3-2
####3.3.3 정밀도와 재현율
* 사이킷런을 이용한 정밀도와 재현율
  ```
  from sklearn.metrics import precision_score, recall_score
  precision_score(y_train_5, y_train_pred)
  recall_score(y_train_5, y_train_pred)
  ```
* F1 점수
  - 정밀도와 재현율의 조화 평균(harmonic mean)
  - 식 3-3
  ```
  from sklearn.metrics import f1_score
  f1_score(y_traint_5, y_train_pred)
  ```
####3.3.4 정밀도/재현율 트레이드오프
* 그림 3-3
  - 임계값을 높이면 정밀도가 높아짐
  - 임계값을 내리면 재현율이 높아짐
* predict() 메소드 대신 decision_function()을 호출하면 각 샘플의 점수 획득
  ```
  y_scores = sdg_clf.decision_function([some_digit])
  threshold = 0
  y_some_digit_pred = (y_scores > threshold)
  threshold = 20000
  y_some_digit_pred = (y_scores > threshold)
  ```
* 적절한 임계값의 선택
  - precision_recall_curve()으로 임계값 계산 가능
  ``` 
  from sklearn.metrics import precision_recall_curve  
  y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
  precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)  
  ```
  - 그림 3-4
####3.3.5 ROC 곡선
* 수신기 조작 특성(receiver operating characteristic:ROC) 곡선
  - 거짓 양성 비율(FPR)에 대한 진짜 양성 비율(TPR)
  - FPR : 양성으로 잘못 분류된 음성 샘플의 비율
    - 1에서 진짜 음성 비율(true negative rate:TNR)을 뺀 값
    - 1-TNR
    - TNR을 특이도(specificity)라고도 함
    - 민감도(재현율)에 대한 1-특이도 그래프
    - 137p 맨 아래
    ``` 
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
    ```
    - 그림 3-6
      - 좋은 분류기는 이 점선으로부터 최대한 멀리 떨어져 있어야 함(왼쪽 위 모서리)
  - 곡선 아래의 면적(area under the curve:AUC)
    ``` 
    from sklearn.metrics import roc_auc_score
    roc_auc_score(y_train_5, y_scores)
    ```
  - 양성 클래스가 드물거나 거짓 음성보다 거짓 양성이 더 중요할 때 PR 곡선 이용
  ``` 
  from sklearn.ensemble import RandomForestClassifier(n_estimators=10)
  forest_clf = RandomForestClassifier(random_state=42)
  y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
  y_scores_forest = y_probas_forest[:,1]
  fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_traint5, y_scores_forest)
  ```
  - 그림 3-7 ROC 곡선 비교
###3.4 다중 분류
###3.5 에러 분석
###3.6 다중 레이블 분류
###3.7 다중 출력 분류
###3.8 연습문제
   