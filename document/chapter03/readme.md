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
  - SGDClassifier-> RandomForestClassifier
    - decision_function() -> predict_proba()
      - 샘플이 주어진 클래스에 속할 확률을 담은 배열 반환
  ``` 
  from sklearn.ensemble import RandomForestClassifier(n_estimators=10)
  forest_clf = RandomForestClassifier(random_state=42)
  y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
  y_scores_forest = y_probas_forest[:,1]
  fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_traint5, y_scores_forest)
  roc_auc_score(y_traint_5, y_scores_forest)
  ```
  - 그림 3-7 ROC 곡선 비교
* 정리
  - 이진 분류기를 훈련시키는 방법
  - 작업에 맞는 적절한 지표 선택
  - 교차 검증을 사용한 평가
  - 요구사항에 맞는 정밀도/재현율 트레이드오프 선택
  - ROC곡선과 ROC AUC 점수를 사용하여 모델 비교
  
###3.4 다중 분류
* 둘 이상의 클래스를 구별
* 여러개의 클래스를 처리하는 알고리즘
  - 랜덤 포레스트, 나이브 베이즈 등
* 이진 분류만 가능한 알고리즘
  - 서포트 벡터 머신, 선형 분류기 등
* 이진 분류기를 여러개 사용해 다중 클래스를 분류 가능
  - e.g. 이진 분류기 10개(0~9)를 훈련시켜 클래스가 10개인 숫자 이미지 분류 시스템 구성 가능
    - 일대다(one-versus-all,one versus-the-rest:OvA) 전략
  - e.g. 0과 1 구별, 1과 2구별 등과 같이 각 숫자의 조합마다 이진 분류기 훈련
    - 일대일(one-versus-one:OvO) 전략
    - 클래스가 N개라면 분류기는 N*(N-1)/2개 필요
    - MNIST 기준으로 45개 분류기를 통과 시켜야함
  - 훈련 세트에 민감한 알고리즘은 OvO 선호(서포트 벡터 머신)
  - 대부분의 이진 분류 알고리즘은 OvA 선호
  ```
  sgd_clf.fit(X_train, y_train)
  sgd_clf.predict([some_digit])
  ```
    - 사이킷런이 실제로 10개의 이진 분류기를 훈련시킴
    ``` 
    some_digit_digit_scores = sgd_clf.decision_function([some_digit])
    index = np.argmax(some_digit_scores)
    sgd_clf.classes_[index]
    ```
      - 분류기가 훈련될 때 classes_ 속성에 타겟 클래스의 리스트를 값으로 정렬하여 저장
  - OvO나 OvA를 사용하도록 강제
    - OneVsOneClassfier나 OneVsRestClassfier 사용
    ``` 
    from sklearn.multiclass import OneVsOneClassifier
    ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, random_state=42))
    ovo_clf.fit(X_train, y_train)
    ovo_clf.predict([some_digit])
    len(ovo_clf.estimators_)
    ```
    - 랜덤 포레스트 분류기 훈련
    ``` 
    forest_clf.fit(X_train, y_train)
    forest_clf.predict([some_digit])
    forest_clf.predict_proba([some_digit])
    corss_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
    ```
      - 다중 클래스로 분류할 수 있기 때문에 OvA나 OvO를 적용할 필요가 없음
      - predict_proba() 메서드를 호출하면 분류기가 각 샘플에 부여한 클래스별 확률을 얻을 수 있음
      - 폴드 검증에서 84% 나옴
        - 입력 스케일 조정으로 정확도 상승 가능
        ``` 
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
        cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
        ```
        
###3.5 에러 분석
* 선택사항 탐색 -> 여러모델 시도 -> GridSearchCV를 사용해 하이퍼파라미터 튜닝
  - 모델 도출
* 모델 도출 후 모델의 성능을 향상시킬 방법
  - 만들어진 에러의 종류를 분석
    - 오차 행렬
    ``` 
    y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
    conf_mx = confusion_matrix(y_train, y_train_pred)
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()
    ``` 
      - matshow() 함수를 사용해 이미지로 표현
      - 숫자 5는 다른 숫자보다 조금 더어두워 보임
      - 오차 행렬의 각 값을 대응되는 클래스의 이미지 개수로 나누어 에러 비율 비교
      ``` 
      row_sums = conf_mx.sum(axis=1, keepdims=True)
      norm_conf_mx = conf_mx / row_sums
      np.fill_diagonal(norm_conf_mx,0)
      plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
      plt.show()
      ```
        - 에러가 정확하게 대칭이 아님
      - 성능 향상 방안 통찰
        - 3과 5, 8과 9를 더 잘 분리해야함
        - e.g. 동심원의 수를 세는 알고리즘
        - e.g. 동심원과 같은 패턴이 드러나도록 이미지 전처리
        - 3과 5의 샘플 보기
          - 선형 모델인 SGDClassfier를 사용했기 때문
            - 클래스마다 픽셀에 가중치를 할당
            - 새로운 미이지에 대해 필셀 강도의 가중치 합을 클래스의 점수로 계산
          - 해결 방안
            - 이미지를 중앙에 위치시키고 회정되어 있지 않도록 전처리
            
###3.6 다중 레이블 분류
* 분류기가 샘플마다 여러개의 클래스를 출력해야 할때도 있음
  - 여러개의 이진 레이블을 출력하는 분류 시스템
  - e.g. 얼굴 인식 분류기
    - 앨리스, 밥, 찰리 세 얼굴을 인식하도록 훈련된 상태
      - 앨리스, 찰리가 있는 사진 인식 -> [1,0,1] 
* 다중 레이블 분류기를 평가하는 방법
  - 프로젝트마다 다름
    - e.g. F1 점수를 구하고 평균 점수를 계산, 모든 레이블에 대한 F1 점수의 평균을 계산
    
###3.7 다중 출력 분류
* 다중출력 다중 클래스 분류(multioutput-multiclass classification) 
  - 간단히 다중 출력 분류(multioutput classification)
  - 다중 레이블 분류에서 한 레이블이 다중 클래스가 될 수 있도록 일반화한 것
    - e.g. 이미지 노이즈 제거
      - 분류기의 출력이 다중 레이블(픽셀당 한 레이블)
      - 각 레이블은 여러개의 값을 가짐(0~255)
      - 노이즈 추가
      ``` 
      noise = rnd.rnadint(0, 100, (len(X_train_), 784))
      X_train_mod = X_train + noise
      noise = rnd.randint(0, 100, (len(X_train_), 784))
      X_test_mod = X_test + noise
      y_train_mod = X_train
      y_test_mod = X_test
      ```
      - 노이즈 제거
      ``` 
      knn_clf.fit(X_train_mod, y_train_mod)
      clean_digit = knn_clf.predict([X_test_mod[some_index]])
      plot_digit(clean_digit)
      ```
      
###3.8 연습문제
   