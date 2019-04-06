##CHAPTER 2 머신러닝 프로젝트 처음부터 끝까지
* 처음부터 끝까지 진행하기 위한 주요 단계
  1. 큰 그림을 봅니다.
  2. 데이터를 구합니다
  3. 데이터로부터 통찰을 얻기 위해 탐색하고 시각화합니다.
  4. 머신러닝 알고리즘을 위해 데이터르를 준비합니다.
  5. 모델을 선택하고 훈련시킵니다.
  6. 모델을 상세하게 조정합니다.
  7. 솔루션을 제시합니다.
  8. 시스템을 론칭하고 모니터링하고 유지 보수합니다.

###2.1 실제 데이터로 작업하기
* 데이터 획득
  - 유명한 공개 데이터 저장소
    - UC 얼바인 머신러닝 저장소(http://archive.ics.uci.edu/ml/)
    - 캐글 데이터셋(http://www.kaggle.com/datasets)
    - 아마존 AWS 데이터셋(http://aws.amazon.com/ko/datasets)
  - 메타 포털
    - https://dataportals.org
    - https://opendatamonitor.eu
    - https://quandl.com
  - 저장소가 나열되어 있는 다른 페이지
    - 위키백과 머신러닝 데이터셋 목록(https://goo.gl/SJHN2k)
    - Quora.com질문(https://goo.gl/zDR78y)
    - 데이터셋 서브레딧(https://www.reddit.com/r/datasets)
* 캘리포니아 주택 가격으로 실습 진행
  - http://lib.stat.cmu.edu/datasets
  - 1990년 캘리포니아 인구조사 데이터 기

###2.2 큰 그림 보기반
* 해야 할 일
  - 캘리포니아 인구조사 데이터를 사용해 캘리포니아의 주택 가격 모델을 만드는 것
    - 데이터는 블록 그룹마다 population(인구), median income(중간 소득), median housing price(중간 주택 가격) 으로 구성됨
      - 블록 그룹은 최소한의 지리적 단위
  - 모델을 학습시켜서 다른 데이터가 주어졌을때 해당 구역의 주택가격을 예측해야함

####2.2.1 문제정의
* 비지니스의 목적 고찰
  - 모델 만드는 것이 목적이 아님
  - 모델을 사용해 어떻게 수익을 낼지에 대한 고민 필요
    - 문제를 어떻게 구성할지
    - 어떤 알고리즘을 선택할지
    - 모델 평가에 어떤 성능 지표를 사용할지
    - 모델 튜닝을 위해 얼마나 노력을 투여할지 결정
  - 예) 그림 2-2
    - 여러 가지 다른 신호와 함께 다른 머신러닝 시스템에 입력으로 사용
* 현재 솔루션의 구성 현황
  - 문제 해결 방법에 대한 정보 및 참고 성능으로 활용 가능
  - 예) 주택가격을 전문가가 수동으로 추정
    - 복잡한 규칙을 사용하여 추정
    - 10%이상 벗어남
  - 문제 정의
    - 지도학습, 비지도학습, 강화학습?
    - 분류, 회귀, etc?
    - 배치 학습, 온라인 학습?

####2.2.2 성능 측정 지표 선택
* 회귀 문제의 성능지표 : 평균 제곱근 오차(Root Mean Square Error:RMSE)
  - 식 2-1
    - 표기법 반드시 리뷰
  - 노름(norm)
    - 노름의 지수가 클수록 큰 값의 원소에 치우치며 작은 값 무시
    - RMSE(l2 norm)가 MAE(l1 norm)보다 조금 더 이상치에 민감함

####2.2.3 가정 검사
* 지금까지 만든 가정을 나열하고 검사해보는 것이 좋음
* 예) 하위 시스템에서 입력으로 들어가게 되는데 이 값을 그대로 사용할 거라 가정
  - 만일 하위 시스템이 ('저렴', '보통', '고가')의 카테고리로 바꾼다면
    - 회귀 -> 분류 문제로 바뀜

###2.3 데이터 가져오기

####2.3.1 작업환경 만들기
* 주요 패키지
  - 주피터, 넘파이, 판다스, 맷플롯립, 사이킷런
* 독립적인 환경 만들기
  - virtualenv(맥,리눅스), conda, docker etc..

####2.3.2 데이터 다운로드
* 데이터 다운로드 및 추출 함수 작성

####2.3.3 데이터 구조 훑어보기
* housing.head()
  - 총 10개의 특성
    - longuitude, latitude, housing_median_age, total_rooms, total_bedrooms, 
    population, households, median_incom, median_house_value, ocean_proximity
* housing.info()
  - 20,640개의 샘플
  - total_bedrooms 특성은 20,433개의 구역만 값이 있으며 207개 구역은 특성을 가지지 않음
    - 적절히 데이터 전처리 해줘야함
  - ocean_proximity는 텍스트 특성
    - 범주형
    - housing["ocean_proximity"].value.counts() 를 통해 카테고리에 있는 구역수 조회 가능
* housing.describe()
  - 숫자형 특성의 요약 정보를 보여줌
* housing.hist()
  - 히스토그램 출력
  1 median_income은 스케일 조정으로 0.5~15의 범위를 가짐
    - 데이터가 어떻게 계산된 것인지 이해하고 있어야 함
  2 housing_median_age, median_house_value는 최대값과 최소값을 한정
    - median_house_value는 레이블로 사용되기 때문에 심각한 문제 발생 가능
      a. 정확한 레이블 값을 구함
      b. 훈련 세트와 테스트 세트에서 해당 구역을 제거(50000$)
  3 특성들의 스케일이 많이 다름
  4 히스토그램의 꼬리가 두꺼움
    - 종 모양이 되도록 변형 필요

####2.3.4 테스트 세트 만들기
* 데이터 스누핑 편향
  - 테스트 세트를 보고 머신러닝 모델을 선택할시 일반화 오차는 적겠지만 실제 운영에서 성능 낮음
* 시도 1 : 20%의 데이터를 랜덤하게 선택
  - 프로그램 재 실행시 다른 테스트 세트가 생성
  - 여러번 시행하면 결국 데이터 스누핑 편향과 비슷해짐
* 시도 2 : 처음 실행에서 테스트 세트를 저장하고 다음번 실행에서 이를 불러옴
* 시도 3 : 항상 같은 난수 인덱스가 생성되도록 seed 지정
* 시도 2,3 모두 데이터셋이 업데이트 되면 문제 발생함
* 시도 4 : 샘플의 식별자를 사용하여 테스트 세트로 보낼지 말지 결정
  - 각 샘플마다 식별자의 해시값을 계산하여 해시의 마지막 바이트 값이 51(256의 20%)보다 작은 샘플만 테스트 데이터로 사용
  - 주택 데이터셋에는 식별자 컬럼이 없지만 행의 인덱스를 ID로 사용하면 간단히 해결
    - 새로운 데이터 추가시 데이터셋의 끝에 추가되어야 하며 행도 삭제되면 안됨
  - 경도와 위도의 조합은 좋은 식별자가 될 수 있음
* 시도 5 : 사이킷런의 train_test_split 이용
* 시도 1-5까지는 무작위 샘플링 방식
  - 데이터의 크기가 충분히 크지 않다면 샘플링 편향이 생길수 있음
* 계층적 샘플링(stratified sampling)
  - 인구 비율이 여성이 51.3%이고 남성이 48.7% 이라면 
    1000명을 뽑았을시 여성 513명, 남성 487명 이어야 함
  - 무작위 샘플링에서는 49%작거나 54%보다 많은 여성이 뽑힐 확률이 12%임.
* 중간 소득이 중간 주택 가격을 예측하는데 매우 중요하다고 가정할 시
  - 테스트 세트가 전체 데이터셋에 있는 소득 카테고리를 잘 대표해야함
  - 중간 소득은 숫자형이므로 카테고리로 만드는 작업 해야함
    - 계층별로 충분한 샘플수가 있어야 하므로 너무 많은 계층으로 나누면 안됨
* 계층적 샘플링은 사이킷런의 StratifiedShuffleSplit 이용
* 테스트 세트 생성은 머신러닝에서 아주 중요함
  - 교차 검증할때도 도움됨

###2.4 데이터 이해를 위한 탐색과 시각화

####2.4.1 지리적 데이터 시각화
* 위도와 경도로 산점도 표시
  - housing.plot(kind="scatter", x="longitude", y="latitude")
  - housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    - alpha 옵션을 0.1로 주어 밀집된 영역 표시
  - housing.plot(kind="scatter", x="longitude", y-"latitude", alpha=0.1
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
    - 원의 반지름 : 구역의 인구
    - 색깔 : 가격
    - 주택 가격은 지역과 인구 밀도와 관련이 많음

####2.4.2 상관관계 조사
* housing.corr()
  - 모든 특성간의 표준 상관계수(standard correlation coefficient) 계산
  - -1 ~ 1 사이의 값을 가짐
  - 1이면 관계가 높음
  - -1 이면 음의 상관관계
  - 0이면 상관관계가 없음(선형의)
  - 상관계수는 선형적인 상관관계만 측정
* 상관관계 확인
  - 숫자형 특성 사이에 산점도를 그려주는 판다스의 scatter_matrix 함수 사용
  ```
  from panda.plotting import scatter_matrix
  attributes = ["median_house_value", "median"income", "total_rooms", "housing_median_ago"]
  scatter_matrix(housing[attributes], figsize=(12,8))
  ```
* 중간 주택 가격과 중간 소득의 상관관계 산점도 확대
  - 상관관계가 매우 강함
  - 학습이 잘 되도록 수평선이 나타는 구간은 제거

####2.4.3 특성 조합으로 실험
* 여러 특성 조합 시도
  - 가구당 방 개수
  - 방당 화장실개수
  - 가구당 인원
* (프로토타입을 만들고 실행 -> 결과 분석 -> 새로운 통찰)  반복

###2.5 머신러닝 알고리즘을 위한 데이터 준비
* 데이터 준비는 함수를 만들어 자동화 해야함
  - 어떤 데이터셋에 대해서도 데이터 변환을 손쉽게 반복
  - 변환 라이브러리를 점진적으로 구축
  - 실제 시스템에서도 사용 가능
  - 다양한 데이터 변환을 쉽게 시도 가능

####2.5.1 데이터 정제
* 누락된 특성 처리
  - 해당 구역을 제거
    ```
    housing.dropna(subset=["total_bedrooms"])
    ```
  - 전체 특성을 삭제
    ```
    housing.drop("total_bedrooms", axis=1)
    ```
  - 특정 값을 채움(0, 평균, 중간값 등)
    ```
    median = housing["total_bedrooms"].median()
    housing["total_bedrooms"].fillna(median,inplace=True)
    ```
    ```
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    housing_num=housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X - imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, colums=housing_num.colums, index = list(housing.index.values)
    ```
    - 모든 숫자형 특성에 적용
* 사이킷럼의 설계 철학
  - 일관성
    - 추정기(estimator)
      - fit()
    - 변환기(transformer)
      - transform()
    - 예측기(predictor)
      - LinearRegression:predict()
  - 검사 기능
  - 클래스 남용 방지
  - 조합성
  - 합리적인 기본값

####2.5.2 텍스트와 범주형 특성 다루기
* 머신러닝 알고리즘은 숫자형을 다루므로 ocean_proximity를 텍스트에 숫자로 바꿔야함
  - 판다스의 factorize() 메서드 사용
  ```
  housing_cat = housing["ocean_proximity"]
  housing_cat_encoded, housing_categories = housing_cat.factorize()
  ```
    - 머신러닝 알고리즘은 숫자간의 연간관계에 의미를 생성
    - 이진 특성을 만들어 해결
* 원-핫 인코딩(one-hot encoding)
  - 한 특성만 1이고 나머지는 0
  ```
  encoder = OneHotEncoder(categories = 'auto')
  housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
  ```
  - scipy의 희소행렬(sparse matrix)
    - 0을 메모리에 저장하는 것은 낭비이므로 1인 index만 기억
    - 수천 개의 카테고리가 있는 범주형 특성일 경우 효율적

####2.5.3 나만의 변환기

####2.5.4 특성 스케일링
* 입력 숫자 특성들의 스케일이 많이 다르면 학습이 잘 안됨
  - min-max 스케일링
    - 정규화(normalization)라 부름
    - 0~1 범위에 들도록 값을 이동하고 스케일을 조정
    - 사이킷런의 MinMaxScaler 변환기 제공
  - 표준화(standardization)
    - 평균을 뺀 후 표준편차로 나누어 결과 분포의 분산이 1이 되도록 함
    - 상한, 하한이 없음
    - 이상한 값에 영향을 덜 받음
    - 사이킷런의 StandardScaler 변환기 제공

####2.5.5 변환 파이프라인
* 사이킷런은 연속된 변환을 순서대로 처리할 수 있도록 도와주는 Pipeline 클래스 제공
  - 마지막 단계에는 변환기와 추정기 모두 사용 가능
  - 나머지 단계는 변환기만 사용 가능
* FeatureUnion을 이용하여 파이프라인을 합칠 수 있음

###2.6 모델 선택과 훈련

####2.6.1 훈련 세트에서 훈련하고 평가하기
* 선형회귀 모델 훈련
  ```
  from sklearn.linear_model import LinearRegression
  lin_reg = LinearRegression()
  lin_reg.fit(housing_prepared, housing_labels)
  ```
* 적용
  ```
  some_data = housing.iloc[:5]
  some_labels = housing_labels.iloc[:5]
  some_data_prepared = full_pipeline.transform(some_data)
  print("Predictions:", lin_reg.predict(some_data_prepared))
  print("Labels:", list(some_labels))
  ```
* 오차 측정
  ```
  housing_predictions = lin_reg.predict(housing_prepared)
  lin_mse = mean_squared_error(housing_labels, housing_predictions)
  lin_rmse = np.sqrt(lin_mse)
  ```
  - 과소적합 해결 방법
    1. 더 강력한 모델 선택
    2. 훈련 알고리즘에 더 좋은 특성 주입
    3. 모델 규제 감소
  - LinearRegression을 DecisionTreeRegressor로 교체
    - 과대 적합 발생
      - 훈련 세트의 일부분으로 훈련을 하고 다른 일부분은 모델 검증에 사용해야함

####2.6.2 교차 검증을 사용한 평가
* 훈련 세트를 더 작은 훈련 세트와 검증 세트로 나누고
  더 작은 훈련 세트에서 모델을 훈련 시키고 검증 세트로 모델 평가
* 교차 검증 기능 사용
  -  K겹 교차 검증(K-fold cross-validation)
    - 훈련 세트를 폴드(fold)라 불리는 10개의 서비셋으로 무작위 분할
    - 10번 훈련 평가
    - 매번 다른 폴드를 선택해 평가에 사용하고 나머지 9개 폴드는 훈련에 사용
    ```
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
        scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores) # 교차함수는 효용 함수
    ```
* RandomForestRegressor 모델 시도
  - 특성을 무작위로 선택해서 많은 결정 트리를 만들고 그 예측을 평균 내는 방식
    - 앙상블 학습이라 부름
* DecisionTree나 RandomForest나 과대적합 발생
  - 모델을 간단히 하거나
  - 규제를 적용하거나
  - 더 많은 훈련 데이터를 모아야 함
* 사이킷런은 모델을 저장하고 불러올 수 잇음
  ```
  from sklearn.externals import joblib
  joblib.dump(my_model, "my_model.pk1")
  my_model_loaded = joblib.load("my_model.pk1")
  ```

###2.7 모델 세부 튜닝

####2.7.1 그리드 탐색
* 하이퍼파라미터 조합을 찾는 가장 단순한 방법
  - 수동으로 조정
* 사이킷런의 GridSerachCV 사용
  - 시도해볼 하이퍼파라미터의 범위 지정
  - 모든 하이퍼파라미터 조합에 대해 교차 검증을 사용해 평가

####2.7.2 랜덤 탐색
* 그리드 탐색 방법은 비교적 적은 수의 조합을 탐구할때 괜찮음
  - 탐색 공간이 커지면 RandomizedSearchCV를 사용

####2.7.3 앙상블 방법
* 모델의 그룹이 최상의 단일 모델보다 더 나은 성능을 발휘할 때가 많음

####2.7.4 최상의 모델과 오차 분석
* 오차를 만들었다면 문제 발생 원인을 찾고 해결방법은 찾아야함

####2.7.5 테스트 세트로 시스템 평가하기
* 테스트 세트에서 최종 모델 평가
  - 테스트 세트에서 성능 수치를 좋게 하려고 튜닝하려 시도하면 안됨
    - 새로운 데이터에 일반화 되기 어려움

###2.8 론칭,모니터링,시스템 유지 보수
* 실시간 성능 체크
  - 성능이 떨어졌을때 알람 통지
* 새로운 데이터를 사용해 주기적으로 모델을 훈련 시켜야함
* 시스템의 예측을 샘플링해서 평가
* 입력 데이터 품질 평가
  - 온라인 학습 시스템에서는 입력을 모니터링 하는 일이 중요
  
###2.9 직접 해보세요!

###2.10 연습문제