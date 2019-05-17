##CHAPTER 9 텐서플로우 시작하기
* 구글 브레인팀에서 개발
* 주요 기능 및 장점
  - 모바일 기기에서도 실행
  - 여러 종류의 신경망을 몇 줄의 코드로 훈련
  - 여러 가지 고수준 API(e.g. 케라스)가 텐서플로를 기반으로 구축
  - c++ API 제공으로 자신만의 고성능 연산 만들수 있음
  - 자동 미분(automatic differentiation:autodiff) 제공
  - 텐서보드 제공
  - 인적 인프라

###9.1 설치

###9.2 첫 번째 계산 그래프를 만들어 세션에서 실행하기
``` 
import tensorflow as tf
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2
```
* 실제로 어떤 계산도 수행하지 않음
  - 텐서플로 세션을 시작해야 함
  ```
  with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
  ```
  - with 문에서 선언한 세션이 기본 세션으로 지정
  ```
  init = tf.global_variables_initializer() # node 생성
  with tf.Session() as sess:
    init.run() # 실제 모든 변수 초기화
    result = f.eval()
  ```
  - global_variables_initializer() 함수는 초기화를 바로 수행하지 않고 노드 생성
  - run() 함수에서 변수 초기화 진행
* 텐서플로는 두 부분으로 나뉨
  - 구성 단계
    - 그래프 만들기
  - 실행 단계
    - 그래프 실행

###9.3 계산 그래프 관리
* 노드를 만들면 자동으로 기본 계산 그래프에 추가
* 독립적인 계산 그래프에 추가하기
  ``` 
  graph = tf.Graph()
  with graph.as_default():
    x2 = tf.Variable(2)
  x2.graph is graph # True
  x2.graph is tf.get_default_graph() # False
  ```
  
###9.4 노드 값의 생애주기
* 노드를 평가할 때 텐서 플로는 하나의 노드가 의존하고 있는 다른 노드들을 자동으로 찾아 평가함
  ``` 
  w = tf.constant(3)
  x = w + 2
  y = x + 5
  z = x * 3
  with tf.Session() as sess:
    print(y.eval()) # 10
    print(z.eval()) # 15
  ``` 
  - w 평가 2번, x 평가 2번
  - 모든 노드의 값은 계산 그래프 실행 간에 유지되지 않음
    - 평가된 w와 x를 재사용하지 않음
  ``` 
  w = tf.constant(3)
  x = w + 2
  y = x + 5
  z = x * 3
  with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val) # 10
    print(z_val) # 15 
  ```
  
###9.5 텐서플로를 이용한 선형 회귀


###9.6 경사 하강법 구현

####9.6.1 직접 그래디언트 계산

####9.6.2 자동 미분 사용

####9.6.3 옵티마이저 사용

###9.7 훈련 알고리즘에 데이터 주입

###9.8 모델 저장과 복원

###9.9 텐서보드로 그래프와 학습 곡선 시각화하기

  