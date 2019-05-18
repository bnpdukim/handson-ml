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
* 텐서플로 연산은 여러 개의 입력을 받아 출력을 만들 수 있음
  - 입력과 출력은 텐서라는 다차원 배열임
  - 텐서를 평가한 결과가 넘파이 배열(ndarray)로 반환됨
* 158p 정규방정식 구현
  ``` 
  import numpy as np
  from sklearn.datasets import fetch_california_housing
  
  housing = fetch_california_housing()
  m, n = housing.data.shape
  housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
  
  X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
  y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
  XT = tf.transpose(X)
  theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
  
  with tf.Session() as sess:
    theta_value = theta.eval() 
  ```

###9.6 경사 하강법 구현
* 그래디언트 수동 계산
* 텐서플로의 자동 미분 기능 사용
* 텐서플로에 내장된 옵티마이저 사용

####9.6.1 직접 그래디언트 계산
```
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta") # feature의 개수
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
# https://medium.freecodecamp.org/machine-learning-mean-squared-error-regression-line-c7dde9a26b93
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul( tf.transpose(X), error)
training_op = tf.assign( theta, theta - learning_rate * gradients )

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  
  for epoch in range(n_epochs):
    if epoch % 100 == 0:
      print("Epoch", epoch, "MSE =", mse.eval())
    sess.run(training_op)
    
  best_theta = theta.eval() 
```

####9.6.2 자동 미분 사용
* 임의의 코드로 작성된 함수의 편미분을 계산하기 힘듬
  - 텐서플로의 자동 미분 기능으로 해결 가능
  - 위의 미분하는 코드를 아래의 코드로 변경
  ``` 
  gradients = tf.gradients(mse, [theta])[0]
  ```
* 텐서플로는 후진 모드 자동 미분(reverse-mode autodiff) 사용
  - 입력이 많고 출력이 적을 때 효율적이고 정확함


####9.6.3 옵티마이저 사용
* 텐서플로는 여러가지 내장 옵티마이저를 제공
  - 경사 하강법 옵티마이저 사용
  ``` 
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  training_op = optimizer.minimize(mse)
  ```
  - 모멘텀 옵티마이저 사용
  ``` 
  optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
  training_op = optimizer.minimize(mse)                                         
  ```
  
###9.7 훈련 알고리즘에 데이터 주입
* 미니배치 경사 하강법 적용
  - 플레이스홀더(placeholder) 노드 사용
    - 아무 계산도 하지 않는 특수한 노드
    - 실행 시에 주입한 데이터를 출력하기만 함
  ``` 
  X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
  y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
  
  batch_size = 100
  n_batches = int(np.ceil(m / batch_size))
  
  def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch
    
  with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
      for batch_index in range(n_batches):
        X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()
  ```
  
###9.8 모델 저장과 복원
* Saver 노드 추가
* 실행 단계에서 save() 메서드 호출
  ``` 
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  
  with tf.Session() as sess:
    sess.run(init)
  
    for epoch in range(n_epochs):
      if epoch % 100 == 0:
        save_path = saver.save(sess, "/tmp/my_model.ckpt")
      sess.run(training_op)
      
    best_theta = theta.eval()
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")
  ```
  - 모델 복원
  ``` 
  with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")
    best_theta_restored = theta.eval() 
  ```


###9.9 텐서보드로 그래프와 학습 곡선 시각화하기
* 훈련 통계값을 전달하면 반응형 그래프를 보여줌
* 계산 그래프의 정의를 사용하여 그래프 구조 확인 가능
``` 
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

with tf.Session() as sess: 
  sess.run(init)                                                                

  for epoch in range(n_epochs):                                                 
    for batch_index in range(n_batches):
      X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
      if batch_index % 10 == 0:
        summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
        step = epoch * n_batches + batch_index
        file_writer.add_summary(summary_str, step)
      sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()    
file_writer.close()
```
  - 그래프 확인 방법 
  ``` 
  tensorboard --logdir tf_logs/
  ```


###9.10 이름 범위
* 이름 범위(name scope)를 만들어 관련 있는 노드들을 그룹으로 묶어 노드를 구분할 수 있음
``` 
a1 = tf.Variable(0, name="a")      # name == "a"
a2 = tf.Variable(0, name="a")      # name == "a_1"

with tf.name_scope("param"):       # name == "param"
  a3 = tf.Variable(0, name="a")  # name == "param/a"

with tf.name_scope("param"):       # name == "param_1"
  a4 = tf.Variable(0, name="a")  # name == "param_1/a"

for node in (a1, a2, a3, a4):
  print(node.op.name)
```

###9.11 모듈화
* DRY(Don't Repeat Yourself) 원칙
* e.g. ReLU 함수
``` 
def relu(X):
  w_shape = (int(X.get_shape()[1]), 1)
  w = tf.Variable(tf.random_normal(w_shape), name="weights")
  b = tf.Variable(0.0, name="bias")
  z = tf.add(tf.matmul(X, w), b, name="z")
  return tf.maximum(z, 0., name="relu")
n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")
```
  - 이름 범위 사용
  ``` 
  def relu(X):
    with tf.name_scope("relu"):
      [...]
  ```


###9.12 변수 공유

###9.13 연습문제