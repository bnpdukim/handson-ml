##CHAPTER10 인공 신경망 소개
* 인경신경망의 초창기 구조
* 다층 퍼셉트론(Multi-Layer Perceptron:MLP)
* MNIST 숫자 분류 문제

###10.1 생물학적 뉴런에서 인공 뉴런까지

####10.1.1 생물학적 뉴런

####10.1.2 뉴런을 이용한 논리 연산
* 그림 10-3 간단한 논리 연산을 수행하는 인공 신경망

####10.1.3 퍼셉트론
* TLU(threshold logic unit) 기반
  - 각각의 입력 연결은 가중치와 연관
  - 입력의 가중치 합을 계산하고 계산된 합에 계단 함수(step function)를 적용
  - 그림 10-4 TLU 
* 그림 10-5 퍼셉트론 다이어그램
  - 다중 출력 분류기(multioutput classifier)
  - 층이 하나뿐인 TLU로 구성
  - 편향 특성은 항상 1을 출력하는 편향 뉴런(bias neuron)으로 표현
* 훈련 방법
  - 헤브의 규칙(Hebbian learning)
    - '서로 활성화되는 세포가 서로 연결된다'
      - 연결된 가중치 강화
      - 식 10-2 펴셉트론 학습 규칙(가중치 업데이트)
* 사이킷런은 하나의 TLU 네트워크를 구현한 Perceptron 클래스를 제공
* 1969년 마빈 민스키와 시모어 페퍼트는 퍼셉트론의 약점 언급
  - e.g. XOR 분류 문제
* 퍼셉트론을 쌓아올려 일부 제약을 줄일 수 있음
  - XOR 분류 문제 해결가능
  - 다층 퍼셉트론(Multi-Layer Perceptron:MLP)
  - 그림 10-6 XOR 분류 문제와 이를 푸는 다층 퍼셉트론
  
####10.1.4 다층 퍼셉트론과 역전파
* 입력층 하나와 은닉층(hidden layer), 출력층(output layer)으로 구성
  - 은닉층이 2개 이상일 때 심층 신경망(deep neural network:DNN)이라고 함
  - 다층 퍼셉트론을 훈련시키기 힘듬
* 1986년 역전파(backpropagation) 훈련 알고리즘 소개
  - 후진 모드 자동 미분을 사용하는 경사하강법(643p 참조)
  - 예측을 만듬(정방향 계산)
  - 오차를 측정
  - 역방향으로 각 층을 거치면서 각 연결이 오차에 기여한 정도를 측정(역방향 계산, 편미분 이용)
  - 이 오차가 감소하도록 가중치를 조금씩 조정
  - 계단함수를 로지스틱 함수로 바꿈
* 그림 10-8 활성화 함수와 해당 도함수
* 클래스가 배타적일 때 출력층의 활성화 함수로 소프트맥스(softmax) 함수 적용
* 피드포워드 신경망(feed forward neural network:FNN)
  - 신호가 한방향으로만 흐르는 구조
* 그림 10-9 분류에 사용되는(ReLU와 소프트맥스를 포함한) 현대적 다층 퍼셉트론

###10.2 텐서플로의 고수준 API로 다층 퍼셉트론 훈련
* 텐서플로우의 TF.Learn 사용
```
dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300,100], n_classes=10,
                                     feature_columns=feature_cols) 
```
* DNNClassifier 클래스의 출력층은 소프트맥스함수고 비용함수는 크로스 엔트로피

###10.3 텐서플로의 저수준 API로 심층 신경망 훈련하기
* 미니배치 경사 하강법을 저수준 API로 구현

####10.3.1 구성 단계
* 입력, 출력, 은닉층 뉴런수 설정
```
n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10 
```
* 플레이스홀더 노드를 사용해 훈련 데이터와 타깃을 표현
``` 
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
```
* 한개의 층을 만드는 neuron_layer() 정의
``` 
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z
```
* 3개의 층 정의
``` 
with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",  activation=tf.nn.relu)
    logits = neuron_layer(hidden2, n_outputs, name="outputs")
```
* 텐서플로의 표준 신경망층을 만드는 함수 사용
  - tf.layers.dens() 사용
    - 예전에는 tf.contrib.layers.fully_connected() 였음
``` 
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
    y_proba = tf.nn.softmax(logits)
```
* 크로스 엔트로피 함수로 sparse_softmax_cross_entropy_with_logits() 사용
``` 
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
```
* GradientdescentOptimizer를 사용해 비용 함수를 최소화시키도록 모델 파라미터 조정
``` 
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
```
* 모델을 평가하는 방법
  - in_top_k(predictions, targets, k) 함수 이용
    - 예측값이 크기순으로 k번째 안에 들면 True, 그렇지 않으면 False
``` 
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
```

####10.3.2 실행 단계
* load_data()를 이용하여 미니배치로 하나씩 적재
``` 
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]
```
* 모델 훈련
``` 
n_epochs = 20
n_batches = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)

    save_path = saver.save(sess, "./my_model_final.ckpt")
```

####10.3.3 신경망 사용하기
* 훈련한 모델을 이용해 예측하기
``` 
with tf.Session() as sess:
    saver.restore(sess, final_model_path)
    accuracy_val = accuracy.eval(feed_dict={X: X_test, y: y_test})
``` 

####10.4 신경망 하이퍼파라미터 튜닝하기

####10.4.1 은닉층의 수

####10.4.2 은닉층의 뉴런 수

####10.4.3 활성화 함수