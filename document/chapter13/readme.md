##CHAPTER13 합성곱 신경망

###13.1 시각 피질의 구조

###13.2 합성곱층
* 합성곱층(convolutional layter)
  - 그림 13-2 층과 제로 패딩 사이의 연결
    - 첫번째 합성곱층은 입력이미지의 모든 픽셀에 연결되는 것이 아님
    - 합성곱층 뉴런의 수용장 안에 있는 픽셀에만 연결
    - 첫 번째 은닉층에서는 저수준 특성을 집중
    - 그다음 은닉층에서는 고수준 특성으로 조합해 나감
  - 제로 페딩(zero padding)
    - 높이와 너비를 이전 층과 같게 하기 위해 입력의 주위에 0을 추가
  - 스트라이드(stride)
    - 연속된 두 개의 수용장 사이의 거리
    - 그림 13-4 스트라이드를 사용해 차원 축소하기
  
####13.2.1 필터
* 뉴런의 가중치는 수용장 크기의 작은 이미지로 표현
  - 특성 맵(feature map)
    - 같은 필터를 사용한 전체 뉴런의 층은 필터와 유사한 이미지의 영역을 강조
    - 훈련 과정에서 CNN은 해당 문제에 가장 유용한 필터를 찾고 이들을 연결하여 더 복잡한 패턴을 학습

####13.2.2 여러 개의 특성 맵 쌓기
* 합성곱층이 입력에 여러 개의 필터를 동시에 적용하여 입력에 있는 여러 특성을 감지할 수 있음
* 그림 13-6 여러 개의 특성 맵으로 이루어진 합성곱층과 세 개의 채널을 가진 이미지

####13.2.3 텐서플로 구현
``` 
import numpy as np
from sklearn.datasets import load_sample_images

china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
dataset = np.array([china, flower], dtype=np.float32)
batch_size, height, width, channels = dataset.shape

filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1 # 수직선
filters[3, :, :, 1] = 1 # 수평선

X = tf.placeholer(tf.float32, shape=(None, height, width, channels))
convolution = tf.nn.conv2d(X, filters, strides=[1,2,2,1], padding="SAME")

with tf.Session() as sess:
  output = sess.run(convolution, feed_dict={X: dataset})
  
plt.imshow(output[0,:,:,1], cmap="gray")
plt.show()
```
* tf.layers.conv2d() 함수
  - 필터 변수를(kernel이란 이름으로) 만들고 랜던하게 초기화
  - 실제 CNN에서는 훈련 알고리즘이 최선의 필터를 자동으로 탐색
  ``` 
  X = tf.placeholder(shape=(None, height, width, channels), dtype=tf.float32)
  conv = tf.layers.conv2d(X, filters=2, kernel_size=7, strides=[2,2], padding="SAME")
  ```

####13.2.4 메모리 요구사항
* 각 층에서 필요한 RAM 양의 전체 합만큼 필요
  - 메모리가 부족하다면 미니배치 크기 조절
  - 스트라이드를 사용해 차원 축소
  - 몇개의 층 제거
  - 32비트 부동소수 대신 16비트 부동소수 사용
  - 여러 장치에 CNN을 분산

###13.3 풀링층
* 풀링층(Pooling layer)
  - 계산량, 메모리 사용량, 파라미터 수를 줄이기 위해 입력 이미지의 부표본(subsample)을 만드는 것
  - 그림 13-8 최대 풀링층(2*2 풀링 커널, 스트라이드 2, 패딩 없음)
  ``` 
  X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
  max_pool = tf.nn.max_pool(X, ksize=[1,2,2,1],strides=1,2,2,1], padding="VALID")
  
  with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict={X: dataset})
    
  plt.imshow(output[0], astype(np.unit8))
  plt.show()
  ```

###13.4 CNN 구조
* 그림 13-9 전형적인 CNN 구조

####13.4.1 LeNet-5

####13.4.2 AlexNet

####13.4.3 GoogLeNet

####13.4.4 ResNet

###13.5 연습문제
