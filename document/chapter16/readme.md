##CHAPTER 16 강화 학습
* 강화 학습이 무엇인지?
* 강화 학습이 어떤 일을 잘할 수 있는지?
* 정책 그래디언트(policy gradient)와 심층 Q-networks(DQN) 기술 소개
* 마르크프 결정 과정(Markov decision process:MDP) 소개

###16.1 보상을 최적화하기 위한 학습
* 에이전트(agent)는 관측(observation)을 하고 주어진 환경(environment)에서 행동(action)을 함
* 그 결과로 보상(reward)을 받음 
* 강화학습 사례
  - 보행 로봇 제어
  - 미스 팩맨 제어
  - 보드 게임 플레이
  - 사람의 요구 예측
    - 온도조절
  - 주식시장의 가격 관찰

###16.2 정책 탐색
* 정책(policy)
  - 소프트웨어 에이전트가 행동을 결정하기 위해 사용하는 알고리즘
* 확률적 정책(stochastic policy)
  - e.g. 청소 로봇
    - 좌우 혹은 앞뒤로 움직일 확률 존재
    - 정책 파라미터(policy parameter)
      - 확률 p와 각도의 범위 r
* 정책 탐색(policy search)
  - 가능한 모든 것들을 해보고 p와 r정의
    - 모래에서 바늘 찾는격
  - 유전 알고리즘(generic algorithm)
  - 정책 그래디언트(policy gradient:PG)
    - 정책 파라미터에 대한 보상의 그래디언트를 평가해서 경사 상승법으로 파라미터를 수정하는 최적화 기법     

###16.3 OpenAI 짐
* e.g. CartPole 환경
  ``` 
  env = gym.make("CartPole-v0")
  obs = env.reset()
  ```
  - obs
    - 그림 16-4 CartPole 환경
    - 카트의 위치(중앙 0), 속도, 막대의 각도, 각속도
  ``` 
  action = 1
  obs, reward, done, info = env.step(action)
  ```
  - obs
    - 관측값
  - reward
    - 무조건 1의 보상
  - done
    - True면 에피소드 종료
  - info
    - 디버깅 정보(학습에 사용하면 안됨)
  ``` 
  def basic_policy(obs):
    angle = obs[2]
    return 0 if angle<0 else 1
  ```
  - 막대의 각도에 따른 action 정의
    - 최대 68번 살아 남음, 평균 42회
  
###16.4 신경망 정책
* 각 행동에 대한 확률 추정
  - 그림 16-5 신경망 정책
  - 새로운 행동을 탐험(exploring)하는 것과 잘 할 수 있는 행동 활용(exploiting)하는 것의 균형
* 각 관측이 환경에 대한 완전한 상태를 담고 있으므로 과거의 행동과 관측은 무시해도 됨
  - 담고있지 않다면 이전의 관측도 고려해야함
  - 관측에 잡음이 있다면 현재의 상태를 추정하기 위해 몇개의 지난 관측 사용해야함
  - CartPole은 관측에 잡음이 없고 환경에 대한 완전한 상태를 담고 있음
  ``` 
  # 1. 신경망 구조 정의
  n_inputs = 4
  n_hidden = 4
  n_outputs = 1
  initializer = tf.variance_scaling_initializer()
  
  # 2. 신경망 구성
  X = tf.placeholer(tf.float32, shape=[None, n_inputs])
  hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
  logits = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
  outputs = tf.nn.sigmoid(logits)
  
  # 3. 추정된 환률을 기반으로 랜덤하게 행동을 선택
  p_left_and_right = tf.concat(axis=1, values=[outputs, 1-outputs])
  action = tf.multinomal(tf.log(p_left_and_right), num_samples=1)
  init = tf.gloabl_variables_initializer()
  
  ```
###16.5 행동 평가:신용 할당 문제
* 강화학습에서 에이전트가 얻을 수 있는 도움은 보상뿐
  - 보상은 드물고 지연되어 나타남
* 신용 할당 문제(credit assignment problem)
  - 에이전트가 보상을 받았을 때 어떤 행동 덕분인지 알기 어려움
  - 각 단계마다 할인 계수(discount factor)를 적용한 보상을 모두 합하여 행동을 평가
    - 할인 계수 : 0.95, 13스텝만큼의 미래에서 받은 보상의 절반 가치
    - 할인 계수 : 0.99, 69스텝만큼의 미래에서 받은 보상의 절반 가치
  - 많은 에피소드를 실행하고 모든 행동의 점수 정규화

###16.6 정책 그래디언트
* PG 알고리즘은 높은 보상을 얻는 방향의 그래디언트로 정책의 파라미터를 최적화하는 알고리즘
* REINFORCE 알고리즘
  - 인기있는 PG 알고리즘 중 하나
  - 몬테카를로 정책 그래디언트(Monte Carlo Policy Gradient)라고도 함
  1. 더 높은 가능성을 가지도록 만드는 그래디언트 계산
  2. 몇 번의 에피스도를 실행한 다음, 각 행동의 점수를 계산
  3. 각 그래디언트 벡터와 그에 상응하는 행동의 점수를 곱함
  4. 모든 결과 그래디언트 벡터를 평균 내어 경사 하강법 스텝을 수행
  ``` 
  y = 1. - tf.to_float(action) 
  learning_rate = 0.01
  cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  grads_and_vars = optimizer.compute_gradients(cross_entropy)
  ```
  - minimize() 메서드 대신 compute_gradients() 메서드 호출
    - 그래디언트 벡터/변수 쌍(훈련 변수마다 하나의 쌍)의 리스트를 반환
  ``` 
  gradients = [grad for grad, variable in grads_and_vars]
  ```
  - https://www.tensorflow.org/api_docs/python/tf/train/Optimizer
  ``` 
  gradient_placeholders = []
  grads_and_vars_feed = []
  for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
  training_op = optimizer.apply_gradients(grads_and_vars_feed)
  ```
  - gradient_placeholder는 뒤에서 그래디언트 평균가로 학습됨
  - dev discount_rewards(rewards, discount_factor)
    - 할인 계수를 적용한 값 도출
  - def discount_and_normalize_rewards(all_rewards, discount_factor)
    - 각 행동 점수를 정규화하여 반환
  - 정책 훈련
    - 훈련 반복마다 10번의 에피소드에 대해 정책을 실행
    - 각 스텝마다 선된 행동이 최선인 양 그래디언트 계산
    - discount_and_normalize_rewards() 함수를 사용하여 행동 점수 ㄱㅖ산
    - 그래디언트 벡터와 행동 점수 곱
    - 평균 그래디언트를 주입하여 훈련 연산 실행

###16.7 마르코프 결정 과정
* 마르코프 연쇄(Markov chain)
  - 정해진 개수의 상태를 가지고 있으며, 각 스텝마다 한 상태에서 다른 상태로 랜덤하게 전이
  - 상태 s에서 상태 s'로 전이하기 위한 확률은 고정
  - 과거 상태에 상관없이 (s,s')의 쌍에만 의존
  - 그림 16-7 마르코프 연쇄의 예
* 마르코프 결정 과정
  - 마르코프 연쇄와 비슷
  - 여러 가능한 행동중 하나를 선택
  - 전이 확률은 선택된 행동에 따라 달라짐
  - 어떤 상태로의 전이는 보상을 반환
  - 에이전트의 목적은 시간이 지남에 따라 보상을 최대화히기 위한 정책을 찾는 것
  - 그림 16-8 마르코프 결정 과정의 예
  - 어떤 상태 s의 최적의 상태 가치(state value) V*(s)를 추정하는 방법
  - 에이전트가 상태 s에 도달한 후 최적으로 행동한다고 가정하고 평균적으로 기대할 수 있는 할인된 미래 보상의 합
  - 식 16-1 벨만 최적 방정식
    - T(s,a,s')는 에이전트가 행동 a를 선택했을 때 상태 s에서 상태 s'로 전이될 확률
    - R(s,a,s')는 에이전트가 행동 a를 선택해서 상태 s에서 상태 s'로 이동되었을 때 에이전트가 받을 수 있는 보상
  - 식 16-2 가치 반복 알고리즘
    - Vk(s)는 알고리즘의 k번째 반복에서 상태 s의 추정 가치
  - Q-가치(Q-Value)라고 부르는 최적의 상태-행동(state-action value) 가치를 추정하는 알고리즘
  - 식 16-3 Q-가치 반복 알고리즘
    - Q함수는 에이전트가 상태 s에 도달해서 행동 a를 선택하고 이 행동의 평균적으로 기대할 수 있는 할인된 미래 보상의 합
    
###16.8 시간차 학습과 Q-러닝

####16.8.1 탐험 정책

####16.8.2 근사 Q-러닝과 딥 Q-러닝

###16.9 DQN 알고리즘으로 미스 팩맨 플레이 학습하기

###16.10 연습문제