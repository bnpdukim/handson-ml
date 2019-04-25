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

###16.5 행동 평가:신용 할당 문제

###16.6 정책 그래디언트

###16.7 마르코프 결정 과정

###16.8 시간차 학습과 Q-러닝

####16.8.1 탐험 정책

####16.8.2 근사 Q-러닝과 딥 Q-러닝

###16.9 DQN 알고리즘으로 미스 팩맨 플레이 학습하기

###16.10 연습문제