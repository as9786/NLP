# Language Models are Few-Shot Learners 

## Abstract
- NLP 학습 방법은 여전히 수천, 수만의 예시 data를 통해 어느 task에 특화된 미세조정 단계를 요구
- 이와는 대조적으로, 사람은 일반적으로 단지 몇 개의 예시, 혹은 간단한 지시사항만으로도 새로운 언어 task를 구사
- 언어 모형의 크기를 키우는 것이 task에 대한 일반성과 few-shot 성능을 높이고, 미세조정 접근법을 사용한 이전의 SOTA와도 비등한 성능을 확보할 수 있음
- 1750억 개의 인자를 가지는 자기회귀 언어 모형인 GPT-3를 학습시켜, few-shot setting에서의 성능을 측정
- 어떤 경사의 update나 미세조정을 거치지 않고 오직 few-shot 설명을 취함

## 1. Introduction
- 최근 연구들은 model 구조가 일반적이더라도 여전히 task-specific한 dataset과 미세조정 단계를 필요로 함
- 위와 같은 한계를 없애는 것은 가치가 있음

1. 새 task마다 labeling이 전부 되어 있는 큰 dataset을 필요로 하는 것은 언어 모형의 활용성을 제한
2. 학습 data에 존재하는 거짓 상관관계를 활용할 수 있는 가능성이 모델의 표현력과 학습 분포의 협소함에 따라 증가.(일반화의 부족)
3. 인간은 대부분의 언어 task를 배우기 위해 대규모 감독학습용 dataset이 필요 없음

- 위와 같은 문제점을 해결하기 위한 방법 : Meta-learning
- Meta-learning : 언어 모형의 문맥에서 model이 학습하는 동안 여러 기술과 pattern 인식 능력을 키우고, 추론 시간에는 이를 원하는 작업에 빠르게 적용시키거나 인식시키는 방법

- 문맥 내 학습 : 각 sequence에 대해 forward-pass 안에서 일어나는 내부 반복 과정


![02](https://user-images.githubusercontent.com/80622859/193557646-0b6fc714-55c7-4a80-984c-afb0481b5b10.png)

- Model이 단어에서 관련 없는 기호를 제거하도록 하는 task에서 few-shot learning 겨로가
- Few-shot learning 성능은 model 크기에 따라서도 크게 증가

## 2. Approach
- Model, data, 학습 등 기본적인 접근법은 GPT-2와 비슷
- Model의 크기를 키웠고, dataset의 크기와 다양성, 학습량을 전부 늘림

### 1. 미세조정
- 미세조정을 진행하지 않음

### 2. Few-shot
- Model이 추론 시간에서 단 몇 개의 예시만을 볼 수 있되, 가중치 최산화는 허용되지 않는 조건
- 일반적인 dataset 예시는 문맥과 원하는 답이 있고, few-shot은 단 K개의 문맥과 답이 주어짐. 이후 마지막으로 단 한 개의 문맥이 주어지면 model은 정확한 답을 생성해 내야함
- K = 10~100, 모델의 문맥 크기 = 2048
- Task-specific한 data에 대한 필요를 줄여줌
- SOTA에는 떨어지는 성능

### 3. One-shot
- Few-shor과 비슷
- 단 한 개의 예시와 task에 대한 자연어 지시문이 제공
- 사람이 소통하는 방법과 가장 흡사

### 4. Zero-shot
- 단 하나의 예시도 없으며, model은 단지 task에 대한 지시문만을 받음
- 가장 어려운 조건
- 사람조차도 예시가 없으면 task에 대해 제대로 이해할 수 없음
- ex) 200m 달리기를 위한 세계기록 표를 만들라 : 표가 어떤 형식이어야 하는지, 어떤 내용이 들어가야 하는지에 대한 명확환 설명 X

![04](https://user-images.githubusercontent.com/80622859/193558609-c6ad23fc-1c41-4c24-8663-c1294ca166e9.png)

## 2.1 Model and Architectures
- GPT-2와 동일한 구조

![05](https://user-images.githubusercontent.com/80622859/193558713-6b02695c-8d31-4e61-ae9d-2714d3a42185.png)

## 2.2 Training Dataset
- 거의 1조 개의 단어

## 2.3 Training Process
- 더 큰 batch를 쓰지만 learning rate는 더 작게
- 학습하는 동안 gradient noise scale을 측정하고 이를 batch size를 선택하는 가이드로 사용
- 각 행렬곱 내에서 모델 병렬화와 network layers에서 모델 병렬화를 섞어 사용

## 2.4 Evaluation
- 평가 dataset의 각 예제에 대해 훈련 set에서 조건으로 K개의 sample을 뽑아 평가


