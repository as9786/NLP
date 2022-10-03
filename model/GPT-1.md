# Improving Language Understanding by Generative Pre-training

## Introduction
- Labeling이 되어 있지 않은 data로 model을 학습시켜 labeling data를 이용했을 때의 단점을 극복하고 사람이 알지 못하는 data의 특성까지 model이 학습하게 하고, 이후 작은 수정만으로 효과적인 전이를 하게 함으로써 높은 성능을 달성할 수 있음 
- Labeling 되어 있지 않은 data로 학습시키는 것은 두 가지 문제가 있음
1. 결과물을 전이하기 위한 text representation을 학습시키는 것에 어떤 최적화 목적이 효과적인지 불분명
2. 학습된 특징을 목표 과업에 어떤 방식으로 전이할지 명확히 정의되어 있지 않음

- 본 연구에서는 비지도 사전 학습과 지도 미세 조정을 합친 준 지도학습 방식을 제안
- 최종 목표 : 일반적으로 높은 성능을 낼 수 있는 특성을 학습시켜 이후 조금의 변화를 통해 다양한 과업에 적용할 수 있는 model 만들기
- 이를 위해 ulabeled data와 task에 알맞은 labeled data가 있다고 가정하고, 해당 model은 labeling 되지 않은 data로 model의 초기 parameter를 학습하고, 이렇게 최적화된 parameter를 원하는 목적에 맞게 labeled data로 추가 학습
- Model의 구조는 transformer 사용 (다양한 task에서 최소한의 미세 조정을 통해 전이 학습 가능)

## Related Works

### Unsupervised pre-training
- 비지도 사전학습의 목적은 이후 수행될 supervised learning에 좋은 초기화 point를 제공
- 사전 학습 기법은 정규화 작용을 하여 DL model을 더 잘 일반화 시킴
- Transformer 구조는 더 긴 길이의 언어적인 구조를 포착할 수 있음
- Generative pre-training 방법은 전이 학습 시 아주 작은 변화만을 필요로 함

### Auxiliary training objectives

1. High-capacity language model을 학습
2. 특정 task에 맞는 labeled data로 미세 조정

## Framework

### Unsupervised pre-training
- Token으로 이루어진 corpus $v={v_1, v_2, v_3,...,v_n}$이 주어지면 다음의 우도를 최대화하는 standard language modeling objective를 사용

![캡처](https://user-images.githubusercontent.com/80622859/193550418-2f059dba-e204-43ee-8c69-16d4925b3eef.PNG)

- k : context window의 크기, $\Theta$ : 신경망의 parameter(해당 parameter는 SGD를 통해 훈련)
- 학습을 위한 language model로 multi-layer transformer decoder를 사용

![캡처](https://user-images.githubusercontent.com/80622859/193550770-9095a0dd-22ed-4b72-9dba-200979f34111.PNG)

- $U = \{u_{-k},...,u_{-1}\}$ : Token의 문맥 vector, n : layer의 수, $W_e$ : Token embedding matrix, $W_p$ : Position embedding matrix

### 


