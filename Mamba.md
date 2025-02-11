# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

## 초록

- 대부분의 모형들이 tranformer 및 attention 기반
- 하지만 transformer는 long-sequence 처리에 있어서 비효율적인 계산
- 이를 해결하기 위해 Mamba라는 모형 제안(상태 공간 모형)

## 1. Problem of transformer
- All text inputs = A sequence of tokens.
- 입력 받은 sequence의 이전 token을 돌아볼 수 있음

### A key component of the Transformers
- Decoder : Generative model
- Decoder = Masked self attention + FFN(Feed Forward Network)
- Self attention은 이전에 나온 모든 token과 비교하는 행렬을 만듦
- 행렬의 가중치는 pair of token이 서로 얼마나 관련이 있는지에 따라 결정
- 병렬 처리 가능 => 속도 향상

### 추론 시 문제점
- 다음 단어를 예측할 때, 전체 단어들에 대한 attention을 다시 계산해야 함
- 시간복잡도 : $O(n^2)$

### 순환 신경망
- 두 개의 입력을 받음 : 현재 입력 t와 이전 시간 입력인 t-1의 은닉 상태
- Transformer와 달리 이전 단계의 정보까지만 활용
- 빠른 추론 가능
- 다만, 이전 단계만 고려하기 때문에 기억 소실 문제 발생
- 병렬 처리 불가능

## 2. 상태 공간 모형(State Space Models, SSM)
- Sequence model

### 상태 공간이란
- 미로라는 공간을 가정하면, 상태 공간은 모든 가능한 공간을 의미
- 각 지점은 고유한 위치를 나타내며, 출구까지 얼마나 떨어져 있는지 등 구체적인 정보를 가지고 있음
- 상태 공간 표현은 현재 위치, 가능한 미래 상태, 다음 상태로 이동하는 변화로 표현
- 상태 공간 모형은 방정식과 행렬을 사용하여 위의 행동을 추적. 
