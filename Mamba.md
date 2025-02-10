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
- 
