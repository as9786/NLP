# Review of XLNet : Generalized Autoregressive Pretraining for Language Understanding

- GPT로 대표되는 자기 회귀 모형과 BERT로 대표되는 auto-encoder model의 장점만을 합한 generalized AR pretraining model
- Permutation language modeling objective와 two-stream attention mechanism

## 1. Introduction

## Autoregressive(AR)
- 언어 모형의 학습 방법으로 이전 token들을 보고 다음 token을 예측하는 문제룰 품
- ELMo, GPT, RNNLM 등

![image](https://user-images.githubusercontent.com/80622859/230704613-6c45e7d9-7956-449a-9bb2-2882919bebb5.png)

- 주어진 input sequence의 우도는 조건부 확률들의 곱으로 나타나짐
- 위와 같은 분포를 학습(Negative log likelihood)
- 방향성이 정해져야 하므로, 한쪽 방향의 정보만을 이용할 수 있음
- 양방향 문맥을 활용해 문장에 대해 깊이 이해하기 어려움
- ELMo의 경우 양방향을 이용하지만, 각각의 방향에 대해 독립적으로 학습 -> 얕은 이해

## Auto Encoding(AE)

- 주어진 입력에 대해 그 입력을 그대로 예측하는 문제를 품. Denoising AE는 noise가 섞인 입력을 원래의 입력으로 예측
- BERT

![image](https://user-images.githubusercontent.com/80622859/230704717-88f178ea-71e4-466d-9233-92e7ee8fbdbd.png)

- 독립 가정 : 주어진 input sequence에 대하 각 [MASK]의 정답 token이 등장할 확률은 독립이라 가정 -> 곱으로 표현 가능
- $x_t$ 가 [MASK]인 경우, $m_t=1$, 나머지 경우에는 $m_t=0$ => [MASK]에 대해서만 예측을 진행
- 양방향 정보를 사용(Bidirectional self-attention)
- 독립 가정으로 모든 [MASK]들이 독립적으로 예측됨으로써 이들 사이의 연관성을 학습 X
- 실제 미세 조정 단계에서는 [MASK]가 등장하지 않으므로, 사전 학습과 미세 조정 사이의 불일치 발생

## Proposed Method: XLNet

- 
