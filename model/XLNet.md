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

- AR가 AE의 장점을 살리고 단점을 극복하기 위한 permutation language modeling method proposal

![image](https://user-images.githubusercontent.com/80622859/230705832-f9fcd051-bd2a-4360-8150-0584f5940800.png)

- Input sequence index의 모든 permutation을 고려한 AR 방식
- $[x_1,x_2,x_3,x_4]$에 대해서 순서의 permutation set은 총 4! = 24개 존재
- $Z_T = [[1,2,3,4],[1,2,4,3],[1,3,2,4],...,[4,3,2,1]]$로 나타낼 수 있음
- 위에서 구한 $Z_T$에 대한 AR LM의 목적 함수를 적용하면 아래와 같음

![image](https://user-images.githubusercontent.com/80622859/230705960-e3182ec3-cbda-4c6b-98ed-c00c8fba89b1.png)

- 각 token들은 원래 순서에 따라 positional encoding이 부여되고, permutation은 token의 index에 대해서만 진행
- Input sequence = ['나는', '세미나', '준비를', '하고', '있다', '.']
- 기존 순서일 경우 P(세미나|나는)P(준비를|나는,세미나)...P(.|나는,세미나,준비를,하고,있다)
- z = [2,3,4,5,6,1], ['세미나','준비를','하고','있다','.','나는']
- P(준비를|세미나)P(하고|세미나, 준비를)....P(나는|세마나,준비를,하고,있다,.)
- Sequence 자체의 순서를 섞는 것이 아니라 p(x)를 조건부 확률들의 곱으로 분리할 때 이 순서를 섞는 것
- 모형은 기존 sequence token들의 절대적 위치를 알 수 없음
- 모든 순열을 고려하는 것은 불가능
- 하나의 text sequence에 대해 하나의 순서를 표본 추출
- 학습하는 동안 거의 모든 순서에 대해 공유되므로, 많은 양의 data를 거치면 모든 순서를 고려하게 됨
- 양방향 문맥 정보 파악 가능

