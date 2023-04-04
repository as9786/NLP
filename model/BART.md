# Bidirectional Auto-Regressive Transformer

## Introduction

- Self-supervised learning(자기 지도 학습) : Masked language model, 문장 내 존재하는 단어의 집합이 가려진 text를 다시 재구성하는 denoising autoencoder
- BERT 이후에는 mask token의 분포를 바꾸어 훈련
- 위와 같은 방법론은 특정 작업이나 종류에서 잘 작동하기 위한 정형적인 방법론. 모든 종류의 작업에 적용 X

![image](https://user-images.githubusercontent.com/80622859/229717106-83f0a377-4aae-4b53-b4d6-ab7ccf92ec81.png)

- BERT : 생성 작업 못함
- GPT : 양방향 문맥정보를 반영 X
- BART : 양방향성과 자가 회귀 모형을 합친 모형
- Seq2seq model로 만들어진 denoising autoencoder. 많은 종류의 downstream task에서 잘 동작
- 사전 학습 단계
1. Text를 임의로 noising(ex. 다른 mask로 교체하거나, 없애거나, 순서를 바꾸는 등)
2. Seq2seq model이 원래의 text를 복원하기 위해 학습

- 표준 transformer 사용
- BERT(Bidirectional encoder)와 GPT(left-to-right decoder)를 일반화
- Noising의 유연성
- 어떤 임의의 변형이라도 기존 text에 바로 적용될 수 있으며, 길이 변화도 가능
- 여러 noising 방법 중 기존 문장의 순서를 임의로 섞고 임의의 길이의 text를 하나의 단일 mask token으로 교체하는 것이 성능이 제일 좋음
- BERT의 MLM과 NSP를 일반화한 것. 모형이 전체적인 문장 길이에 대해 학습하고, 변형된 입력에 더 많이 집중하는 효과
- Text generation에 미세 조정하였을 때 효율적. 이해 작업에서도 좋은 성능
- Abstractive dialogue, QA, 요약에서 SOTA
- 기계 번역에서도 좋은 성능

## Model

- 
