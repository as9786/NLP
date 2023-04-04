# Bidirectional Auto-Regressive Transformer

## Introduction

- Self-supervised learning(자기 지도 학습) : Masked language model, 문장 내 존재하는 단어의 집합이 가려진 text를 다시 재구성하는 denoising autoencoder
- BERT 이후에는 mask token의 분포를 바꾸어 훈련
- 위와 같은 방법론은 특정 작업이나 종류에서 잘 작동하기 위한 정형적인 방법론. 모든 종류의 작업에 적용 X
- BERT : 생성 작업 못함
- GPT : 양방향 문맥정보를 반영 X
- BART : 양방향성과 자가 회귀 모형을 합친 모형
- Seq2seq model로 만들어진 denoising autoencoder. 많은 종류의 downstream task에서 잘 동작
- 사전 학습 단계
1. Text를 임의로 noising(ex. 다른 mask로 교체하거나, 없애거나, 순서를 바꾸는 등)
2. Seq2seq model이 원래의 text를 복원하기 위해 학습
3. 
