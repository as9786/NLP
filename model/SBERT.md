# Sentence-BERT : Sentence Embeddings using Siamese BERT-Networks

## Introduction

- 의미론적으로 유의미한 sentence embeddings를 얻을 수 있는 siamese network와 triplet network를 사용하여 BERT network를 변형한 모형
- 대규모의 의미론적의 유사성 비교, 군집화, semantic research를 통한 정보 회수와 같이 BERT에서 적용되지 못한 작업들을 사용 가능
- BERT는 cross-encoder 사용. 
- 두 개의 문장이 입력으로 들어가게 되고, 정답을 예측
- 이러한 경우 조합의 경우의 수가 너무 많기 때문에, 다양한 pair regression task에 부적절
- BERT를 사용 시 10,000개의 문장 중에서 가장 높은 유사도 쌍을 계산할려면, 10000 * 9999 / 2 만큼의 연산이 필요
- 군집화와 semantic search를 다루는 일반적인 방법은 의미론적으로 유사한 문장이 가까워지도록 각 문장을 vector space에 mapping하는 것
- 가장 널리 사용되는 BERT의 출력층을 평균하거나 또는 CLS token을 사용하는 것은 GloVe보다 좋지 않을 경우도 있음
- Siamese network는 fixed-size vector를 얻을 수 있음
- 그 후 유사도를 계산하여 의미론적으로 유사한 문장이 탐색됨
- Fine tuning with NLI dataset

## Model

- 고정된 크기의 sentence embedding을 얻기 위해 BERT/RoBERTa의 출력에 대하여 pooling 수행
- Pooling에는 3 가지 방법이 있음
1. CLS token 사용
2. 모든 output vectors의 평균 연산
3. Output vectors의 max-over-time 연산

- 기본 설정 값은 2번
- BERT/RoBERTa를 미세 조정하기 위해, siamese network와 triplet network를 생성하여 가중치를 최신화. 생성된 sentence embedding이 유의미하고 cosine similarity로 비교될 수 있도록

### Classification Objective Function

- Sentence embedding u and v를 element-wise difference |u-v|와 연결하고 이를 학습 가능한 가중치로 곱함
- $ o = softmax(W_t(u,v,|u-v|))$
- n : dimension of sentence embedding, k : num of label
- cross entropy 

![image](https://user-images.githubusercontent.com/80622859/235296343-e970521c-2fb9-4876-b613-216d8a5b468c.png)

### Regression Objective Function

- u와 v 간 유사도 계산
- MSE

![image](https://user-images.githubusercontent.com/80622859/235296367-315cde3e-8cd1-40d1-be17-3b979d4af1a5.png)

### Triplet Objective Function

- Anchor sentence $\alpha$

![image](https://user-images.githubusercontent.com/80622859/235296456-6823ebc8-da3f-477e-b2e0-79735c06cdfe.png)

- SBERT는 SNLI와 Multi-Genre NLI dataset을 결합하여 학습
- SNLI는 contradiction(대조), entailment(수반), neutral(중립)의 label로 구성된 570,000개의 문장 쌍의 집합
- MulitNLI는 430,000개의 문장 쌍, 구어체와 문어체 범위를 다룸
- SBERT는 3-way softmax classifier objective function으로 1 epoch만큼 미세 조정
- 16 batch size, adam, 2e-5 learning rate

