# ALBERT : A Lite BERT for Self-supervised Learning of Language Representations


## 1. 서론
- 큰 신경망을 학습하고 작은 모형으로 증류하는 것이 일반적
- Parameter reduction technic
  - Factorized embedding parameterization : Embedding matrix -> Two small matrices
  - Cross-Layer parameter sharing : 신경망 깊이에 따라 가중치 수가 커지는 것을 방지
  - Reduce BERT parameter => Parameter-Efficiency
  - SOP(Sentence-Order Prediction)
  - 일관성 학습

 ## 2. 관련 연구

 ### 2-1. Cross-Layer Parameter Sharing
 - Network with cross-layer parameter sharing is better than standard transformer
 - The embeddings in BERT tend to oscillate rather than converge

## 3. 방법 

### 3-1. Factorized embedding parameterization
- 자연어 처리는 일반적으로 단어 사전 크기가 클수록 좋음
- 이로 인해 가중치의 수가 늘어나게 되며 학습 최신화는 천천히 됨
- The embedding parameters are factorized into two smaller matrices

### 3-2. Cross-Layer parameter sharing
- 층 사이의 모든 가중치를 공유 

### 3-3. Inter-Sentence coherence loss
- SOP loss. 문장 간 일관성 모형화
- Positive sample : Two consecutive segments from the same document. Negative sample : The order of the two segments is swapped


