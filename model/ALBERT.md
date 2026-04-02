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
