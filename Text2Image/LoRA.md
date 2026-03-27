# LoRA : Low-Rank Adaptation of Large Language Models

## 초록
- 자연어 처리에서 중요한 추세는 대규모 사전 학습 모형을 특정 작업에 적용시키는 것
- 사전 학습 모형의 가중치가 커짐에 따라 모든 모형의 가중치를 재학습하는 방식은 현실적이지 않음
- LoRA : Freezes the pre-trained model weights + Injects trainable rank decomposition matrices into each layer of the transformer
- Reduces the number of parameters that need to be trained for downstream task
- 
