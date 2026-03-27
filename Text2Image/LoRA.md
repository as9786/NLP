# LoRA : Low-Rank Adaptation of Large Language Models

## 초록
- 자연어 처리에서 중요한 추세는 대규모 사전 학습 모형을 특정 작업에 적용시키는 것
- 사전 학습 모형의 가중치가 커짐에 따라 모든 모형의 가중치를 재학습하는 방식은 현실적이지 않음
- LoRA : Freezes the pre-trained model weights + Injects trainable rank decomposition matrices into each layer of the transformer
- Reduces the number of parameters that need to be trained for downstream task

## 1. 서론

### 1-1. 기존 미세 조정 문제
- 모든 가중치를 재학습
- 시간 소요

### 1-2. LoRA
- 모형을 미세 조정할 때 실제로 필요한 가중치 변화는 낮은 차원에 있음
- 기존 가중치 : W(d x k)
- 미세 조정 후 가중치 : $W'=W+\nabla W$
- $\nabla$ : Low-Rank Matrix. 복잡 X. Low-Rank
- $\nabla=A \times B$ (A : d x r, B : r x k, r < d, k)
- 사전 학습 가중치를 고정한 채로, 각 층들의 행렬 분해를 최적화함으로써 기존의 층들을 간접 학습
- Rank decomposition matrix = Low-Rank matrix
- 저차원 행렬들을 다르게 둠으로써 다양한 작업에서 사용 가능
- 간단한 선형 연산으로 학습 가중치들과 기존 가중치를 병합하기 때문에 추론 지연 발생하지 않음



