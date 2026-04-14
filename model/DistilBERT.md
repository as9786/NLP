# DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter

## 초록
- 대규모 언어 모형은 모형의 크기가 커서 연산 비용이 큼
- Using knowledge distillation in pre-training phase, reduces the size of BERT by 40% and is 60% faster, while maintaining approximately 97% of its original performance
- Triplet loss

## 1. 서론
- 기존 언어 모형보다 훨씬 더 작고, 가볍고, 속도가 빠르면서 성능이 좋은 모형 제안

## 2. 방법

### 학생 모형
- The ocverall architecture is identical to BERT, but the number of layers is reduced by half, and the token-type embeddings and pooler are removed
- 층을 줄이는데 집중
- 교사 모형의 2개 층마다 하나의 층을 선택하여 학생 모형의 초기 가중치로 사용
- RoBERta method 
