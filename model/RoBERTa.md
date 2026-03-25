# RoBERta : A Robustly Optimized BERT Pretraining Approach

## 1. 서론
- 자기지도학습 방법을 사용 시 뛰어난 성능 향상
- 하지만 어떤 요소가 가장 큰 기여를 하는지 파악하기 쉽지 않음
- 모형을 더 오래 훈련. 더 큰 배치. More data
- Remove NSP
- 더 긴 문장 사용
- Dynamic masking

## 2. 방법

### 2-1. Dynamic masking
- BERT : Apply masking only once during the training data preprocessing stage
- Same mask pattern
- Generate and apply a new masking pattern at every training iteration
- 다양성 증가

### 2-2. Remove NSP
- 긴 문맥 학습 불가
- DOC-SENTENCES
- FULL-SENTENCES
- 긴 문장을 활용하는 것이 더 효과적
