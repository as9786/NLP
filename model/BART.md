# Bidirectional Auto-Regressive Transformer

## Introduction

- Self-supervised learning(자기 지도 학습) : Masked language model, 단어의 무작위 하위 집합이 masked text를 재구성하도록 훈련된 noise auto-encoder
- 위와 같은 방식은 일반적으로 특정 유형의 작업에만 중점(범위 예측, 생성 등)
- 양방향 및 자가 회귀 모형을 결합 => BART
- BART는 매우 광범위한 최종 작업에 적용될 수 있는 seq2seq model로 구축된 noise removal auto-encoder
- 
