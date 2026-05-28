# Training language models to follow instructions with human feedback

## 초록
- 인간의 선호(Human preference)를 활용하여 언어 모형이 사용자의 의도에 더욱 부합하는 응답을 생성하도록 만드는 방법론 제안
- 기존 언어 모형은 next token prediction 방식으로 학습되기 때문에 사용자의 실제 의도나 선호를 충분히 반영 못함
- 이를 해결하기 위해 보상 모형(return model) 학습
- PPO(Proximal Policy Optimization)을 이용하여 언어 모형이 더 높은 보상을 받는 방향으로 정책 최적화

## 1. 서론
- 최근 언어 모형은 실제 사용 환경에서는 다양한 문제 발생
  - 사용자의 의도 정확히 따르지 못함
  - 불필요한 응답 생성
- 기존의 언어 모형은 data의 statistic pattern을 학습하는 방식으로 동작
- 모형의 목적 함수는 다음 token을 정확하게 예측하는 것
- 인간 선호 기반 최적화를 통해 도움이 되는 답변 생성
- 인간 평가자의 선호를 보상 신호(reward signal)로 변환하여 강화학습에 활용

## 2. 방법

### 2-1. Pipeline

1. 사전 학습된 언어 모형
2. 미세 조정(지도 학습)
3. 보상 모형 학습
4. PPO based reinforcement learning

### 2-2. Supervised Fine-Tuning(SFT)
- 
