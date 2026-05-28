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
- 인간이 작성한 data를 사용하여 모형을 지도 학습 방식으로 미세 조정
- SFT만으로는 인간 선호를 반영하기 어려움. Data가 한정적

### 2-3. 보상 모형 학습
- 인간 평가자는 하나의 prompt에 대해 여러 응답을 비교하고 순위를 매김
- 보상 모형은 이러한 pairwise preference를 학습
- 선호되는 응답 -> 높은 보상
- Bradley-Terry model based loss
- 보상 차이를 기반으로 확률 계산
- $P(y_w \succ y_l)=\frac{e^{r_\theta(y_w)}}{e^{r_\theta(y_w)}+e^{r_\theta(y_l)}}$
- $y_w$ : 선호된 응답, $y_l$ : 덜 선호된 응답, $r_{\theta}$ : 보상 모형
- 인간이 더 좋아한 응답이 더 높은 보상을 가지도록 학습

### 2-3. PPO based reinforcement learning
- PPO를 통해 언어 모형 정책 최적화
- 목표는 보상 모형이 높은 점수를 주는 응답을 생성하도록 만드는 것
- $\max_\phi \mathbb{E}{y \sim \pi\phi}[r_\theta(y)]$
- 보상만 최대화하면 모형이 이상한 방향으로 붕괴 가능
- 위 문제를 보완하기 위해 KL penalty 추가
- $r(y)=r_\theta(y)-\beta D_{KL}(\pi_\phi || \pi_{SFT})$
- 보상은 높게 유지. 기존 언어 분포와 너무 멀어지지는 않도록 제한

## 3. 기여 및 한계
### 3-1. 기여점
- 언어 모형 일치 문제를 강화학습 관점에서 해결

### 3-2. 한계점
- 인간 평가 비용이 매우 큼
- 보상 모형 편향 문제
- 학습 불안정성
- Preference data quality dependency 
