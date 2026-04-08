# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

## 1. 기존 문제점
- RLHF(Reinforcement Learning Human Feedback) 과정
  1. 사람의 선호도를 반영한 보상 모형 학습
  2. 강화학습 -> 정책이 평가하는 보상 최대화
- 비용이 높음. 복잡

## 2. DPO
- 보상 모형 X, 강화 학습 X
- 오직 대규모 자연어 모형만 학습

### 2-1. 손실 함수
- RLHF loss function

<img width="405" height="38" alt="image" src="https://github.com/user-attachments/assets/c3c4ca6a-8840-494a-8487-a4064573b2a7" />

- 보상 모형의 모수를 정책 모형의 모수로 바꿔야 함
- 1단계 학습 가능
- DPO loss function
  - $L_{DPO} (\pi_\theta;\pi_{ref}) = -E_{(x,y_w,y_l)\sim D}[log \sigma (\beta \frac{\pi_\theta (y_w|x)}{\pi_{ref} (y_w|x)}-\beta \frac{\pi_\theta (y_l|x)}{\pi_{ref} (y_l|x)})]$
  - y : 모형 추론 결과 (w : 선호, l : 비선호), $\pi$ : 최적화 모형
  - 선호되는 결과와 선호되지 않은 결과를 최적화 모형에 입력하여 선호되는 결과의 확률을 높이고 선호되지 않은 결과의 확률을 낮춤
