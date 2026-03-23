# Finetuned language models are zero-shot learners

## 서론
- 자연어 처리 작업을 지시문의 형태로 변환하여 미세 조정 진행
- 이전 -> 입력 : 너의 이름은 뭐니? 
- 해당 방법론 -> 입력 : 너의 이름은 뭐니를 영어로 번역해줘

<img width="690" height="226" alt="image" src="https://github.com/user-attachments/assets/b3ac6883-c9b8-4b41-904e-2c691e326b57" />

## 방법
- 기존 대규모 자연어 모형을 지시문을 통해 조정

### Task & Templates
- Dataset are grouped by task to form as single cluster

### Evaluation splits
- 학습에서 다루지 않은 작업에 대한 평가

### 학습 세부 사항
- LaMDA-PT model
- Example-Proportional mixing scheme -> Dataset size limit

<img width="186" height="67" alt="image" src="https://github.com/user-attachments/assets/70a270da-e252-4d30-b1c6-76619a5d9b8f" />

- N : 작업의 개수, $e_n$ : Each dataset corresponding to each task, m : 여러 작업들 중 임의의 작업을 뽑았을 때. 해당 작업의 식별자, $r_m$ : 임의의 m번째 작업의 dataset으로부터 추출할 확률

<img width="718" height="327" alt="image" src="https://github.com/user-attachments/assets/9ddb1e90-0b67-4585-8a7f-c76ae8b14247" />


## 결과
- 성능 향상에 도움
- 불완전한 문장이나 문단을 완성하는 작업에 대해서는 효과적이지 않음
