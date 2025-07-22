# Deep contextualized word representations

## 초록
- 각 단어마다의 복잡한 특성과 문법적 이해를 언어적 맥락에 따라 학습
- 문맥에 따라 같은 단어라도 다르게 embedding을 주어 학습하여야 함(Embedding from Language Model)
- 문맥 학습을 위해 양자의 방법으로 학습(Bidirectional language model)
- 준지도 학습

## 서론
- 기존 방식은 같은 단어를 같은 embedding -> 복잡한 언어의 특성, 언어적 맥락 해결 X
- Bidirectional LSTM을 통해 문장 단위 embedding
- 두 가지 LSTM이 앞두 ㅣ문맥을 파악
- 낮은 층(입력과 가까운 층)은 품사 등 문법 정보. 높은 단계 층은 문맥 정보 학습

## ELMo: Embeddings from Language Models
- 문장 단위의 단어 표현

### 1. Bidirectional language models

<img width="719" height="695" alt="image" src="https://github.com/user-attachments/assets/8b151a34-89ee-4b0b-8769-e7aa61b94e44" />

- Forward와 backward의 각 parameter를 별도로 유지하면서 두 log likelihood를 공통적으로 최대화

<img width="366" height="98" alt="image" src="https://github.com/user-attachments/assets/9d322296-5f4b-4f99-8e12-fded86376eb7" />

- 두 parameter가 서로 완전히 독립적이지 않고 가중치들을 공유

### 2. ELMo
- 두 LSTM의 층 표현의 결합
- 각 token $t_k$마다 L-layer biLM과 처음 token layer를 합치면 2L+1개의 표현이 나옴

<img width="349" height="68" alt="image" src="https://github.com/user-attachments/assets/a1077292-ef9e-4a1f-8e80-5983885d8fa4" />

- 위 모형을 downstream task에 적용시키기 위해 모든 층들을 하나의 vector로 압축해야 함
- 단순하게 마지막 하나의 층을 선택

