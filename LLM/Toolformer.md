# Toolformer: Language Models Can Teach Themselves to Use Tools

## 초록
- 간단한 연산이나 사실 확인 등에 어려움을 겪음
- The model learns to use external tools in a simple API and present a language model capable of handling both text and computation effectively
- It learns which API to use in which situation, how to utilize the appropriate arguments and how to optimally combine the results
- 자기 지도학습을 통해 진행

## 1. 서론 
- 언어 모형은 확장성에 한계가 있음
- 해결하기 위한 가장 쉬운 방식은 외부 도구를 활용
- 하지만 위 방법은 인간의 주석에 의존. 또는 특정 작업에만 사용 가능하기에 광범위하지 못함
- Toolformer는 인간의 주석 없이 자기 지도학습을 통해 학습. 인간과 모형이 중요하게 생각하는 부분이 다를 수 있기 때문에 중요함
- 언어 모형은 일반성을 잃지 않고, 어떤 도구를 어떻게 사용할 것인지를 스스로 결정해야 됨. 이것은 기존 방식과 달리, 도구의 활용을 특정 작업에만 한정 짓지 않음
- In this study, human-written demonstrations are used to enable the language model to annotate potential API calls across a large-scale language dataset
- It uses self-supervised learning to determine which API calls are beneficial for the model in predicting future tokens
- The model is fine-tuned on API calls that it considers useful on its own
- 이 방식은 모형의 일반성과 언어 모형의 성능을 잃지 않음

## 2. 방법
- The objective of the model is to enable the language model to invoke diffferent API calls as needed according to the semantics of the API request
- The input and output of each API should be representable as text sequences
- Special tokens are assigned to the beginning and end of an API call, allowing API calls to be naturally inserted into any text

<img width="743" height="152" alt="image" src="https://github.com/user-attachments/assets/df0a0c79-3aa5-4430-9413-7e40873a0fb0" />

### 2-1. Sampling API Calls
- A prompt is designed for each API to encourage the language model to annotate API calls within example sentences
- 사람이 작성한 몇 개 샘플들 
- 예시) 입력 : 상점에서 사과를 12개씩 한 박스로 하여 3박스를 판다. 출력 : 상점은 [계산기 API(3 x 12)] 36개의 사과를 판다
- 모형은 위에서 숫자 계산이 필요하면 계산기 API를 호출한다는 것을 학습
- 새로운 문장에도 자동으로 API 삽입
- API 호출이 필요해 보이는 위치를 모형이 확률적으로 선택
- $p_M(<API>|z_1, ..., z_i)$
- 최대 k개까지 샘플링. 모든 위치에 API를 넣으면 너무 비효율적

