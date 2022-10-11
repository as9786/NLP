# Language Models are Unsupervised Multitask Learners

## Introduction
- 대규모 dataset, large model, 지도 학습 => 학습한 작업에서 좋은 성능, 그러나 민감하고 망가지기 쉬움
- General system

#### 주로 사용되는 방식
- 원하는 작업에 대한 훈련 dataset 수집
- 위와 같은 동작을 모방하도록 훈련
- Narrow expert에게는 좋은 결과
- 그러나 일반화 부족
- 광범위한 domain과 작업에 성능 측정 필요
- 사전 훈련된 지도 미세 조정 구조(여전히 지도 학습 필요)

- 본 논문에서는 전이 학습 사용
- 매개변수나 model 수정없이 zero-shot => 범용성 있는 언어 model 능력 가능성
- Single task 학습은 P(output|input)을 추정하는 확률 framework로 표현
- General system은 여러 다른 과제들을 수행해야 함(입력에 작업이 함께 표현)

## 2.1 Training dataset
- 이전의 작업은 news article, wikipedia를 대부분 사용(Single domain of text)
- 본 논문은 large and diverse dataset
- Common crawl과 같은 것이 존재(data 품질 문제)
- WebText 사용
- 인간에 의해 정제된 web page
- 중복 제거, wikipedia 문서 제거 등 전처리
- 40GB, 800만 개 이상 문서

## 2.2 Byte Pari Encoding
- 글자와 단어의 중간 단위 사용
- OOV 문제 해결
- Subword들을 활용하기 때문에 OOV와 신조어 같은 단어에 강점(하나의 단어는 더 작은 단위의 의미 있는 subword로 이루어져 있다는 가정)
- 기존에 있던 단어를 분리
- 글자 단위에서 점차적으로 단어 집합을 만들어내는 bottom-up 방식
- 단어들을 글자 또는 unicode 단위로 단어를 만들고 가장 많이 등장하는 unigram을 하나로 통합
- ex) aaabdaaabac
- 기본적으로 연속적으로 가장 많이 등장한 글자의 쌍을 찾아서 하나의 글자로 병합
- 위에서 가장 자주 등장하고 있는 byte pair는 'aa' -> Z로 치환
- ZabdZabac, Z = aa
- 위 문자열 중에서 가장 많이 등장하고 있는 byte pair는 ab
- ZYdZYac, Z = aa, Y = ab
- 가장 많이 등장하고 있는 byte pair ZY
- XdXac, X = ZY, Y = ab, Z = aa

## 2.3 Model
- Transformer decoder
- 기본적으로 GPT-1과 동일
- 층 정규화가 각 sub block의 입력으로 옮겨짐
- 층 정규화가 마지막 self-attention block 이후에 추가
- Model 깊이에 따른 residual path 초기화 방법 변경
- 사전 개수 5만 여개로 확장
- Context size가 1024개의 token으로 늘어남
- Batch size = 512
