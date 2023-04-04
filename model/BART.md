# Bidirectional Auto-Regressive Transformer

## Introduction

- Self-supervised learning(자기 지도 학습) : Masked language model, 문장 내 존재하는 단어의 집합이 가려진 text를 다시 재구성하는 denoising autoencoder
- BERT 이후에는 mask token의 분포를 바꾸어 훈련
- 위와 같은 방법론은 특정 작업이나 종류에서 잘 작동하기 위한 정형적인 방법론. 모든 종류의 작업에 적용 X

![image](https://user-images.githubusercontent.com/80622859/229717106-83f0a377-4aae-4b53-b4d6-ab7ccf92ec81.png)

- BERT : 생성 작업 못함
- GPT : 양방향 문맥정보를 반영 X
- BART : 양방향성과 자가 회귀 모형을 합친 모형
- Seq2seq model로 만들어진 denoising autoencoder. 많은 종류의 downstream task에서 잘 동작
- 사전 학습 단계
1. Text를 임의로 noising(ex. 다른 mask로 교체하거나, 없애거나, 순서를 바꾸는 등)
2. Seq2seq model이 원래의 text를 복원하기 위해 학습

- 표준 transformer 사용
- BERT(Bidirectional encoder)와 GPT(left-to-right decoder)를 일반화
- Noising의 유연성

![image](https://user-images.githubusercontent.com/80622859/229723207-e72b6683-b2bc-444c-869e-154cc60e8175.png)

- 어떤 임의의 변형이라도 기존 text에 바로 적용될 수 있으며, 길이 변화도 가능
- 여러 noising 방법 중 기존 문장의 순서를 임의로 섞고 임의의 길이의 text를 하나의 단일 mask token으로 교체하는 것이 성능이 제일 좋음
- BERT의 MLM과 NSP를 일반화한 것. 모형이 전체적인 문장 길이에 대해 학습하고, 변형된 입력에 더 많이 집중하는 효과
- Text generation에 미세 조정하였을 때 효율적. 이해 작업에서도 좋은 성능
- Abstractive dialogue, QA, 요약에서 SOTA
- 기계 번역에서도 좋은 성

## Model

- 손상된 문서를 기존 문서로 되돌리는 denoising autoencoder 
- Noise text를 bidirectional encoder(BERT)가 encoding하고 이를 left-to-right autoregressive decoder(GPT)가 받음
- 사전학습은 MLE 사용

## Architecture

- GPT에서 사용되는 ReLU 함수를 GeLU로 바꿈. Parameter 초기화 N~(0,0.2)
- Base와 large 모형 존재
- Base는 encoder와 decoder가 6개 씩
- Large는 encoder와 decoder가 12개 씩
- 기존 transformer 모형처럼 decoder에는 self-attention과 encoder-decoder attention(cross attention)이 존재
- BERT는 단어를 유추해내기 위해 추가적인 FFN을 사용하지만 BART는 사용하지 않음(Encoder에서 masking된 단어를 유추하지 않음)
- BERT보다 약 10% 더 많은 parameter를 가짐


## Pretraining BART

- 재구성 손실 함수. Decoder output과 기존 문서와의 cross entropy를 최적화하는 것으로 훈련
- 어떠한 종류의 문서 corruption이든 모두 적용 가능(Output 개수가 정해져 있지 않고, 자기 회귀적으로 생성되기 때문)
- 극단적인 상황인 원몬에 대한 모든 정보가 없는 상황에서도 BART는 원래의 언어 모형과 동일하게 동작

### 1. Token masking
- BERT에서 제시한 기법
- Random token이 sampling되어 [MASK] token으로 치환
- 모형은 mask token이 어떤 token이었는지를 맞춤

### 2. Token Deletion

- 임의 token이 입력에서 삭제됨
- Token masking과 달리 모형은 누락된 입력 위치를 결정

### 3. Text infilling
- 몇 개의 text들을 샘플링 하여 한 span의 길이는 포아송 분포를 따름
- Poisson distribution 추출한 span length($\lambda = 3$)를 사용하여 여러 text span을 표본으로 추출
- 각 Span은 단일 mask token으로 대체
- 길이가 0인 span 또한 mask token으로 치환
- SpanBERT 기반. SpanBERT에서는 span의 길이를 다른 분포로 sampling하고 각 span을 가려진 단어 길이와 동일한 만큼 [MASK]로 치환
- 모형이 얼마나 많은 token이 하나의 span으로부터 없어졌는지 예측

### 4. Sentence prediction
- 하나의 문서가 마침표를 기준으로 문장별로 모두 분리. 분리된 문장들을 섞음
- 섞인 문장들을 원래 순서로 배열
- XLNet 기반

### 5. Document Rotation
- 하나의 token이 임의로 동일한 확률로 선택. 문서가 섞여 해당 token이 문서의 시작 지점
- 모형이 그 문서의 시작점을 찾음

## Fine-tuning BART

![image](https://user-images.githubusercontent.com/80622859/229819258-0469e18b-fcd7-4d1a-98aa-1ba1c86b17b7.png)

### 1. Sequence Classification Tasks
- Sequence를 분류하는 작업
- 동일한 입력이 encoder와 decoder로 들어가고, 마지막 decoder token의 마지막 은닉 상태가 새로운 분류기에 들어감
- BERT가 [CLS]로 분류하는 것에 영감받은 방법
- 추가적인 token을 마지막에 추가하여 decoder에 있는 마지막 token의 representation이 전체 입력이 attention을 수행할 수 있도록 함. 마지막 output은 모든 입력을 반영

### 2. Token Classification Tasks

- Token 단위로 분류를 수행
- 주어진 본문 내에서 정답을 찾는 작업
- 정답에 해당하는 start point와 end point를 찾아야 함
- 모든 문서를 encoder와 decoder를 입력으로 하고, 디코더의 최종 은닉 상태를 각 token representation으로 사용
- 각 token들의 representation을 분류하는데 사용

### 3. Sequence Generation Tasks

- 기존 BERT가 할 수 없었던 작업
- 추상적 질의응답 및 추상적 요약과 같은 생성 작업에 바로 적용 가능

### 4. Machine Translation

- 사전학습된 decoder를 사용하는 것에는 큰 이점이 없다고 알려져있음
- BART에서는 모형 전체를 하나의 사전 학습된 기계 번역을 위한 decoder로 사용할 수 있다는 것을 보여줌
- 새로운 encoder parameter를 추가
- BART encoder의 embedding layer를 새롭게 초기화된 encoder로 교체
- 해당 모형은 end-to-end 학습, 새로운 encoder를 학습시키는 것으로 외국어 단어들을 영어로 mapping하여 BART가 외국어 denoising할 수 있도록 함
- 새로운 encoder는 기존 BART 모형과 다른 단어 사전을 사용하여도 됨

![image](https://user-images.githubusercontent.com/80622859/229732879-ceeaefe1-fdcb-44e3-88bf-9357abf62770.png)

- 기계 번역을 위한 사전 학습된 decoder를 사용하고 새로운 encoder를 추가해서 encoder-decoder를 미세 조정
- 새로운 encoder는 외국어를 BART가 학습한 언어와 사상되는 역할
- 새로운 encoder는 BART와 다른 단어를 사용할 수 있음
- 학습을 두 단계로 함. Cross-entropy loss로 역전파 수행
1. 대부분의 BART parameter들은 그대로 두고 encoder와 BART의 position embedding, BART encoder의 첫 번째 layer(projection)만 학습
2. 모든 parameter 학습

## Comparing Pre-training Objectives

### Comparison Objectives

- 

