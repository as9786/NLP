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

#### 1. Language Model

- GPT. Left-to-right transformer language model
- BART의 decoder와 동일(cross-attention을 수행하진 않음)

#### 2. Permuted Langauge Model
- XLNet 기반 모형. 1/6 token만큼 sampling하여 이를 임의의 순서로 생성하는 모형
- 문장의 양방향을 고려하여 학습 가능

#### 3. Masked Language Model
- BERT와 같은 modeling 방법, 15%의 token을 [MASK]로 치환하고 모형을 각 token마다 기존 token을 예측하도록 훈련

#### 4. Multitask Masked Language Model

- UniLM에서 제안한 방법, MLM을 추가적인 self-attention mask를 통해 훈련
- Self attention의 mask는 임의의 비율로 선택
- 1/6 left-to-right, 1/6 right-to-left, 1/3 unmasked, 그리고 1/3의 처음 50% token은 unmask, 나머지 비율은 left-to-right mask

#### 5. Masked Seq-to-Seq 
- MASS
- 50% token을 포함하는 span을 mask하고, seq2seq model로 masked token을 예측

- left-to-right 방식 : 현재 time step의 왼쪽 token들만 반영
- 논문 실험 조건
1. 작업을 일반적인 seq2seq 문제로 취급. 입력을 encoder에 넣고 정답은 decoder의 출력이 된다
2. Source를 target의 prefix에 추가하여 decoder에 넣고, sequence의 target part의 손실만 계산(전형적인 decoder model)

- 1이 BART에 더 잘 작동

## Tasks

### SQuAD

- Wikipedia 문단에 대한 extractive QA 
- Wikipedia에서 따온 본문과 질문이 주어지면 주어진 본문으로부터 정답에 해당하는 text span을 찾는 문제
- 문제와 context를 이어붙인 것을 encoder의 입력으로하고 decoder를 통해 예측
- 각 token의 시작과 끝 index를 예측하는 분류기가 포함되어 있음
- Start token과 end token의 위치를 예측하는 분류기가 2개

### MNLI
- 두 개의 문장에 대한 분류 작업으로 하나의 문장이 다른 문장을 포함하는지, 즉 이전 문장과 이후 문장의 관계가 성립하는지 예측하는 작업
- BART는 두 개의 문장을 EOS token(문장의 끝)을 추가해 합치고, 이를 encoder와 decoder에 넣음
- EOS가 문장 관계를 분류하는데 사용

### ELI5
- 긴 형식의 abstractive QA
- 문제와 추가적인 문서를 붙인 것으로 조건으로 주어 답을 생성

### XSum
- News summarizationt task
- 함축된 요약을 생성

### ConvAI2
- 대화와 답변에 대한 생성 작업
- Context와 persona(화자)를 조건으로 줌

### CNN/DM
- News summarizationt task

![image](https://user-images.githubusercontent.com/80622859/229827898-d40e5bf2-a32d-4754-9c5c-ade85965b74a.png)

1. 사전 학습 방법론의 성능은 task 별로 확연한 차이가 있음
- 사전학습 방법론의 효율성은 작업에 크게 의존
- 간단한 언어 모형은 ELI5 dataset에서 최고의 성능, SQuAD task에서는 최악

2. Token Masking is important
- Rotating document나 permuting sentences 기반 사전학습 방법론은 해당 목적 함수로만 훈련시켰을 때 성능이 좋지 않음
- 성공적인 방법론들은 token deletion이나 token masking, 혹은 self-attention mask를 사용하는 방법
- Token deletion은 생성 작업에서 token masking보다 더 좋은 성능

3. Left-to-right 기반 언어 모형은 생성 작업에 효과적
- Masked Language model과 permuted language model은 생성 작업에서 다른 것들보다 성능이 떨어짐
- 위 두 모형은 사전 학습 단계에서 left-to-right auto-regressive language modeling을 적용하지 않은 모형들

4. Bidirectional encoder is important in SQuAD

5. 사전 학습 방법론 이외에도 중요한 요소가 있음
- Permuted language model은 기존 XLNet보다 성능이 떨어짐
- XLNet 구조를 그대로 따르지 않고 사전 학습 방법만 따름

6. Vanilla language model이 ELI5에서 최고의 성능
- BART는 느슨하게 연관되어 있는 문서에 대해 출력을 내는 작업에 덜 효과적

7. BART는 가장 일관성 있게 강력한 성능
- BART를 text infilling으로 학습한 모형이 모든 작업에서 좋은 성능을 보여줌

## Large-scale Pre-training Experiments

- 사전 학습은 큰 배치 크기와 큰 corpora로 이루어 졌을 때 downstream task의 성능이 엄청나게 증가(GPT)
- BART를 RoBERTa와 같은 scale로 훈련

### Experimental Setup

- Encoder와 decoder를 각각 12개를 두고 1024의 은닉 크기로 둔 큰 모형을 사전 학습
- 사전학습으로 8000 배치 사이즈, 모형을 500000 번 학습
- BPE 이용
- Text infilling, sentence permutation을 조합한 사전 학습 함수 사용
- 30% token을 각 문서에 masking, 모든 문장을 섞음
- 마지막 10% 학습에서는 dropout 적용 X
- RoBERTa에서 사용한 data 사용(News, 책, 이야기 ,web text로 이루어진 160GB의 크기)

### Result

### Discriminative tasks

![image](https://user-images.githubusercontent.com/80622859/229830956-dd81e00a-a440-490f-bff7-c1bfbcb3833b.png)

- 생성 작업에서의 성능 향상이 판별 작업에 대한 성능에 영향을 미치지 않음
- RoBERTa와 전체적으로 비슷함

### Generation Tasks

- BART는 미세 조정 단계에서 label smoothed cross entropy loss 사용. Smoothing parameter는 0.1로 설정
- Beam size = 5, beam search에서 중복된 trigram을 삭제, min-len, max-len, length penalty를 validation set을 생성할 때 모형에 적용

#### Summarization

![image](https://user-images.githubusercontent.com/80622859/229831752-5b4cd4cb-1b87-4f2d-b552-2d89e1a63abf.png)

- 상당한 성능 향상
- 질적으로도 요약에 대한 질이 좋았음

#### Dialogue

![image](https://user-images.githubusercontent.com/80622859/229832278-34eb2ac7-811e-4e5c-812e-a586917a0792.png)

- 이전 문맥과 text로 명시된 화자 둘 다 고려하여 응답을 생성
- F1 score, valid perplexity를 이용

#### Abstractive QA

![image](https://user-images.githubusercontent.com/80622859/229832450-1adb31b1-0af1-47f1-ba19-ba5dcf8b1757.png)

### Translation Task

![image](https://user-images.githubusercontent.com/80622859/229832656-1425ecbe-b191-4d29-b23b-2cc404d99e86.png)

![image](https://user-images.githubusercontent.com/80622859/229832928-16bbd275-ca52-4b89-af7b-23a4bc232d2f.png)

- BART는 요약 쪽에 강점

## Conclusions

- 손상된 문서를 기준 문서로 사상하는 것으로 학습하는 사전학습 방법론
- 생성 분야에서 강점
