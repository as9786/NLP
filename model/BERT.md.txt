# Bert

- Transformer의 encoder 부분만을 사용

![캡처](https://user-images.githubusercontent.com/80622859/177325498-afa579c4-a77f-4765-9fc2-16567be0d46a.PNG)

- L : encoder block의 수, H : 임베딩 벡터 또는 hidden state 벡터의 차원 수, A : multi-head attention에서 사용된 attention의 수
- BERT는 두 개의 문장으로 구성된 입력 데이터를 입력 받음.(하나의 문장도 가능)
- 입력 데이터를 토큰 단위로 쪼갬(WordPiece tokenization)
- 두 개의 토큰을 추가(가장 앞부분 : [CLS], 두 개의 문장을 구분하는 : [SEP])

![캡처](https://user-images.githubusercontent.com/80622859/177326006-21667fb2-a2c7-4bef-a870-c685918b9ce4.PNG)

- BERT가 출력하는 값은 각 토큰에 대한 hidden state 정보
- [CLS]는 입력된 두 개의 문장으로 구성된 시퀀스 데이터의 전체적인 특성 정보를 반영
- 토큰의 임베딩 정보 + positional embedding 정보 + 문장 임베딩 정보(BERT에서 추가)

![캡처](https://user-images.githubusercontent.com/80622859/177326448-30836d2c-d233-4172-a5b4-755004576f72.PNG)

- Sentence Embedding : 각 토큰이 어떤 문장에 속하는지 나타내기 위해 사용

## BERT 학습

- 두 가지 작업을 동시에 수행하면서 입력된 데이터를 구성하고 있는 토큰들의 임베딩 정보와 모형이 갖는 파라미터들의 값을 학습
1. Masked Language Model(MLM) : 입력된 데이터를 구성하는 토큰 중에서 일부의 토큰을 비워놓고 해당 단어를 맞히는 작업
- 입력된 텍스트 데이터에서 임의의 단어를 맞히는 것, 15%의 토큰을 random하게 mask
- 15% 토큰 중에서 80%는 mask로 , 10%는 임의의 단어로, 10%는 원래 단어로 대체 => 실제 데이터 분석을 위한 fine-tuning 과정에 존재하는 mismatch를 줄이고자
- 교차 엔트로피 함수를 통해서 학습

### Masking 과정

ex) 'my dog is cute and it has a long tail;

- 임의로 선택된 토큰이 'cute'라고 가정
- 그 중 80% [mask], 10% 임의로 'cute' ->'your', 10% 원래 단어 'cute'->'cute'
- [CLS] my doig is [MASK] and it has a long tail[SEP]
- 위의 문장을 학습 시 각 토큰에 대한 hidden state 정보 출력
- 위의 결과 중에서 [MASK] 토큰에 대한 결과인 T_[MASK]를 softmax 활성화 함수의 입력값으로 전달 

![캡처](https://user-images.githubusercontent.com/80622859/177327765-f28c9ef7-84e8-42ae-a9e5-705775ae943b.PNG)

2. Next Sentence Prediction(NSP)
- 두 개의 문장을 하나의 시퀀스 데이터로 입력 받아서 두 번째 문장이 실제로 첫 번째 문장 다음에 출현하는 문장인지 아닌지를 예측
- 정답일 경우 'IsNext', 아닐 경우 'NotNext'가 되는 학습 데이터를 가지고 학습
- [CLS] 토큰에 대한 hidden state 정보를 사용하여 정답을 예측 

- BERT는 위의 두 가지 작업의 비용함수를 이용해서 전체적인 비용함수를 구성하고, 해당 전체 비용함수를 최소화하게끔 파라미터들을 학습
- BERT는 사전학습모형 = 전이학습
1) fine-tuning : 우리가 풀고자 하는 새로운 문제에 대한 학습데이터를 이용해서 새롭게 학습, 사전학습모형이 갖고 있는 파라미터 값들을 초기값으로
2) Feature 기반 : 사전학습모형이 가지고 있는 파라미터의 값 그대로 사용
