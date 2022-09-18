# Distributed Representations of Words and Phrases and their Compositionality

## Abstract
- Skip-gram : 단어의 syntactic(구문론), sementic 관계를 효율적으로 표현할 수 있는 model
- 구문론 : 의미를 무시하고 기호 사이의 형식적 관계를 취급하는 학문
- 자주 사용되는 단어에 대한 subsampling을 통해 속도 향상과 더욱 규칙적인 단어를 표현 가능
- 계층적인 softmax의 대안을 제시

## Introduction
- Vector space에서 단어의 분산적인 표현은 유사한 단어를 grouping
- Efficient estimaton of word representations in vector space : 많은 양의 text data로부터 양질의 단어 표현을 할 수 있는 skip-gram model 제안. 이전에 사용하던 word vector를 훈련시키기 위해 신경망 구조와 다르게 행렬곱 연산을 사용하지 않음 => 효율적
 
![다운로드 (3)](https://user-images.githubusercontent.com/80622859/189517486-d740dfbb-028b-475e-b75b-7b776ae1c51b.png)

- 해당 논문에서는 이전 skip-gram의 성능을 개선
- 빈도수가 높은 단어를 sub-sampling함으로 훈련 속도를 2~10배 향상
- 빈도수가 적은 단어들의 표현에 대한 정확도를 높일 수 있음
- 계층적인 softmax 대신에 Noise Contrastive Estimation(NCE)를 단순화시켜 skip-gram model에 사용 => 속도 향상, 빈도수가 높은 단어에 대하여 더 나은 표현 가능
- 모든 관용구에 대하여 vector represent => skip-gram의 표현력 향상
- Recursive autoencoders : 결합된 단어로 구성된 문장의 의미를 표현하기 위한 기술. Word vector 대신 phrase vector 사용 가능
- Data-driven approach를 이용하여 다수의 phrases 확인 -> Phrases를 훈련 과정에서 개별적인 token으로 취급
- Phrases vector의 평가를 위해 word와 pharses를 모두 포함하고 있는 analogical reasoning task 
- 단순한 vector의 덧셈으로 의미있는 결과를 만들어낼 수 있음 발견
- ex) Russia + river ~= Volga River
- 언어에 대한 이해 정도를 수학적인 연산을 통해 

## The Skip-gram Model
- 문장이나 문서에서 주변 단어들을 예측하는 단어 표현을 찾는 것
- $w_1, w_2,w_3,...,w_t$에 대하여 log 확률의 평균을 최대화하는 것 

![다운로드 (4)](https://user-images.githubusercontent.com/80622859/189517716-171139d9-bc51-452b-bd7d-fb50f2d78d18.png)

- C : training context의 크기, 크기가 클수록 결과는 좋아지지만 훈련에 소요되는 시간도 증가
- 수식

![render (1)](https://user-images.githubusercontent.com/80622859/189517820-a09f16bb-5ad2-4f5c-8539-d7724efd201b.png)

### Skip-gram example
- window size = 2

![다운로드 (5)](https://user-images.githubusercontent.com/80622859/189517859-57fc84ec-a4dc-421d-b9a2-637ced132698.png)

![다운로드 (6)](https://user-images.githubusercontent.com/80622859/189517869-7990cfa2-53c9-4ab6-8bd7-7fae790712eb.png)

- 단일 은닉층만 존재하는 얕은 신경망
- N : 은닉층의 크기, V : 단어 집합의 크기, W : 입력층과 은닉층 사이의 가중치 행렬, V x N 크기, input word matrix, $W'$ : 은닉층과 출력층 사이의 가중치 행렬, N x V 크기, output word matrix
- $W\,,W'$는 전치 관계가 아닌 서로 다른 행렬이며 학습 전에는 모두 무작위 값을 가짐
- 학습

![다운로드 (7)](https://user-images.githubusercontent.com/80622859/189517953-de06a0f6-8309-4e87-8293-0357e2f8b362.png)

- 손실 함수 : Cross entropy

### Hierarchical Softmax
- 연산이 너무 비대해지는 것을 막기 위해 고안된 방법
- Full softmax에 근접하면서 연산을 효율적으로 할 수 있는 방법
- Softmax를 통해 확률을 구하고자 할 때 분모로 모두 더하는 방식
- Model 구조 자체를 full binary tree 구조로 바꾼 후에 단어들은 leaf node에 배치

![images_xuio_post_80b8fcca-1f8e-42c3-ac62-91993efa2641_image](https://user-images.githubusercontent.com/80622859/189518139-903eb12c-7295-4c7b-a2cf-48f003337c04.png)

- Leaf node까지 가기 위해 거쳐가는 node들은 vector가 존재하며 이를 학습시키는 것이 목적
- ex) w4라는 단어의 주변 단어가 w2라는 단어일 때 이 둘에 대한 확률 값 계산
- 각 leaf node까지 가는데 만나는 node에 해당하는 vector들을 내적해주고 sigmoid 함수( $\theta$ )를 통해서 확률값으로 만들어준 이후 이들을 곱해나가며 leaf node까지 감
- 결과적으로 sigmoid 함수만 이용해 softmax를 사용하지 않아도 됨

![1](https://user-images.githubusercontent.com/80622859/189518191-c76e61df-0177-4ed5-85a2-436c05faedeb.png)

- Root로부터 내려갈 때 이전 node로부터 좌측으로 이동하면 +1, 우측으로 이동하면 -1
- sigmoid 함수의 특징 : $\sigma (x) + \sigma (-x) = 1$

## Negative Sampling
- 단어에 대한 학습을 보다 효율적으로 하기 위해서 고안된 방식
- Parameter를 update 시킬 negative sample 몇 개를 일부만 뽑아서 그것에 해당하는 parameter를 update
- 기본적으로 영향도가 높은 단어들이 선택 되고, 일부가 무작위로 선택
- 영향도가 높은 sample에 대해서는 positive를 부여, 그렇지 않은 단어에는 negative 부여 => 이진 분류
- 주변부 단어로 선택되는 것은 빈번하게 등장하는 단어들 -> 빈번하게 출현하는 단어들에 대한 학습에 유리
- 전체 corpus 중 문장에 사용되는 비중이 높은 단어를 우선적으로 가중치를 줘서 선별

![images_xuio_post_783bec69-7553-42a7-be9a-7fb19ced9640_image](https://user-images.githubusercontent.com/80622859/189522691-33d29414-fcd6-4fb5-adba-70d52153566e.png)

- 해당 확률은 i번째 단어에 대한 $f(w_i)$ = 단어의 빈도 = 출현횟수/전체 corpus 수
- 보통 위의 식에 3/4 제곱을 취해줌

### Skip-Gram with Negative Sampling,SGNS)
- 중심 단어와 주변 단어가 모두 입력이 되고, 이 두 단어가 실제로 window 크기 내에 존재하는 이웃 관계인지 그 확률을 예측
- 기존의 skip-gram dataset -> SGNS dataset

![그림3](https://user-images.githubusercontent.com/80622859/189522781-8dc491bb-8c26-4cb0-a61c-e71a007e3bf4.png)

- 기존의 skip-gram dataset에서 중심 단어와 주변 단어를 각각 입력1, 입력2로 둠
- 이 둘은 실제로 window size 내에서 이웃 관계였으므로 label = 1
- label = 0인 dataset

![그림4](https://user-images.githubusercontent.com/80622859/189522832-8ead96b9-d08b-4c88-8e27-ef076f2733f8.png)

- 주변 단어 관계가 아닌 단어들을 입력2로 삼기 위해 단어 집합에서 무작위로 선택한 단어들을 입력2로 하고, label은 0으로 지정

![그림5](https://user-images.githubusercontent.com/80622859/189522899-e491911e-c78b-48ad-b88b-3ae56ff5f993.png)

- 입력 1인 중심 단어의 table look up을 위한 embedding table, 다른 하나는 입력 2인 주변 단어의 table look up을 위한 embedding table
- 각 단어는 각 embedding table을 table look up하여 embedding vector로 표현 

![그림7](https://user-images.githubusercontent.com/80622859/189523081-785fd889-da9a-4edc-b67d-f43458800208.png)

- 중심 단어와 주변 단어의 내적값을 이 model의 예측값으로 하고, label과의 오차로부터 역전파 수행
- 학습 후에는 좌측의 embedding matrix를 embedding vector로 사용할 수 있고, 두 행렬을 더한 후 사용하거나 연결하여 사용할 수도 있음
 
## Subsampling of Frequent Words
- 드물게 등장하는 단어와 빈번하게 등장하는 단어의 가중치 측정
- 모든 문장과 문서에서 자주 등장하는 단어는 별로 의미가 없음. ex) a, the, in...
- 더 많이 발생하는 조합에 대해서 상관관계를 만드는데 있어 더 낮은 가중치를 부여
- 빈번하게 등장하는 단어는 학습에 있어서 큰 변화를 주지 못하도록 함

![다운로드](https://user-images.githubusercontent.com/80622859/189523343-27561b24-15ab-4d14-bc2a-bd2e6b999273.png)

- $f(w_i)$ : 단어 $w_i$의 빈도수, t : 정해진 threshold, 일반적으로 $10^{-5}$
- 빈도수가 매우 높은 단어들은 subsampling 과정을 통하여 sampling될 확률이 감소
- 이는 중요한 단어의 representation quality를 개선하는 결과

## Learning Phrases
- 많은 구들은 개별적인 단어들의 결합으로 얻어질 수 없는 의미를 가지고 있음
- 특정 맥락에서만 자주 등장하는 단어쌍을 하나의 token으로
- ex) New York Times -> Unique token in training data <-> this is -> 그대로 남겨둠
- 어휘의 크기를 크게 키우지 않으면서도 많은 reasonable phrases를 얻을 수 있음
- 이론적으로 skip-gram model에서 모든 n-gram에 대하여 학습 가능, but causes out of memory
- Phrases를 text에서 구분하기 위해 data-driven approach 이용. 
- Use score by using Uni-gram and bi-gram

![1](https://user-images.githubusercontent.com/80622859/189523753-a5afb6d4-bf57-4db0-9e33-f072dec1bc53.png)

- δ : hyperparameter, 매우 빈번하지 않은 단어로 구성된 너무 많은 구를 방지, 별로 사용되지 않은 구는 제외
- Threshold를 넘어선 점수를 가진 bi-gram이 선택

- 간단한 vector 간의 연산을 통해 정교한 유추 가능
- 각각 vector들의 element-wise 합을 통해 의미적으로 단어를 결합 = 또 다른 linear structure
- 문장 안의 주변 단어를 예측하는 word vector의 훈려과정에서 단어가 나태는 맥락의 분포를 표현 가능
