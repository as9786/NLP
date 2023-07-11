# Big Bird : Transformers for Longer Sequences

## Graph Sparcification

- Self-attention -> Fully-connected graph : Self-attention을 각 token들의 linking으로 본다면 fully-connected graph로 표현 가능
- Fully-connected graph -> Sparse random graph : Self-attention graph를 훨씬 더 크게, 그리고 sparse하게 만들면 더 긴 sequence를 처리하면서 성능 유지 가능

## Introduction 

- Self-attention mechanism은 병렬 계산이 가능하여 연산량의 강점. 성능적으로도 좋음
- RNN model에서 문제가 되는 sequential dependency 해결

- Transformer model은 연산량은 길이의 제곱에 비례. 입력 길이에 제한을 둠
- 보통 512 token. 대부분 문헌의 특성상 자체의 크기는 매우 크지만, context의 길이 분포가 exponential 하기 때문에 큰 문제 발생 X
- 하지만 512 token보다 훨씬 긴 sequence에 대해서는 효과적이지 못함
- 이를 해결하기 위해 여러 연구 진행

1. Sliding window
- Sequence 길이가 제한된 모형을 여러 번 사용
- 문서를 쪼개서 모형에 여러 번 입력하여 출력값을 이어 붙이거나, 긴 문서에서 관련 있는 부분만 찾는데 사용

2. Full attention 탈피
- Masking의 길이를 늘리거나 global token 수를 늘림

- Big Bird는 graph sparsification에 착안하여 full self-attention을 sparse하게 만듦
- 결과적으로 기존 모형보다 8배 더 긴 입력 길이를 다룰 수 있음(4096 token)
- Random attention, window attention, global attention으로 구성된 sparse attention mechanism 적용

## Big Bird Architecture

- Transformer based
- MHSA와 FFN으로 구성된 층을 여러 겁 쌓아서 만든 구조. Self-attention layer에서 full-attention이 아닌 sparse attention으로 연산

### Generalized attention mechansim

- d 차원으로 embedding 된 n개의 input sequence $X = (x_1,\cdot\cdot\cdot,x_n) \in R^{nxd}$
- Node는 모든 token $[n]=1,\cdot\cdot\cdot,n$, edge가 attention mechanism이 수행되는 sef of inner-product, directed graph D

![image](https://github.com/as9786/NLP/assets/80622859/b8f49abb-1e90-4237-97e1-69f3fd482432)

- i 번째 token에 대한 generalized attention mechanism output vector

![image](https://github.com/as9786/NLP/assets/80622859/6301829f-fb08-4229-ba0b-ec9f0dc5b74a)

- $N_i$ : Set of out-neighbors of node i(out-neighbor : Directed graph에서 node i가 가리키는 node들)
- $Q_h, K_h(R^d \rightarrow R^m)$ : Query & key function, token embedding $x_i \in R^d$를 $R^m$ 차원으로 사상
- $\sigma$ : Scoring function(softmax, hardmax)
- H : Num of head
- $X_{N_{(i)}}$ : 모든 input token이 아닌 ${x_j : j \in N_{(i)}}$에 해당하는 token embedding만 쌓아서 만든 행렬
- 어떤 인접행렬이 0과 1로 구성되어 있고, query i가 key j에 연결되면 1, 아니면 0을 의미한다고 할 때, BERT와 같은 full self-attention을 인접행렬로 표현하면 전부 1로 채워진 인접행렬이 됨
- 이와 같이 self-attention을 fully-connected graph로 나타낼 수 있으니 graph sparsification 이론을 적용해서 복잡도를 낮출 수 있음
- Random graph를 확장시켜 complete graph의 여러 특징들을 근사 가능

#### Small average path length between nodes

- Complete graph에서 edge들이 특정 확률에 의해 남겨진 random graph에서 두 node의 최단 경로는 node의 개수에 logarithmic하게 비례
- Random graph의 크기를 키우고 sparse하게 만들어도 shortest path가 기하급수적으로 증가 X

#### Notion of locality
- 어떤 token의 정보는 대부분 주변 token의 정보에서 얻어지고, 멀리 떨어진 token에서 얻게 되는 정보량은 적음
- 하지만 global token의 존재가 중요
- BigBird-ITC(Internal Transformer Construction) : 언어 말뭉치에 존재하는 token 중 특정 몇 개를 global token으로 지정하여 모든 token들에 대해 attention 계산
- BigBird-ETC(Extended Transformer Construction) : Sequence에 g개의 global token을 추가

