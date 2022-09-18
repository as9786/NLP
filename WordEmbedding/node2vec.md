# Scalable feature learning for networks

## Abstract
- 현재의 기능 학습 접근 방식은 network에서 관찰된 연결 pattern의 다양성을 포착하기에는 충분히 표현적이지 않음
- Network의 node들의 연속적인 특징 표현을 위한 framework = node2vec
- Node의 network 이웃을 보존할 가능성을 최대화하는 기능의 저차원 공간에 대한 node의 mapping을 학습
- Node의 network 이웃에 대한 유연한 개념을 정의하고 다양한 이웃을 효율적으로 탐색하는 편향된 무작위 보행 절차 설계


## Introduction
- Network 분석에서 많은 중요한 작업은 node와 edge에 대한 예측을 포함
- Network에서 가장 가능성이 높은 nodel label을 예측하는데 관심
- 한 쌍의 node가 그들을 연결하는 edge를 가져야 하는지 여부를 예측 ex) Social network에서 실제 친구 식별
- Node와 edge에 대한 feature vector를 구성해야 함

### Homophily & Structural Equivalence

![image](https://user-images.githubusercontent.com/80622859/189602654-38e4d9e7-6ca6-4343-92a2-a778a01047d3.png)

- Homophily(동질성) graph : 유사한 node가 가까운 거리에 위치, u가 $s_1\,, s_2\,, s_3\.. s_4$와 유사
- Structural equivalence : 가까운 거리에 연결되어 있다고 해도 node의 성질이 비슷하다고 할 수 없음. 거리가 멀더라도 구조적으로 비슷한 역할을 하는 node가 유사한 노드. u와 $s_6$

- Real-world networks commonly exhibit a mixture of such equivalences
- 따라서 동일한 network community에서 node를 내장하는 표현을 학습할 수 있을뿐만 아니라 유사한 역할을 공유하는 node가 유사한 embedding을 갖는 표현을 학습할 수 있는 두 가지 원칙을 준수하는 유연한 algorithm을 허용해야 함

### present work
- SGD를 사용하여 사용자 지정 graph 기반 목적 함수를 최적화
- d차원 feature space에서 node의 network 이웃을 보존할 가능성을 최대화하는 feature representation을 반환
- Use a 2nd order random wakl approach to generate network neighborhoods for nodes
- Key : defining a flexible notion of a node's network neighborhood
- node2vec은 적절한 이웃 개념을 선택함으로써 network의 역할 및 속한 community를 기반으로 node를 구성하는 표현을 학습
- 이전 작업을 일반화하고 network에서 관찰된 동등성의 전체 spectrum modeling
- 개별 node의 feature representation이 pairs of noded(edge)로 확장 -> Node뿐만 아니라 edge까지 포함하는 예측 작업
- 두 가지 예측 작업
1. 모든 node에 하나 이상의 class label이 할당되는 multi-label classification
2. Node 쌍이 주어진 edge의 존재를 예측하는 link prediction task

## Related Work
- Graph의 다양한 행렬표현
- Laplacian and the adjacency matrices

### Laplacian
- 일차 미분

![img](https://user-images.githubusercontent.com/80622859/189599856-a84edf1a-0076-4afe-8c7a-3d0ba4d1612c.png)

- 2차 편미분

![다운로드](https://user-images.githubusercontent.com/80622859/189599900-d88228fd-afb9-4278-b6bd-1b52e22f418b.png)

- 2차 미분

![img (1)](https://user-images.githubusercontent.com/80622859/189599935-c3f7da1d-de5b-46fa-8122-f7f3a241ddba.png)

- 이차 미분을 나타내는 연산자는 $\nabla^2$이며 라플라시안 또는 델타 스퀘어라고 읽음
- Laplacian Operator

![다운로드 (1)](https://user-images.githubusercontent.com/80622859/189600328-cfd0ef68-115c-493b-a1a5-0e5aa6ec6305.png)

- 선형(PCA) 및 비선형(IsoMap) 차원 축소 기법은 계산 효율성 측면에서 안 좋고, 다양한 pattern에 강하지 않음
- Skip-gram model
- 문서의 단어를 scan하고 모든 단어들을 포함시켜 단어의 특징들이 가까운 단어들을 예측할 수 있도록 하는 것이 목표
- 단어 특징 표현은 negative sampling과 함께 SGD를 사용하여 우도 목표를 최적화
- Skip-gram model의 영감을 받아 network를 문서로 표현함으로써 network에 대한 유추 확립
- 문서가 단어의 집합인 것과 같은 방식으로, 기본 network에서 node 순서를 sampling하고 network를 순서 있는 node 순서로 바꿀 수 있음


## Feature Learning Framework

### Classic search strategies

#### 1. 깊이 우선 탐색(DFS, Depth-First Search)
- 최대한 깊이 내려간 뒤, 더 이상 깊이 갈 곳이 없을 경우 옆으로 이동

![img](https://user-images.githubusercontent.com/80622859/189601502-28c64761-f4bb-4fc0-86cf-f339469db6f6.gif)

- Root node(혹은 다른 임의의 노드)에서 시작해서 다음 분기(branch)로 넘어가기 전에 해당 분기를 완벽하게 탐색하는 방식
- 모든 node를 방문하고자 하는 경우
- BFS보다 간단
- 속도는 BFS보다 느림
- Stack or 재귀함수

#### 2. 너비 우선 탐색(BFS, Breadth-First Search)
- 최대한 넓게 이동한 다음, 더 이상 갈 수 없을 때 아래로 이동

![img (1)](https://user-images.githubusercontent.com/80622859/189601784-87ae4e10-ea47-4a8f-93d6-4141352338c7.gif)

- Root node(혹은 다른 임의의 node)에서 시작해서 인접한 node를 먼저 탐색하는 방법, 시작 정점으로부터 가까운 정점을 먼저 방문하고 멀리 떨어져 있는 정점을 나중에 방문하는 순회 방법
- 최단 경로
- Queue

- BFS, DFS 시간복잡도 : 인접 리스트 : O(N+E), 인접 행렬 : $O(N^2)$ (N : node, E : 간선)

### node2vec
1. 어떤 기준으로 node embedding 간의 유사성을 정의하는가(dot product)
2. 어떤 방식으로 특정 node의 이웃을 정의하는가(parameterized random walk)

#### Embedding Learning
- V : 모든 node들의 집합. f : Embedding function, 각각의 node를 d차원의 고정 길이 vector로 mapping, $\mathbb{R}^{|V|\times d}$ 형태의 parameter. $\mathcal{N}_s (u)$ : 특정 node u의 이웃 node 
- Embedding을 학습하는 기본적인 접근

![캡처](https://user-images.githubusercontent.com/80622859/189604149-1d24d9e7-b26b-4d36-a7f1-2aa2581ee44f.PNG)

- 어떤 node u의 embedding(encoding된 표현)이 주어졌을 때, 이것을 decoding(reconstruct)한 결과가 u의 이웃 node일 확률을 최대화시키는 f를 찾음 = Embedding 함수에 대한 MLE
- $P_\gamma (\mathcal{N}_s (u)|f(u))$는 아래와 같이 개별 확률의 곱으로 표현(이웃 node들간의 독립 가정)

![캡처](https://user-images.githubusercontent.com/80622859/189604506-b882053a-d1c0-42a5-8f3e-98c5b3772550.PNG)

- $P_\gamma (n_i|f(u))$을 dot product와 softmax로 아래와 같이 정의

![render](https://user-images.githubusercontent.com/80622859/189605002-4184dbe3-9c8d-4947-9a7a-18eecc348a57.png)

- 분모 부분을 계산하는데 필요한 계산 비용이 매우 큼 -> 모든 node에 대해 분모 항을 계산하지 않고 일정 개수의 netative node를 sampling 하는 방식으로 분모 항을 근사
- 최종적인 optimizer objective

![캡처](https://user-images.githubusercontent.com/80622859/189605226-459d2bc6-8ccf-48f9-9234-60fa88fdc956.PNG)

#### Defining Neighbor via Biased Random Walk

![캡처](https://user-images.githubusercontent.com/80622859/189605691-e177c6f2-ac25-4b87-9c79-0776ec6b055b.PNG)

- Node t에서 random wakl를 1회 수행하여 v로 이동하였다고 할 때, 1/p에 비례하는 확률로 이전 node로 돌아가고 1/q에 비례하는 확률로 아예 새로운 node를 탐색하며 1에 비례하는 확률로 이전 노드 t와의 거리가 1인 $x_1$로 이동

![캡처](https://user-images.githubusercontent.com/80622859/189605908-0d37b350-09b4-4563-81d5-e7a2a99be9ce.PNG)

- $\alpha_{pq} (t,x)$ : Biased random walk를 위한 정규화되지 않은 확률(unnormalized probability)
- $d_{tx}$ : 이전 node t와 이동 후의 node x간의 거리
- p와 q는 hyperparameter
- p : Return parameter, 높으면 위치적으로 거리가 멀더라도 이웃 node로 정의, 작으면 local node(위치적으로 가까운 node)를 이웃 node로 정의
- q : In-Out parameter, q > 1 -> 이전 node인 t와 가까운 node를 이웃 node로 정의, q < 1 -> 이전 node t로부터 거리가 먼 node를 이웃 node로 정의
- p를 크게 하고, q를 작게 정의하면 멀리 있는 node도 이웃 node로 정의(structural equivalence)
- p를 작게 하고, q를 크게 하면 가까이 있는 node만 이웃 node(homophily)

### Learning edge feature
- Random wakl는 기본적으로 network의 node 간 연결 구조를 기반으로 하기 때문에 개별 node의 feature representation을 통해 bootstrap 접근 방식을 사용하여 node 쌍으로 확장
- 두 node u와 v가 주어지면, g : V x V -> $mathbb{R}^{d'}$ where d' is a representation g(u,v)
- 연산이 간선이 존재하지 않더라도 모든 쌍의 node들에 대해서 정의되기를 원함 -> Link prediction에 도움이 됨

## Discussion and Conclusion
- Exploration-exploitation trade-off를 기반으로 고전적인 검색 전략을 설정 가능
- 예측 작업에 적용될 때 학습된 표현에 대한 해석 가능성을 어느 정도 제공
- 매개변수 p와 q를 통해 network 이웃을 탐색하는데 유연하고 제어 가능
- 실용적인 관점에서 확장 가능
