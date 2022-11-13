# 잠재 의미 분석(Latent Semantic Analysis)

- Topic modeling을 위한 최적화된 algorithm X
- Topic modeling이라는 분야에 idea를 제공한 algorithm
- LDA는 LSA의 단점을 보완해서 만들어진 algorithm

- BoW에 기반한 DTM(문서 단어 행렬)이나 TF-IDF는 기본적으로 단어의 빈도 수를 이용한 수치화 방법이기 때문에 단어의 의미를 고려하지 못함
- 즉, 단어의 주제를 고려하지 못함
- 위의 문제점을 해결하기 위해 DTM의 잠재된 의미를 이끌어내는 방법으로 LSA 등장
- LSI라고 부르기도 함. Latent Semantic Indexing
- 특잇값 분해에 기반한 방법

## 특이값 분해(Singular Value Decomposition, SVD)

- A라는 m x n 행렬이 있을 때, 아래와 같이 3개의 행렬의 곱으로 분해하는 것

<img width="98" alt="캡처" src="https://user-images.githubusercontent.com/80622859/201514753-78a2a91a-9fd0-48f9-8913-2e3380281222.PNG">

- U : m x m 직교행렬
- V : n x n 직교행렬
- $\Sigma$ : m x n 대각행렬

### 전치 행렬(Transposed Matrix)
- 원래의 행렬에서 행과 열을 바꾼 행렬

<img width="107" alt="캡처" src="https://user-images.githubusercontent.com/80622859/201514823-f4ce191e-a25f-4e0a-bcc2-5e23c01bda2b.PNG">

### 직교 행렬(Orthogonal matrix)
- n x n 행렬 A에 대해서 $A x A^T = I$를 만족하면서 교환 법칙도 만족하는 행렬
- $A^{-1} = A^T$

### 대각행렬(Diagonal matrix)
- 주대각선을 제외한 곳의 원소가 모두 0인 행렬

<img width="109" alt="1" src="https://user-images.githubusercontent.com/80622859/201514958-dbd74f4a-d32c-4080-a34c-43216104203f.PNG">

<img width="109" alt="2" src="https://user-images.githubusercontent.com/80622859/201514960-d339ee97-87ab-49a8-a205-3ea6d50c05b2.PNG">

<img width="123" alt="3" src="https://user-images.githubusercontent.com/80622859/201514963-186d955c-ccbc-4565-bed8-4f1deaf442f6.png">





