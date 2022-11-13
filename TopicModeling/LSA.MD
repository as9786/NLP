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

- SVD를 통해 나오 대각 행렬은 내림차순으로 정렬 됨
- 해당 원소의 값을 A의 특이값(singular value)라고 함

<img width="126" alt="캡처" src="https://user-images.githubusercontent.com/80622859/201514993-66b3d2dd-6cb4-419f-a61f-0211b8e31e42.PNG">

# 절단된 SVD(Truncated SVD)
- Full SVD에서 나온 3개의 행렬에서 일부 vector들을 삭제시킨 것

<img width="315" alt="캡처" src="https://user-images.githubusercontent.com/80622859/201515033-4e353ccf-f78f-459a-818c-0a76a13728a5.PNG">

- 대각 행렬의 특이값 중에서 상위 t개만 남김
- U행렬과 V 행렬의 t열까지만 남김
- t는 우리가 찾고자 하는 주제의 수(초매개변수)
- t를 크게 잡으면 기존의 행렬 A로부터 다양한 의미를 가져갈 수 있음
- t를 작게 잡아야 noise 제거(설명력이 낮은 정보 제거)
