# Glove: Global Vectors for Word Representation

## Abstract 
- 최신 연구들에서 의미와 구문 규칙을 파악하는 것 성공. 그 규칙에 대한 기원은 불분명
- 규칙이 word vector representation에 필요한 모형의 특성을 분석 및 명시
- Global log bilinear regression
- Word-Word co-occurence metrix의 0이 아닌 원소들만 훈련

## Instroduction
- 각 단어는 real number vector로 표현
- 대부분의 vector의 성능을 평가하기 위해 word vector 사이의 거리나 각도를 이용. 
- 각 차원의 차이를 조사하여 word vector space의 미세 구조를 확인
- 대표적인 모형
1. Global matrix factorization methods (LSA)
2. Local context window methods(skip-gram)

- 1번 방법의 경우에는 통계 자료를 효과적으로 사용하지만, 단어 유추 작업을 제대로 수행 X
- 2번 방법의 경우에는 유추 작업에서는 더 낫지만, 말뭉치의 통계 자료를 활용 X(동시 발생 횟수 대신 context window 사용)
- 본 논문에서는 specific weighted least squares 사용

## The GloVe Model

- X : 단어-단어 동시 발생 횟수의 행렬 
- $X_{ij}$ : X의 항목, 단어 i의 context에서 단어 j가 나타나는 횟수
- $X_i = \sum_k X_{ik}$ : 단어 i의 contet 내에서 어떠한 단어라도 나타나는 횟수(i행의 값을 모두 더한 값)
- $P_{ij} = P(j|i) = X_{ij}/X_i$ : 단어 i의 context 내에서 단어 j가 나타날 확률
- 60억 단어의 말뭉치에서 선택한 몇 단어들과 target word인 ice와 steam의 동시발생 확률

![image](https://user-images.githubusercontent.com/80622859/230825200-5db8970a-2ac2-4db9-8d5c-79d2bc641e1b.png)

- 큰 값은 ice의 특성과 연관되고, 작은 값은 steam의 특성과 연관
- solid는 ice와 연관. gas는 steam과 연관. water와 fashion은 각각 두 단어 모두 관계가 있고, 없기 때문에 비율의 값이 1에 가까움
- 이런 비율을 통해 관련 있는 단어와 관련 없는 단어를 구분
- 해당 확률 비율로부터 word vector training
- $F(w_i,w_j,\tilde w_k) = \frac{P_{ik}}{P_{jk}}$
- w : word vector(중심 단어), $\tilde w$ : 주변 단어
- 우변은 말뭉치로부터 구할 수 있음. F는 아직 지정되지 않은 parameter에 의해 결정

### 1. F를 word vector space에 투영
- $\frac{P_{ik}}{P_{jk}}$ encoding
- Vecotr space는 선형 구조를 가짐. Vector의 차이를 이용(F는 두 대상 단어 차이에 의존)

![image](https://user-images.githubusercontent.com/80622859/230825843-27177c04-3f80-4445-b940-93f98622f2c5.png)

### 2. 인자들을 내적
- F의 인자는 vecotr, 우변은 scala

![image](https://user-images.githubusercontent.com/80622859/230825899-9d214f76-c294-4221-a7cb-ac2d73455a4e.png)

- 선형 공간에서 단어의 의미 관계를 표현하기 위해 뺼셈과 내적

### 3. $w <-> \tilde w$
- 위의 값들은 무작위 선택이므로 이 둘의 관계는 자유롭게 교환이 되어야 함
- Ex) ice와 steam, steam과 ice는 같다
- 동시출현 행렬의 전치행렬은 원 행렬과 같다.(대각행렬)
- 위의 조건이 성립하기 위해서는 준동형을 만족해야 함(Homomorphism)
- F(a+b) = F(a)F(b)와 같도록 만족해야 함
- 뺄셈에 대한 준동형식 = 나누기

![image](https://user-images.githubusercontent.com/80622859/230826511-1d153f02-6cf0-4933-ae8b-47a6b57943ef.png)

![image](https://user-images.githubusercontent.com/80622859/230826534-e566adb3-d4b7-4879-b36b-ae9f9cc79fe6.png)

![image](https://user-images.githubusercontent.com/80622859/230826547-e3a08f2e-f189-470e-8a0d-2f86399f2ffc.png)


- 위의 F를 만족하는 함수를 찾아야 함
- 지수 함수(Exponentioal function)
- F를 지수 함수 exp로 가정

![image](https://user-images.githubusercontent.com/80622859/230826668-d8354abf-7305-4ec7-8ece-1a0c178e3468.png)

![image](https://user-images.githubusercontent.com/80622859/230826817-408ef4d0-02d1-4fc8-b009-472fd0a10b70.png)

- log를 사용하게 되면 교환 법칙 성립 X
- 이를 해결하기 위해 편향 추가

![image](https://user-images.githubusercontent.com/80622859/230826882-bef19f16-4640-47f4-9c06-37630315dd83.png)

- 좌변과 우변의 차이를 최소화하는 방향으로 학습

![image](https://user-images.githubusercontent.com/80622859/230826911-6807b656-1db5-4170-8115-2d144a173b84.png)

- V : 단어 집합의 크기
- $log X_{ik}$에서 $X_{ik}$ 값이 0이 될 수 있음
- 가중치 함수 추가
- $f(X_{ik}$ graph : 가중치 함수

![image](https://user-images.githubusercontent.com/80622859/230827006-e3682c8f-5b3c-4ea5-8e3d-f8def4ea640e.png)

- 값이 작을 수록 함수의 값은 작도록 하고 값이 크면 함수의 값이 크도록
- 함수의 최대값은 정해 놓음(it is와 같은 불용어의 동시 등장 빈도수가 높다고 해서 지나친 가중을 받으면 안됨)
![image](https://user-images.githubusercontent.com/80622859/230827100-224a3ecf-ce15-40d0-b9cd-34ea86ffabc3.png)

![image](https://user-images.githubusercontent.com/80622859/230827125-85eb2cbc-8883-4725-afb3-388aa7af8230.png)






