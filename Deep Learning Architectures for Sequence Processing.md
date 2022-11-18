- 언어는 일시적, 시간적 특성을 갖고 있음
- N-gram 및 FNN은 제한된 context, 구문에서 단어의 의미가 함께 결합되는 방식을 학습하기 어려움

# 1. Language Models Revisited
- 앞의 문맥이 'Thanks for all'이고, 다음 단어가 fish일 가능성을 확인할려면 P(fish|Thanks for all)을 계산
- 언어 모형은 가능한 모든 단어에 위와 같은 조건부 확률을 할당하여 전체 어휘에 대한 분포를 제공
- 조건부 확률을 연쇄 법칙과 함께 사용하여 전체 sequence에 확률을 할당할 수 있음

![image](https://user-images.githubusercontent.com/80622859/202709474-bef8d049-bd22-410e-a80c-498cd25733de.png)

# 2. Recurrent Neural Networks
- 신경망 연결 내에 주기를 포함하는 망.
- 어떤 단위의 값이 입력으로서 자체의 초기 출력에 직접적으로 또는 간접적으로 의존

![image](https://user-images.githubusercontent.com/80622859/202709688-3d4f9ced-fdeb-421b-8696-f4bf37fcf9b0.png)

- 일반적인 FNN과 마찬가지로 현재 입력인 $x_t$를 나타내는 입력값에 가중치 행렬을 곱한 다음 비선형 활성화 함수를 통과하여 은닉층의 값을 계산
- 은닉층은 출력값을 계산하는데 사용
- $x_t$ : 시간 t에서 입력값 x
- 이전 시점의 은닉층 값을 은닉층에 추가하여 계산에 대한 입력 증가
- Context에 고정 길이 제한을 부과하지 않음
- 모든 연결은 역전파를 통해 계산

![image](https://user-images.githubusercontent.com/80622859/202710201-b9df30f6-5988-429b-846a-f670c13b6b8c.png)

![image](https://user-images.githubusercontent.com/80622859/202710559-f07763be-faf1-46f0-9bd0-3fa5d198201e.png)

## 1. Training
- 역전파
- 2-pass algorithm
- 1-pass : $h_t, y_t$를 계산하여 각 단계에서 손실을 누적하고 각 단계에서 은닉층의 값을 저장하여 다음 시간 단계에서 사용
- 2-pass : 순서를 역으로 처리하고, 진행하면서 필요한 기울기를 계산하고, 각 단계의 은닉층에서 사용하기 위해 역전파 수식을 진행
- 이와 같은 과정을 시간을 통한 역전파 라고 함

![image](https://user-images.githubusercontent.com/80622859/202711034-5d950c01-c8c4-4fd7-974c-3c2308b7d105.png)

# 3. RNN as Language Models
- 입력 시퀀스를 한 번에 한 단어씩 처리하여 현재 단어와 이전 은닉층에서 다음 단어를 예측하려고 시도

![image](https://user-images.githubusercontent.com/80622859/202711271-5bd79f98-676e-4354-8c2b-978b39ad032a.png)

- E : Embedding vector
- 손실함수 : Cross Entropy
- 학습 시에는 다음 단어에 대한 정답 정보를 알려줌
- ex) 다음 단어를 규찬이라고 예측하였지만 실제 답이 성주일 경우 규찬이 아니라 성주를 사용
- 위와 같은 방법을 교사 강요라고 부름

![image](https://user-images.githubusercontent.com/80622859/202711628-f468cb86-6f94-456e-bd21-e3fc97dd7325.png)

# 4. RNNs for other NLP tasks

## 4.1 Sequence Labeling

![image](https://user-images.githubusercontent.com/80622859/202712920-079e2858-ed62-4398-ac2e-272fc5b78b57.png)

## 4.2 RNNs for Sequence Classification
- Token이 아닌 전체 sequence를 분류하는 것
- Text의 마지막 token인 $h_n$에 대한 은닉층을 사용하여 전체 sequence의 압축된 표현을 구성할 수 있음

![image](https://user-images.githubusercontent.com/80622859/202713092-1bded85e-f5f6-4e56-b70c-e40d05e372ca.png)

- 마지막 요소 앞의 경우에는 중간 출력이 필요하지 않음
- 손실함수는 최종 text 분류 작업에만 사용
- Downstream application의 손실을 사용하여 신경망을 통해 전체 가중치를 조정하는 훈련을 end-to-end training이라고 함
- 모든 은닉층의 출력값을 평균내는 것도 하나의 방법

![image](https://user-images.githubusercontent.com/80622859/202713327-e6a5c958-cda8-40b1-b9c5-7926c80e40ff.png)


## 4.3 Generation with RNN-Based Language Models
- 문장 마커의 시작인 <s>를 첫 번째 입력으로 사용한 결과로 발생하는 소프트맥스 분포의 출력에서 단어를 샘플링합니다.
- 첫 번째 단어에 대한 임베딩이라는 단어를 다음 시간 단계에서 네트워크에 대한 입력으로 사용한 다음 같은 방식으로 다음 단어를 샘플링합니다. 
- 문장 끝 마커인 </s>가 샘플링되거나 고정 길이 제한에 도달할 때까지 생성을 계속합니다.
- 자기 회귀 모형 : t-1,t-2 등에서 이전 값의 선형 함수를 기반으로 시간 t에서 값을 예측하는 모형

![image](https://user-images.githubusercontent.com/80622859/202714160-d9d82221-97b5-41ef-a60e-84245d51b95e.png)

# 5. Stacked and Bidirectional RNN architectures

## 5.1 Stacked RNNs

![image](https://user-images.githubusercontent.com/80622859/202714274-08f5a10f-86c8-452c-8cf9-bec976c373b2.png)

- 한 계층의 출력이 후속 계층의 입력값으로 들어감
- 일반적으로 단일 계층 신경망보다 성능이 뛰어남
- 신경망 계층에 거쳐 서로 다른 수준의 추상화로 표현을 유도하기 때문
- Stack의 수가 증가할수록 cost가 빠르게 증가


