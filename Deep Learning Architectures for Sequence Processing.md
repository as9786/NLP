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
- 
