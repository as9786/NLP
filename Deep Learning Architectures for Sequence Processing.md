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

## 5.2 Bidirectional RNNs
- 기본적으로 RNN은 왼쪽 정보 이용
- 오른쪽에서도 정보를 이용하는 방안으로 왼쪽->오른쪽, 오른쪽->왼쪽의 흐름을 학습하도록 구성

![image](https://user-images.githubusercontent.com/80622859/202715661-83adbaf3-f864-481b-9a4c-aceb8060b18c.png)


- $h^f_t$ : 단순히 t 시간의 은닉층에 대한 정보를 나타내며, 지금까지 수집한 모든 것을 나타냄

![image](https://user-images.githubusercontent.com/80622859/202715787-f04a4287-5b4d-422f-b91d-50f8c8e5a51d.png)

- 역으로 입력을 받음
- 위 두 개의 정보를 vector 연결

![image](https://user-images.githubusercontent.com/80622859/202715886-cc3b4859-403b-4ce8-8557-c4bd21c721d7.png)

- 양방햔 순환신경망은 분류에서 상당히 효과가 좋다고 알려져 있음

![image](https://user-images.githubusercontent.com/80622859/202716045-c34fd095-ec0b-4673-bc95-44abecdcbbb4.png)


# 6. LSTM
- 거리가 멀리 있는 단어에 대한 정보를 잘 기억해야함
- Vanilla RNN은 은닉층과 더 나아가 은닉층의 값을 결정하는 가중치가 현재 결정에 유용한 정보를 제공하고 미래에 필요한 정보를 최신화하고 전달하는 두 가지 작업을 동시에 수행하도록 요청 받음
- 반복적인 곱셈 과정으로 가중치 값이 0에 가까워짐
- 위의 두 가지 이유로 거리가 먼 단어에 대한 학습이 잘 안될 수 있음
- 이러한 문제를 해결하기 위해 더 복잡한 모형들은 신경망이 더 이상 필요하지 않은 정보를 잊어버리고 앞으로 있을 의사결정에 필요한 정보를 배울 수 있게 함으로써 시간이 지남에 따라 관련 맥락을 유지하는 작업을 관리하도록 설계
- 대표적인 모형 : LSTM
- LSTM의 gate는 Feed forward layer, sigmoid, gate layer와 함께 요소별 곱셈으로 구성
- 활성화 함수로 sigmoid를 쓰는 이유는 0과 1로 값을 출력하고 이는 요소별 곱을 수행할 시 masking 효과를 줄 수 있음
- Mask에서 1에 가까운 값과 일치하는 gate layer의 값은 거의 변경되지 않고 통과 됨
- 더 낮은 값들은 지워짐
- forget gate : 더 이상 필요 없는 context에서 정보를 삭제

![image](https://user-images.githubusercontent.com/80622859/202716901-55942ea0-d86d-4baa-b17c-fe81ebd05e9e.png)

![image](https://user-images.githubusercontent.com/80622859/202716917-b81c76af-6069-471c-bf13-0521ac8bb92f.png)

![image](https://user-images.githubusercontent.com/80622859/202716931-db4fcc17-3b6e-43e7-92c3-7085c075d299.png)

![image](https://user-images.githubusercontent.com/80622859/202716941-55639751-a725-4dcf-b072-2511f05906bb.png)

![image](https://user-images.githubusercontent.com/80622859/202716956-34341860-9bb8-4b32-9657-53ae20293e07.png)

- Output gate : 현재 은닉층에서 필요한 정보를 결정하는데 사용

![image](https://user-images.githubusercontent.com/80622859/202717013-8f055863-c069-429d-b45a-36928fa0f058.png)


## 6.1 Gated Units, Layers and Networks

![image](https://user-images.githubusercontent.com/80622859/202717111-9e16b102-9b41-4da4-9196-9c6da820e7d8.png)

- LSTM unit의 증가된 복잡성은 unit 자체 내에 캡슐화

# 7. Self-Attention Networks: Transformers
- Gate를 활용하면 더 먼 정보를 처리할 수 있지만 기본 문제를 해결하지 못함.
- 순차적인 특성은 병렬 처리를 힘들게 함
- Transformer는 input vector의 sequence를 동일한 길이의 output vector의 sequence에 mapping
- self-attention layer, feed forward network의 결합이 된 다층 transformer block의 stack으로 구성
- self-attention은 신경망이 순환신경망과 같이 중간 반복 연결을 통해 정보를 전달할 필요 없이 임의의 큰 context에서 직접 정보를 추출하고 사용할 수 있게 함

![image](https://user-images.githubusercontent.com/80622859/202717651-ffbfa86b-d594-4f0b-82ac-1740499fc8ba.png)

 - 입력의 각 항목을 처리할 때 입력 정보 이외에는 접근 불가
 - 이러한 접근 방식은 언어 모형을 생성하고, 생성 모형을 사용할 수 있음을 보장
 - 전진 추론과 훈련을 모두 쉽게 병렬화 가능

![image](https://user-images.githubusercontent.com/80622859/202717851-3aa2b89c-ca46-4054-ab0d-9c1181bd22a0.png)

- y3의 계산은 입력 x3와 이전 요소들 그리고 자기 자신을 비교 세트로 계산
- 내적의 결과는 양의 무한대에서 음의 무한대, 값이 클수록 vector 간 유사
- 점수를 효과적으로 사용하기 위해 softmax 연산 -> 가중치 vecotr $\alpha$ 생성

![image](https://user-images.githubusercontent.com/80622859/202718120-00f80a9b-a3f0-448b-a76b-8c057216ca5b.png)

- 지금까지 본 입력의 합계를 각각의 $alpha$ 값으로 가중치를 부여하여 출력값 $y_i$ 생성
- Q : 다른 이전 쿼리 입력과 비교할 때 현재 attention의 초점으로 사용
- K : 현재의 attention 초점과 비교되는 선행 입력으로서의 역할
- V : 현재 attention의 초점에 대한 출력을 계산하는 데 사용

![image](https://user-images.githubusercontent.com/80622859/202718450-0f6f839d-fcb6-471d-be3a-af435b01acf0.png)

- Transformer의 입력 x와 출력 y는 모두 동일한 차원을 가짐. (1 x d)
- 모든 가중치 행렬들은 dxd 차원
- 논문에서 d = 1024
- 큰 값이 나왔을 경우 지수화하면 경사 손실이 발생할 수 있음
- 이를 방지하기 위해 아래처럼 식을 수정

![image](https://user-images.githubusercontent.com/80622859/202718732-03e2be58-b0ef-47bd-96e4-2c92595ba3f7.png)

![image](https://user-images.githubusercontent.com/80622859/202718741-cc16b96e-002c-40e4-8b38-d30ef342b3d2.png)

![image](https://user-images.githubusercontent.com/80622859/202718761-9109e577-c6f2-4e76-8c7e-9732ea88146a.png)

![image](https://user-images.githubusercontent.com/80622859/202718786-9640ed58-2054-44ca-9486-35dd30ac6ff8.png)

- 언어 모형의 구조를 올바르게 짜기 위해 아래와 같이 masking 처리

![image](https://user-images.githubusercontent.com/80622859/202718869-c08e86a8-d07a-4a92-867d-c8434c0fac9c.png)

## 7.1 Transformer Blocks
- residual layer에서 layer normalization 진행 : z-점수의 변형
- Residual layer : 중간 과정을 거치지 않고 하위 계층에서 상위 계층으로 정보 전달 => 학습 성능 개선

![image](https://user-images.githubusercontent.com/80622859/202719039-59c01667-bc4f-4065-bfc9-8935203d20fc.png)

![image](https://user-images.githubusercontent.com/80622859/202719207-b704dbb3-0305-4909-b7b9-dc4e481e34f4.png)

![image](https://user-images.githubusercontent.com/80622859/202719218-d76f2a5a-1941-4a82-b748-2f938aaab945.png)

![image](https://user-images.githubusercontent.com/80622859/202719233-37667ead-2460-47d4-8a48-03d6ffd5793f.png)

## 7.2 Multihead Attention
- 동이한 깊이의 Multi head attention을 사용함으로써 병렬 처리
- 각 head는 동일한 추상화 수준에서 입력 사이에 존재하는 관계의 다른 측면을 학습할 수 있음
- 각 head는 고유한 가중치 행렬들을 갖게 됨

![image](https://user-images.githubusercontent.com/80622859/202719454-770a2655-9584-4bfa-9a4e-44ed69d55500.png)


## 7.3 Modeling word order: positional embeddings
- 각 token의 위치 정보를 알려주기 위해 사용

![image](https://user-images.githubusercontent.com/80622859/202719554-a11f77b3-4d1a-4f40-85d5-ab66857ce5d0.png)

![image](https://user-images.githubusercontent.com/80622859/202719574-0b295b9d-6b70-4565-bf83-46a6cd819107.png)

# 8. Transformers as Language Models

![image](https://user-images.githubusercontent.com/80622859/202719642-1f97d32e-d6c3-4689-9454-a10f4c3010e0.png)

- 새로운 text를 자동 회귀적으로 생성 가능 

# 9. Contextual Generation and Summarization

![image](https://user-images.githubusercontent.com/80622859/202719761-907a00e3-556b-4c9b-a852-9eabd0bffaf8.png)

![image](https://user-images.githubusercontent.com/80622859/202719821-ed12f5e4-5e06-4c24-8ca9-7b1cee4bebef.png)





