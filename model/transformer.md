# Transformer

- encoder-decoder 알고리즘, attention 기반
- Bert : Transformer의 encoder 부분, GPT : Transformer의 decoder 부분, BART : encoder/decoder 모두

## 1. Attention

- Seq2Seq가 갖는 문제점을 보완하기 위해 제안
- Seq2Seq 문제점 : 입력된 시퀀스 데이터에 대해서 하나의 고정된 벡터 정보(마지막 hidden state)만을 decoder로 전달 => 각 단어들의 정보가 제대로 전달 x

### Attention 추가

- 각 단어의 hidden state 정보(행렬)를 모두 decoder로 전달
- decoder 부분의 주된 목적은 encoder에서 넘어온 정보를 사용하여 그에 해당하는 다른 도메인의 시퀀스 데이터 생성
- Hidden state 정보를 그냥 사용하는 것이 아닌 예측하고자 하는 단어와 관련이 더 많은 단어에 대한 hidden state에 더 많은 주의를 기울임
- encoder에서 넘어온 각 입력 단어들에 대한 hidden state 정보에 가중치를 줌
- 입력된 단어에 서로 다른 가중치를 주는 것 = Attention(Decoder 부분에서 사용)
- decoder에서 예측하고자 하는 단어에 대해 더 많은 가중치를 부여

![캡처](https://user-images.githubusercontent.com/80622859/177313852-b610a396-a6ff-4dd8-8d68-b0f1eae5aa67.PNG)

- SOS : start of sequence
- 'Today' 라는 단어를 예측할 경우, Attention 층에서는 hs에 포함이 되어 있는 벡터 정보 중에서 예측하고자 하는 Today와 관련이 제일 많은 오늘은 단어에 해당하는 hidden states $h_0$에 제일 많은 가중치를 주게 됨

![캡처](https://user-images.githubusercontent.com/80622859/177314125-ad74d62e-140b-40ba-a0b2-2dee94d0e5ef.PNG)

- ho * 0.8 + h1 * 0.1 + h2 * 0.1 이 attention 층에서 출력되는 'Today'를 예측하는 데 사용되는 벡터
- 이러한 벡터가 디코더 RNN 층에서 출력되는 hidden state 벡터와 이어붙이기 연산이 이뤄지고, 이 결과가 softmax 층의 입력값으로 사용

### 가중치 계산

- 가중치는 hs의 각 hidden state와 decoder에 예측하고자 하는 단어에 대한 hidden state와의 유사도를 가지고 계산 -> 내적(코사인 유사도와 비례한 값)

![캡처](https://user-images.githubusercontent.com/80622859/177315615-fad85cb5-e890-41b9-b802-2d441467a8ea.PNG)

- 위의 값들은 attention score라고 함. Attention score의 값이 클수록 관련도가 큼. 가중치를 확률로 표현하기 위해 softmax 함수 사용

## 2. Self-attention

- Decoder에 encoder에서 넘어온 정보에 가중치를 주는 방식 = encoder-decoder attention
- self-attention : 입력된 텍스트 데이터 내에 존재하는 단어들 간에 가중치를 주기 위해 사용. 입력된 단어들 중에서 특정 단어와 관련이 높은 단어들을 찾아서 그 정보를 사용
- 지시대명사의 의미 파악 가능
- ex) 'The dog likes an apple. It has a long tail.'
- it이 지칭하는 바를 파악 가능

- 자기 자신을 포함하여 다른 단어들과의 내적을 통해 유사도 계산(Attention score)->softmax 함수 -> 이 가중치를 각 단어의 embedding vector에 곱하고, 그 결과 벡터들의 합이 self-attention의 결과

- Query : 질의 벡터, Key : 각 단어에 대한 ID 정보, Value : 단어의 고유한 특성 정보

![캡처](https://user-images.githubusercontent.com/80622859/177317493-fdf17086-f43e-48c3-af5d-b44133263b3c.PNG)

- 각 입력 벡터들에 대해서 입력벡터 당 새로운 3개의 새로운 벡터 생성(Query,Key,Value)
- 위의 새로운 벡터를 만들기 위해서는 3개의 가중치 행렬 필요(가중치 행렬은 학습을 통해서 정해지고, 처음에는 랜덤값)

### Attention-score 구하기

- Query 벡터와 각 단어의 Key 벡터와의 유사도를 계산하여 가중치를 구하고, 그 가중치를 각 key에 대한 value에 적용하요 최종 결과 벡터 추출
- ex) 단어 1에 대한 output 구할 경우
1) 단어 1에 대한 query 벡터를 자기 자신을 포함한 모든 단어에 key vector와 내적 후 softmax 연산
2) 위 값을 각 단어의 value 벡터의 곱
3) 결과로 도출된 각 벡터를 원소별로 더함


## 3. Transformer

### 1. 구조

![캡처](https://user-images.githubusercontent.com/80622859/177318532-f230df3c-1109-46ea-b54f-cf03da1196d8.PNG)

### 2. Encoder

![캡처](https://user-images.githubusercontent.com/80622859/177318604-1da67490-1528-4f21-a035-669401d88b05.PNG)

- 위의 하나의 block을 encoder block이라고 함. 이러한 encoder block을 여러 개 사용하여 encoder 구성
1) 아래에서 전달되는 단어들의 정보를 먼저 multi-head attention 층에서 입력을 받고, 
2) 그 결과가 add & norm으로 전달. 
3) 다시 그 결과에 feed-forward neural network 적용
4) 다시 한 번 add & norm 층 적용

#### Multi-head attention

- Self-attention이 여러 개 사용

![캡처](https://user-images.githubusercontent.com/80622859/177319034-b1aea852-c694-4583-b28e-1f046b5e9ed7.PNG)

- Self-attention에서 구하는 attention score와 유사하지만 transformer에서는 내적값을 $\sqrt{d_k}$로 나눠줌 ( $d_k$ 는 원소의 갯수) => Scaled dot product attention
- 각 scaled dot-product self-attention 결과를 이어붙이기해서 multi-head attention 계산
- Q, K, V 값을 그대로 쓰지 않고 선형 변환을 함
- Attention을 사용하는 이유
1) 각 단어에 대해서 주목해야하는 다른 단어가 무엇인지를 더 잘 파악할 수 있다.
2) 각 단어가 가지고 있는 의미적, 위치적 특성을 더 잘 표현할 수 있다

#### ADD & Norm 층

![캡처](https://user-images.githubusercontent.com/80622859/177319998-cd210e31-7ab4-4255-9ee6-421bbcac334b.PNG)

- Multi-head attention이 출력하는 값과 encoder block의 입력값을 더해서 (원소별 덧셈) layer normalization 수행
- encoder block의 입력값
1) 첫번째 encoder block : 각 단어들의 임베딩 정보
2) 그 이후 encoder block : 첫 번째 encoder block에서 나온 벡터값
- Encoder block의 입력값과 출력값을 다시 더해주는 것 = skip connection(identity mapping) => 경사소실문제 해결
- Normalizatin => 학습이 잘 되고, 학습의 결과가 향상

#### Position-wise feed-forward network

- position : 각 단어 의미
- 서로 다른 FNN이 각 단어에 대한 결과에 독립적으로 적용, transformer의 경우에는 각 단어마다 두 개의 position-wise FNN 사용
- 첫 번째 FNN -> ReLU, 두 번째 FNN -> 활성화 함수 없음

![캡처](https://user-images.githubusercontent.com/80622859/177322437-f3fad46e-4636-40c2-a364-5e4720be2c99.PNG)

![캡처](https://user-images.githubusercontent.com/80622859/185107341-94e542c7-9dce-4348-8c94-aad99897fe87.PNG)


- x : 첫 번째 FNN에 입력되는 입력벡터(Add & Norm 층이 출력하는 각 단어에 대한 결과 벡터)
- 위의 식을 통해서 나온 값들에 다시 한 번 Add & Norm 층을 적용하여서 encoder block의 최종 결과물 출력 = encoder block의 hidden state

#### 위치정보 임베딩(Positional embedding)

- Transformer는 단어들의 embedding 정보뿐만 아니라 입력된 시퀀스 데이터 내에서의 위치 정보도 사용 => 단어들 간의 상대적인 거리 파악 가능, 순환신경망 구조를 사용하지 않음(attention 사용)
- 첫 번째 embedding block에 입력된는 각 단어들의 embedding 정보는 각 단어의 원래의 embedding 정보와 위치기반 embedding 정보의 합
- Positional embedding <- 삼각함수(sin, cos), 출력값 [-1,1]

![캡처](https://user-images.githubusercontent.com/80622859/177323165-b2b43649-58c4-42d1-a8a4-1e6ade880be1.PNG)

- PE_{(i,j)} : 단어 i의 positional embedding vector의 위치 j의 원소값

### 3. Decoder

- 6개의 deocder block으로 구성
- encoder block과 유사, 가장 큰 차이점은 서로 다른 두 종류의 attention이 사용
- Masked multi-head attention(self-attention), Multi-head attention(encoder-decoder attention)

1. 단어의 임베딩 정보와 positional embedding 정보가 합산되어 첫 번째 decoder block으로 입력
2. 각 decoder block에 입력된 정보는 먼저 self-attention을 거치고, 그 결과에 Add & Norm 층이 적용
3. encoder-decoder attention 
4. 최종 결과물은 softmax 활성화함수로 갖는 출력층의 입력값으로 전달
5. 해당 출력층에 존재하는 출력노드는 다음 단어에 대한 확률을 출력

- encoder-decoder attention : encoder와 유사하게 작동 다만 encoder 부분에 입력된 단어들의 hidden state 정보나 decoder 부분에 입력된 단어들의 임베딩 정보를 직접적으로 사용하지 않음
- query : decoder 부분에 입력된 단어, key, value : encoder 부분에서 각 단어에 대한 값

- Transformer의 decoder는 teacher forcing이라는 방법을 사용
- 정답 데이터 정보를 이용해서 각 단계의 단어들을 예측
- 자신보다 뒤에 나오는 값을 -infinite로 하여서 softmax를 적용하였을 때 확률값을 0으로 만듦 = masking
