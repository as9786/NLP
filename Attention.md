# Attention

![image](https://user-images.githubusercontent.com/80622859/222952173-4a8e5feb-7b1d-41d7-8e7d-76a79feaa385.png)

- Decoder 단에서 어떤 encoder 정보에 집중해야 하는지 확인

## Query, Key-Value

- Query : 질의. 찾고자 하는 대상
- Key : 저장된 data를 찾고자 할 때 참조하는 값
- Value : 값. 저장되는 data
- Dictionary : Key-value pair로 이루어진 집합

### Querying

![image](https://user-images.githubusercontent.com/80622859/222952310-8fc2a9ad-8adc-4c94-8c1f-37fb9c3d32cc.png)

## Attention mechanism

![image](https://user-images.githubusercontent.com/80622859/222958903-98a6ab7b-b9fa-47f4-8216-4511fadf41b4.png)

- Q에 대해 어떤 K가 유사한지 비교하고, 유사도를 반영하여 V들을 합성한 것 = Attention value
- Compare 함수로는 dot-product(inner product)가 많이 쓰이며, aggregation은 weight sum을 많이 사용


## Seq2seq - Key-Value

- 대부분의 attention network에서는 key와 value를 같은 값을 사용
- Seq2seq에서는 encoder의 은닉층들을 key와 value로 사용
- Decoder의 은닉층들을 query로 사용
- Encoder와 달리 하나 앞선 time-step의 은닉층을 사용
- RNN으로 은닉 상태를 입력하기 전에, attention value를 이어 붙여서 입력

### Seq2seq - Attention mechanism

![image](https://user-images.githubusercontent.com/80622859/222959072-cb15a5e2-f323-4e73-a1e9-d4680897fac3.png)

![image](https://user-images.githubusercontent.com/80622859/222959094-18b9dd01-f942-4b34-8f6f-45cafe2d38c6.png)

- 은닉 상태에 attention value를 이어 붙이기까지 하면 끝

## Attention의 구현

- Input/Output tensor에 대한 이해가 중요

### Encoder 입출력

![image](https://user-images.githubusercontent.com/80622859/222959822-553ce3b7-53d0-4763-ac09-a71b80c99321.png)

### Deocder 입출력(학습 단계)

![image](https://user-images.githubusercontent.com/80622859/222959859-9c7ec045-b0e8-4002-aa07-ad892e5c3641.png)

### Output w/ Attention (학습 단계)

![image](https://user-images.githubusercontent.com/80622859/222959878-20b655ab-2544-4882-ad35-45fc94a2f3c2.png)

