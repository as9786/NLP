# Enriching Word Vectors with Subword Information

## Abstract

### 기존 model의 한계
- 기존 model은 단어마다 다른 vector를 할당하여 단어의 형태를 무시
- OOV(Out of vacaburary, 모르는 단어) 혹은 낮은 빈도수로 출현하는 단어(오타가 섞인 단어)에 대해서는 word representation을 얻을 수 없음
- 기존 model은 parameter를 공유하지 않은 다른 vector로 단어를 표현=> 형태학적(Morphological)으로 복잡한 언어는 잘 표현 못함
- 위와 같은 문제점을 해결하기 위해 skip-gram을 기반으로 한 model에 각 단어를 n-gram vector의 조합으로 표현
- 철자 단위 정보를 사용하여 더 좋은 단어 표현 생성

## Model

### Subword model
- where -> \<where\>. 단어의 양 끝에 <, >를 더하여 접두사와 접미사를 구분할 수 있도록 함
- <wh, whe, her, ere, re> (n = 3)
- $3 \leq n \leq 6$ 범위의 n-gram 사용

![캡처](https://user-images.githubusercontent.com/80622859/189586139-e86b6381-12b2-4922-8179-cf2964bed971.PNG)

- OOV에 대한 학습 능력 향상
- ex) birthplace라는 단어를 학습하지 않은 상태였더라도 다른 단어의 n-gram에서 birth와 place를 학습한 적이 있다면 birthplace의 embedding vector를 만들 수 있게 됨
- 오타와 같은 빈도수가 낮은 단어에 대한 학습 능력 향상
- ex) where를 오타를 포함한 wherre와 3-grams로 비교
- where -> <wh, whe, her, ere, re>
- wherre -> <wh, whe, her, err, rre, re>
- 2개의 subword만 다름

![render](https://user-images.githubusercontent.com/80622859/189525697-2b43779f-5f2d-44c4-bff7-b1ebb5ec4be6.png)

- 단어를 n-gram vector의 합으로 나타냄
- 단어 간에 표현을 공유하도록 하여 희소 단어도 의미 있는 표현을 배움
- ex) eat, eating, eats와 같이 eat이라는 원래 단어에서 파생된 단어들의 표현을 공유함으로써 학습

## Experimental setup
- Dataset from wikipedia using 9 languages

## Results

![캡처](https://user-images.githubusercontent.com/80622859/189585670-b8a65eb5-3b25-47f4-9bcd-b63d74a21fe5.PNG)

- Arabic, german, russian에서 더 좋은 성능
- 형태적으로 복잡하거나 합성어가 많은 언어에서 성능 개선이 많이 일어남(형태적 분석에 의존한 방법)
