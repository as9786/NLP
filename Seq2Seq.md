# Sequence-to-sequence model

![image](https://user-images.githubusercontent.com/80622859/222951690-916e03df-90a7-42df-b13a-d68b0ce5a37f.png)

- 번역 문제를 학습하기 위해 사용되는 RNN 구조
- 왼쪽(Encoder) : 단어의 입력을 받음, 마지막 출력값을 context라고 표현
- 오른쪽(Decoder) : Context를 받아 단어를 생성. <SOS>, <EOS> tokens 사용
  
## 영어 문장의 데이터화
 
 - 학습을 위해서는 tokenizer와 embedding이 필요 
 - 단어 분리, 문장부호 제거
 - Ex) What a beautiful place! -> ['what','a','beautiful','place']
 - Tokenize
 - Ex) ['what','a','beautiful','place'] -> [4,1,234,612](Sparse representation)
 - 정해준 단어 내에 존재하는 단어들은 숫자로 변환
 - Embedding
 - Token들을 하나의 vector로 embedding 

 ![image](https://user-images.githubusercontent.com/80622859/222951814-20c0a133-5191-418b-a6f2-06fec1c96d42.png)

 ## 한글 문장의 데이터화
  
 
