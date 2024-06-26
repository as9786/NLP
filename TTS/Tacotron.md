# Tacotron: Towards End-to-End Speech Synthesis

## 초록

- TTS(Text-To-Speech, 문자 음성 변환)은 일반적으로 text 분석을 위한 frontend와 음향 모형(acoustic model), audio synthesis module로 구성
- 각 구성 요소 구축을 위해서는 많은 지식이 필요
- 그래서 문자로부터 바로 음성을 생성
- Tacotron은 주어진 <text, audio> 쌍을 이용하여 임의적으로 초기화된 신경망을 처음부터 끝까지 완벽하게 학습


## 서론

- 현대의 문자 음성 변환 pipeline은 굉장히 복잡
- 언어적 정보를 추출하는 frontend model, duration을 분석하는 모형, 음향적 정보를 예측하는 모형, 복잡한 신호 처리 기반의 vocoder로 구성
- 구성 요소별로 독립적 학습을 진행하기 때문에 구성 요소들의 오차는 누적되어 더 큰 오차로 이어짐
- End2End 모형을 통해 아래와 같은 장점을 얻음
1. Feature engineering의 필요를 줄임
2. 발화자(Speaker), 언어(language), 감성(sentiment)과 같은 고차원의 특징을 쉽게 조절
3. 새로운 data에 대한 적용이 쉬움
4. 여러 단계로 구성된 모형보다 견고함

- 문자 음성 변환은 대규모 역변환 문제(inverse problem)
- 매우 압축된 text를 압축을 해제함으로써 음성으로 변환
- 기존에는 신호 단위에서 주어진 입력에 대한 다양한 변화를 처리해야 했음
- 음성 인식, 기계 번역과 다르게 출력값은 연속적이고 일반적으로 입력된 값보다 길었음
- Attention을 포함한 seq2seq 기반의 문자 음성 변환 생성 모형 제안
- 문자(Character)를 입력 받고, linear-spectrogram을 출력

## 모형

![image](https://github.com/as9786/NLP/assets/80622859/d99f0226-f7c4-4481-bf56-bd86f6d40f49)

### CBHG Module

![image](https://github.com/as9786/NLP/assets/80622859/3a210c50-dc4f-43f3-819e-fe4e21f1f093)

