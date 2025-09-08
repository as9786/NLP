# BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

## 초록

<img width="1280" height="516" alt="image" src="https://github.com/user-attachments/assets/580062a9-885f-4ea7-ba1c-18cb1af50b73" />

- New VLP framework
- 이해 기반 작업과 생성 기반 작업 수행
- Web에서 수집된 noise가 많은 data로부터 학습 성능을 극대화하기 위한 data bootstrapping 제안

## 서론

## VLP
- 사진과 글을 함께 이해하고 활용하는 기술
- Noise가 많은 data를 사용하는게 문제
- CLIP, ALBEF, SimVLM
- Encoder based model은 글을 생성하기에는 적합하지 않음
- Encoder-Decoder based model은 image-text matching과 같은 이해 작업 성능이 부족

## 모형 구조

<img width="1280" height="724" alt="image" src="https://github.com/user-attachments/assets/65f7a5e0-93a8-4a9b-a685-4a08b68af709" />

- Multimodal mixture of Encoder-Decoder(MED)와 data quality를 개선하는 Captioning and Filtering(CaFilt) 방법론 적용

### 1. MED
- 세 가지 mode 존재
1. Unimodal encode
  - 사진과 글을 개별적으로 encoding하여 각각의 representation vector 생성
  - 제일 앞에 [CLS] token
2. Image-grounded Text Encoder
  - Text encoder 각 transformer block에 cross attention 추가
  - 영상으로부터 얻은 정보를 text encoder에 통합하여 multi-modal representation 생성
  - 글과 사진의 관계를 학습
3. Image-grounded Text Decoder
  - 영상을 기반으로 글을 생성
  - Casual self-attention을 사용하여 입력된 사진의 정보를 바탕으로 순차적으로 글을 생성
  - 글 생성을 시작하는 [Decode] token과 끝을 나타내는 [EOS] token 포함

### 손실 

### 1. Image-Text Contrastive Loss(ITC)
- 사진과 글 간의 특징 공간을 정렬하기 위한 손실
- 동일한 사진-글 쌍은 가까운 embedding space에 위치하도록 학습
- Momentum encoder & Soft label

### 2. Image-Text Matching Loss(ITM)
- 글과 사진이 실제로 사상되는지를 이진 분류 방식으로 학습
- Hard negative mining 사용 -> 모형의 정교함 상승

### 3. Language Modeling Loss(LM)
- 영상을 기반으로 글ㅇ르 생성하는 방법을 학습
- 자기 회귀 방식. Label smoothing을 통한 모형의 일반화 능력 향상
- Text encoder와 decoder가 self-attention 계층을 제외한 매개변수를 공유

## CapFilt

<img width="1280" height="212" alt="image" src="https://github.com/user-attachments/assets/42455d39-bda5-4453-b984-d8c465f53e73" />

- Web에서 수집한 noisy image-text data 정제

### 1. Captioner
- Image-grounded text decoder를 활용해 web image에서 synthetic caption 생성
- 생성된 caption은 각 사진에 하나씩 매칭

### 2. Filter
- Image-grounded text encoder를 사용하여 원본 web text와 synthetic caption의 정합성을 평가
- 비정합으로 분류된 data 제거

## 학습
- MED를 web dataset으로 full training(Noise data 포함)
- 그 후, 고품질의 소규모 dataset으로 미세 저종
- 미세 조정된 MED 모형을 활용하여 data bootstrapping 진행
- Bootstrapping을 통해 새롭게 생성된 dataset을 사용하여 MED 모형을 다시 사전 학습 

