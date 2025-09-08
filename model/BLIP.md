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

## 방법론

<img width="1280" height="724" alt="image" src="https://github.com/user-attachments/assets/65f7a5e0-93a8-4a9b-a685-4a08b68af709" />

- Multimodal mixture of Encoder-Decoder(MED)와 data quality를 개선하는 Captioning and Filtering(CaFilt) 방법론 적용

### 1. MED
- 세 가지 mode 존재
1. Unimodal encode
  - 사진과 글을 개별적으로 encoding하여 각각의 representation vector 생성
  - 제일 앞에 [CLS] token 
