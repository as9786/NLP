# DALL-E: Zero-Shot Text-to-Image Generation

- Denosing VAE를 사용하여 pixel 단위의 image를 image token으로 변환
- Large data and large model

## Abstract
- Transformer
- AUto-regressive
- Zero-shot

## Background

- Input : text, output : image
- Dataset : MS-COCO, CUB

#### MS-COCO
- Dataset for object detection, segmentation, captioning task
- There are 5 captions each image when using image captioning

![캡처](https://user-images.githubusercontent.com/80622859/200106834-900c1e79-73bc-45e4-bccc-10552613729d.PNG)

#### CUB
- 북비 새 200 종에 대한 11,788개의 image dataset
- 각 이미지에 대한 5개의 fine-grained 설명 포함

![캡처](https://user-images.githubusercontent.com/80622859/200106878-d25cd461-1422-4207-a095-403cdc03e4a4.PNG)

### 평가 척도

#### IS(Inception Score)
- 생성된 image의 질을 평가하기 위한 척도. 특히, 생산적 적대 모형 평가에 사용. 사전 학습된 DL을 사용하여 생성된 image를 분류(특히, inception-V3 model 사용)
- Image quality : Image는 어떤 특정 물체 같아 보이는가?, Image variety : 다양한 물체가 생성되었는가?
- 1 ~ 1000 점

#### FID(Frechet inception distance)
- Score calculated by calculating the distance between the feature vectors between the real image and the generated image.
- Inception-V3 model. 마지막 pooling layer에서 나온 vector 간의 거리를 평가
- 2020년 기준 생산적 적대 신경망의 결과를 평가하는 표준 척도
- 낮을수록 좋은 모델

#### 기존 접근법
- 기존의 연구는 특정 dataset에 대해 잘 작동하는 modeling 기법을 찾아내는 데 집중
- 이와 같은 과정에서 복잡한 구조나 보조 손실함수, 추가적인 정보가 활용. ex) AttnGAN, DM-GAN, DF-GAN 등


