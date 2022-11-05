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

## Idea
- 120억 개 parameters의 auto-regressive transformer modle을 2억 5천만 장의 image-text 쌍에 대해 학습
- 유연하면서 자연어로 통제 가능한 image generation model을 학습
- MS-COCO dataset에서 zero-shot으로도 높은 성능
- 자연어로 묘사한 다양한 컨셉을 꽤나 창의적이고 그럴듯한 방법으로 조합

![캡처](https://user-images.githubusercontent.com/80622859/200107176-1408bb45-bb7f-46aa-84de-2e7f71bdd096.PNG)

- DALL-E의 목표 : 자가 회귀적으로 text와 image token을 하나의 stream으로 받아들여 transformer를 학습
- 엄청난 memory의 소모를 해결하기 위해 2 단계 학습 사용

### Stage 1.
- Discrete VAE를 학습하여 256x256 RGB image를 32x32 grid image token으로 압축
- 각각의 image token은 8192가지 값을 가질 수 있다고 가정
- 이러한 압축을 통해 transformer가 처리해야 하는 context 크기를 192배 압축하면서 시각적 품질은 유지

![캡처](https://user-images.githubusercontent.com/80622859/200107289-bb2bdaf3-159d-4c26-91d4-e096f050d9df.PNG)

### Stage 2.
- 256개의 BPE된 text token과 32x32=1024 image token을 이어 붙여서 transformer에 입력
- Text와 image token에 대한 결합 확률 분포를 학습

![캡처](https://user-images.githubusercontent.com/80622859/200107712-6f1e9821-e43a-4067-a70b-b4415def2b6d.PNG)

- x : image, y : caption
- Encoding된 RGB image에 대한 token x의 결합확률분포에 대한 ELB(Evidence lower bound)를 최대화하는 과정
- lower bound

![캡처](https://user-images.githubusercontent.com/80622859/200107768-81182801-265f-4aec-a76a-cd008483b870.PNG)

## Method-detail

### Stage One : Learning the Visual Codebook
- $\pi$와 $\theta$에 대해 ELB를 최대화 = image만에 대해 dVAE를 학습
- $q_p h_i$는 이산 확률 분포이기 때문에 미분값을 최대화하기 위해 reparameterize하기 어려움
- Gumbel-softmax relaxation을 사용. $q_p h_i$에 대한 기댓값을 $q_p h_{i}^T$로 대체
- Relaxed ELB는 Adam을 최대화하고, exponentially weighted iterate averaging을 사용

- 안정적인 학습을 위해 다음과 같이 세 가지 사용
1. Relaxation temperature와 step size에 대한 annealing schedule : T = 1/16으로 하면 relaxed validation ELB가 실제 ELB와 거의 유사
2. Encoder의 마지막과 decoder의 시작점에 1x1 합성곱 사용 : 일반화 높아짐
3. Encoder와 decoder resblock에서 나가는 activation에 작은 상수 곱하기 : 시작 부분에서 학습이 안정적

- KL weight beat = 6.6으로 설정했을 때 학습이 끝난 후 재구성 손실 값이 가장 작다는 것을 실험적으로 발견

### Stage Two : Learning the Prior 
- $phi$와 $theta$를 고정한 text와 image token에 대한 prior distribution을 학습
