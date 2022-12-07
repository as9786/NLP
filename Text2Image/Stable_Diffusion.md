# Stable Diffusion

- 여러 연구자들이 모여서 Image generation model을 보다 상용화하기 위한 목적으로 만듦
- 모형 자체의 혁신보다는 모형을 공개했다는 점에서 더 큰 관심을 받음
- 비싼 resource를 들여 학습한 모형과 code 뿐 아니라 사용한 data를 모으는 방법 등을 모두 공개
- 일반 GPU 1장으로도 충분히 추론할 수 있을 정도로 효율적이며 성능이 잘 나옴
- Promt engineering을 조금만 진행하면 GPU 1대만 있는 그리 높지 않은 성능의 server에서도 준수한 결과
- 이는 후에 많은 응용 모형과 service의 등장이 기대 됨

## 방식

### Imagen의 기본 구조
- 사용자가 입력하는 caption은 text encoder로 들어가고 이는 numerical representation으로 변경
- 위의 tet encoding 값을 활용해서 Image generation model에서는 샘플 노이즈로부터 output을 생성해 냄
- 처음 이미지의 크기는 매우 작지만 super resolution을 거쳐서 최종적으로 1024 x 1024 크기의 이미지 생성
- Super resolution 과정 중에도 text encoding 값을 활용하여 성능을 향상시킴

![image](https://user-images.githubusercontent.com/80622859/204794471-62eed031-70eb-491f-9313-bd1e9365881a.png)

### Text encoder
- Transformer 기반
- Imagen이 학습하는 과정에서는 frozen된 상태로 유지
- 단순히 text encoding을 생성하는 데만 사용

### Diffusion model
- Training dataset과 비슷한 data를 생성하는 방법
- Training data에 지속적으로 noise를 추가해서 data를 망가뜨리고, 이를 원상 복구하는 과정을 학습

![image](https://user-images.githubusercontent.com/80622859/206161609-7d384fd9-b32f-4133-8ced-600ece8755ab.png)

- Denoising autoencoder와 매우 유사
- 내부에 사용되는 model 등이 좀 더 개선

### Image generator & Caption conditioning
- DDPM 논문에 소개된 UNet 구조를 그대로 사용

![image](https://user-images.githubusercontent.com/80622859/206161903-4f79909e-be16-4457-99b3-7a264a35d4ed.png)

- Text encoding 값을 조건으로 주도록 하여 원하는 결과가 나오도록 함

![image](https://user-images.githubusercontent.com/80622859/206162042-fddec6d4-8e57-4b96-a1d3-976a7961152d.png)

### Super resolution
- Image generator로 생성된 사진은 64 x 64로 크기가 작기 때문에 해상도를 향상시키는 것이 필요
- Diffusion 사용
- Image generatio 과정과 유사. 다만 완전한 noise가 아닌 image generator로부터 생성된 image와 text encoding 값을 조건으로 주어서 해상도를 높임

![image](https://user-images.githubusercontent.com/80622859/206162716-ba176156-d05b-40e3-9054-4a83334202f8.png)

- 기존 UNet이 아닌 속도 개선 등 최적화를 한 Efficient UNet 구조를사용
- Imagegen에는 두 개의 SR model이 존재
1. Small to medium : 64x64 -> 256x256
2. medium to large : 256x256 -> 1024x1024
- 위의 구조는 MTL의 경우에는 self attention이 제거됨

## Timestep conditioning & Caption conditioning
- Denoising을 하는 각 시점에서 조건화를 하기 위해서 transformer에서 기본적으로 많이 사용되는 positional encoding을 활용
- 각 시점 별로 encoding이 생성되고 UNet의 서로 다른 resolution을 적용
- image, time, text가 모두 encoding되어 조건화

![image](https://user-images.githubusercontent.com/80622859/206163686-1f71c14c-c182-42cc-ac83-aea7e6984395.png)

### Guidance weight
- Static/dynamic thresholding을 활용
- Static thresholding은 pixel value의 range를 [-1,1] 사이로 clip
- Image가 saturated 되어 생성이 제대로 안되거나 부자연스럽게 나오는 부분을 해결해 줄 수 있지만 image detail을 떨어뜨림
- Dynamic thresholding : 특정 percentile의 pixel value가 선택이 되고 이 percentile을 넘어가면 s 값으로 나누도록 하여 [-s,s] 사이에 위치하도록 하는 방법

![image](https://user-images.githubusercontent.com/80622859/206164897-e22cad03-dae6-44cc-ba17-153893b3fab6.png)
