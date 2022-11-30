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
