# DreamBooth : Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation

## 초록
- Large Text2Img model은 높은 질의 영상 생성을 가능하게 함. 그러나 주어진 피사체를 모방하는 능력이 부족
- 본 논문에서는 주제에 대해 적은 영상으로만으로도 Text2Img model을 미세 조정하는 방식을 제안
- 특정 주제에 대한 고유한 확인자(identifier)를 묶고, 모형에 삽입되어 있는 의미적 사전 지식(semantic prior)을 prior preservation loss와 힘께 주제의 다양한 형태를 포현할 수 있도록 미세 조정

## 1. 서론

- 가장 중요한 생성 부분의 과제는 semantic pair(강아지라는 단어를 여러 개에 대한 instance에 대입할 수 있음)
- Large Text2Img model은 주제 유지가 안됨
- 근본적인 이유는 output domain이 제한되어 있음

![image](https://github.com/user-attachments/assets/8bc9957d-3467-48b0-8307-1c6eadce9c00)

- 본 논문은 개인화(personalization)에 대한 새로운 접근법을 제안
- 목표는 모형의 language-vision dictionary의 확장. 이를 통해 사용자들이 특정 주제를 자신이 원하는 새로운 단어와 결합하여 생성
- 주어진 소량(3~5장)의 주제 사진에 대해, 논문의 목적 함수가 주제를 모형의 output domain에 심음. 사용자는 주제에 묶인 unique identifier를 활용해 영상을 합성

1. 주어진 주제를 rare token identifier로 표현하는 기법
2. Diffusion based text to image framework model fine-tuning

    - Diffusion model framework는 text로부터 저해상도 영상을 생성 -> 초해상도 모형을 통한 upscale로 구성.
    - 미세 조정 또한 두 단계로 구성
    1. Unique identifier를 포함하고 있는 text prompt에 대해 저해상도 Text2Img model fine-tune(과적합 및 language drift를 방지하기 위해 class-specific prior preservation loss 적용)
    2. 초해상도 모형 미세 조정(주제의 중요하지 않은 작은 세부적인 부분도 높은 재현을 위함)
- 본 논문의 기여점
    1. 새로운 문제 정의 : Subject-Driven generation
    2. 새로운 기법 제안 : Few-Shot setting에서 text-to-image diffusion model을 기존의 semantic knowledge를 유지하면서 미세 조정하는 기법
 
  ## 2. 관련 연구 
