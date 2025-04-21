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

### Cascaded Text-to-Image Diffusion Models
- 확산 모형은 정규 분포에서 표본 추루한 변수를 점진적으로 denoising하여 data distribution을 학습하는 확률적 생성 모형
- 고정 길이의 Markov forward process의 reverse process를 학습
- Cascaded 방식은 출력 해상도가 64인 기본 text-to-image model을 사용하고, 2개의 text conditioning SR model을 사용

### Vocabulary Encoding
- 글 조건은 시각적 품질과 의미론적 정확도에 매우 중요
- 기존 논문들은 학습된 사전 분포를 사용하여 image embedding으로 변환(CLIP, 대규모 언어 모형 등)
- 본 논문에서는 사전 학습된 대규모 언어 모형 사용
- Tokenized text prompt embedding 생성. Vocabulary encoding은 prompt embedding을 위한 전처리 단계에서 중요
- Sentencepiece tokenizer 사용
- Text prompt를 sentencepiece로 tokenize 시, 고정된 길이의 vector f(P)를 얻음
- 언어 모형은 해당 token index vector로 조건화하여 embedding 생성
- 마지막으로 text2image model이 직접적으로 c로 조건화

## 3. 방법 
- 글 설명 없이 특정 피사체에 대해 간단히 포착된 사진(3~5개)들이 주어져을 때, 높은 정확도로 text prompt에 의해 지시된 피사체의 새로운 사진을 생성하는 것이 본 논문의 목표
- 피사체는 다양한 문맥을 가질 수 있음
- Ex) 피사체가 있는 장소 변경, 색상, pose 등

![image](https://github.com/user-attachments/assets/cd1f4963-f22e-473b-a309-757b784179e1)

- 첫 번째 작업은 subject instanece를 모형의 output domain에 이식하고 피사체를 고유 식별자로 묶는 것
- 피사체를 보여주는 작은 사진들을 미세 조정하면 주어진 사진에 대해 과적합되는 문제 발생
- Language drift
- 모형은 동일한 class의 다른 주제를 생성하는 방법을 잊고 다양성과 해당 class에 속하는 instance의 자연스러운 변형에 대한 지식을 잃음
- 본 논문은 확산 모형이 주제와 동일한 class의 다양한 instance를 계속 생성하도록 장려하여 과적합을 완화하고 langauge dirft를 방지하는 autogenous class-specific prior preservation loss 제시
- 꼼꼼하기 위해서는 모형의 초해상도 부분도 미세 조정 해야 됨
- Pre-trained Imagen model

![image](https://github.com/user-attachments/assets/19183145-d814-4cb0-ba1e-2c47691a2efe)

### 3.1 Representing the Subject with a Rare-token Identifier

#### Designing Prompts for Few-shot Personalization
- 새로운 쌍을 확산 모형의 사전에 삽입하여 주제에 대한 key가 주어지면 text prompt로 유의미한 의미 수정을 통해 이 특정 주제의 완전히 새로운 사진을 생성할 수 있도록 하는 것
- "a [identifier] [class noun]"의 모든 입력 사진에 label을 지정
- [identifier] : 주제에 연결된 고유 식별자. [class noun] : 주제의 대략적인 class descriptor
- Class descriptor는 분류기를 통해 얻음
- Class의 사전 분포를 고유한 주제에 연결하기 위해 문장에서 class descriptor를 특별히 사용
- 본질적으로 확산 모형의 특정 class에 대한 사전 분포를 활용하고 이를 대상의 unique identifier embedding과 얽히게 하려고 함
- 다양한 맥락에서 피사체의 새로운 표현을 생성하기 전에 시각적인 요소를 활용
- 언어 모형과 확산 모형 모두에서의 weak prior를 갖는 식별자가 필요

#### Rare-token Identifiers 
- 단어에서 상대적으로 rare-token을 찾은 다음 text space로 반전시킴
- 먼저 단어에서 rare-token lookup을 수행. Rare token identifier의 sequence $f(\hat V)$를 얻음
- f : Tokenizer, $f(\hat V)$ : Text sequence를 token에 mapping하는 함수, $\hat V$ : token $f(\hat V)$에서 파생된 decoded text, k : 가변 길이(초매개변수)
- 상대적으로 짧은 sequence(k=1,2,3)에서 잘 작동
- $f(\hat V)$에 de-tokenizer를 사용해 단어를 반전하여 고유 식별자 $\hat V$를 정의하는 일련의 문자를 얻음

### 3.2 Class-specific Prior Preservation Loss

#### Few-shot Personalization of a Diffusion Model
- 목표 대상을 묘사하는 작은 image set and text prompt 'a [identifier] [class noun]'에서 얻은 동일한 condition vector $c_s$를 사용하여 원본 확산 모형의 denoising loss로 text2img fine tuning
- 위 방식은 두 과적합과 language drift라는 문제 발생

