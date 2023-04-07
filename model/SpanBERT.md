# SpanBERT : Improving Pre-training by Representing and Predicting Spans

## Abstract

- Text span을 잘 표현하고 예측하도록 design된 사전 학습 방법 제안
- Random token이 아닌 연속적인 random span을 masking
- 개별 token representation에 의존하지 않고, mask 범위의 전체 내용을 예측하도록 span boundary representation 학습

## 1. Introduction

- Span-leve pre-trian
- Masking random contiguous span
- Span-Boundary Objective(SBO) : 관찰된 token으로부터 전체 masked span을 예측하도록 모형을 학습
- SBO는 모형이 span-level information을 boundary token에 저장 -> 미세 조정 시 쉽게 접근

![image](https://user-images.githubusercontent.com/80622859/230544351-31c33da5-6df1-44e9-8d69-64f281055523.png)

- 위의 그림에서 was와 to(boundary token)을 사용하여 maksed span을 예측

## 2.Model

- Individual token이 아닌 다른 방식으로 token span을 masking
- Span boundary에서 token representation만 사용하여 전체 masked span을 예측하고자 하는 새로운 auxiliary objective인 SBO 제안
- Training example에 대해 하나의 연속적인 text segment를 추출(NSP 사용 X)

### 2.1 Span Masking

- Token으로 구성된 sequence X = $(x_1, ..., x_n)이 주어지면 masking budget(15% of X)이 사용될 때까지 text span을 반복적으로 추출
- token Y, $Y \subseteq X$
- 각 반복마다 기하 분포 $\ell ~ Geo(p)$를 통해 span length sampling
- 해당 분포는 더 짧은 span으로 편향됨(skewed)
- Span의 시작점을 무작위로 균일하게 선택
- p = 0.2로 설정하고, $\ell_{max} = 10$으로 제한. $\bar\ell=3.8$의 평균 span length를 반환
- Subword token이 아닌 완전한 단어를 span lenth로 측정하여 mask range를 넓게 만듦

![image](https://user-images.githubusercontent.com/80622859/230545172-c08f237c-ecd0-4e9e-932e-a6e2cd4278a2.png)

- BERT에서와 같이 총 15% token을 masking
- Masked token의 80%는 [MASK], 10% random token, 10% original token으로 설정
- Span level에서 수행하기 때문에 범위의 모든 token이 [MASK] 또는 표본 추출된 token

### 2.2 Span Boundary Objective(SBO)

- 일반적으로 boundary token(시작과 끝)을 사용하여 span의 고정된 길이 표현을 만듦
- Boundary에서 관찰된 token의 표현만을 사용하여 masked span의 각 token을 예측
- Masked span $(x_s,...,x_e) \in Y$, (s,e)는 시작과 끝의 위치
- External boundary token $x_{s-1}, x_{e-1}$을 사용하여 encoding target token $P_i$의 position embedding을 사용하여 각 span의 token $x_i$를 나타냄

![image](https://user-images.githubusercontent.com/80622859/230545849-baebdc87-f9af-4324-ae9f-a66d599f4846.png)

- 본 논문에는 GELU 및 계층 정규화를 사용하여 f를 2개의 층을 가진 순전파 신경망으로 구현

![image](https://user-images.githubusercontent.com/80622859/230545995-c81d43ff-cc47-4f32-92aa-20b8d6d57fca.png)

- y를 사용하여 x를 예측하고 cross-entropy loss 계산



