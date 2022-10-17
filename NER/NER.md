# NER이란?
- Named Entity Recognition
- 이름을 가진 개체를 인식
- 개체명 인식
- 미리 정의해 둔 사람, 회사, 장소, 시간, 단위 등에 해당하는 단어를 문서에서 인식하여 추출 분류하는 기법
- 문자열을 입력으로 받아 단어 별로 해당되는 tag를 내뱉게 하는 multiclass 분류 작업
- 일반적인 개체명(generic NEs) : 인물, 장소 등
- 특정 분야 개체명(domain-specifit NEs) : 전문 분야의 용어

# NER이 필요한 이유
- 기계 번역의 품질을 높이며, 사용자에게 맞춤형 번역을 제공할 수 있도록 도와줌

# NER 성능평가 지표
- 정밀도, 재현율, F1-score를 이용해 성능을 평가하며 문장 단위가 아닌 token 단위로 평가를 진행
- ex) I work as TWIGFARM' : I, work, at, TWIGFARM, . 과 같이 5개로 나눠서 하나하나에 대해 평가 진행

# NER tagging system and label
- 문장을 token 단위로 나누고, 이 token들을 각각 tagging 해서 개체명인지 아닌지를 분간
- 단일 token이 아닌 여러 개의 tokens의 결합으로 하나의 개체명이 완성되는 경우가 있음
- 여러 개의 token을 하나의 개체명으로 묶기 위해 tagging system 사용

## BIO system
- B-(begin) : 개체명이 시작할 때
- I-(inside) : 개체명 중간에 있을 때
- O(outside) : 개체명이 아닐 경우

![화면 캡처 2022-10-17 113047](https://user-images.githubusercontent.com/80622859/196076559-e7d65cdf-eaf6-406c-993e-68ec94b52453.png)

## BIESO system
- B-(begin) : 개체명이 시작할 때
- I-(inside) : 개체명 중간에 있을 때
- E-(end) : 개체명의 마지막에 위치할 때
- S-(singleton) : 하나의 token이 곧 하나의 개치명일 때
- O(outside) : token이 개체명이 아닐 때

![화면 캡처 2022-10-17 113121](https://user-images.githubusercontent.com/80622859/196082285-29f792d2-4304-4ed3-80ab-0038594877d6.png)
