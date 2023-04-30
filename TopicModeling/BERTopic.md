# Neural topic modeling with a class-based TF-IDF procedure

## Abstract

- Topic modeling은 문서 모음에서 숨겨진 주제를 발견하는데 유용
- 최근 연구는 군집화 작업으로 진행
- class based TF-IDF를 통해 topic representation을 추출(확장된 모형)
- 사전 훈련된 모형으로 document embedding, embedding clustering, topic representation 생성

## Introduction
- Topic modeling은 공통된 주제를 발견하거나 text에서 근본적인 이야기를 찾는데 강력한 비지도 학습 모형
- LDA(문서를 BoW으로 설명하고, 각 문서를 모형화)
- 위와 같은 모형의 단점으로는 BoW를 통해 단어 간의 의미론적 관계 무시
- 문장의 단어 문맥을 고려하지 않기 때문에 BoW가 문서를 정확하게 표현 X
- 이에 대한 해답으로 text embedding에 대한 기술이 급속도로 대중화
- Variant of BERT
- BERT 모형에서 사용되는 vector representation의 의미론적 속성은 유사한 text가 vector space에서 가까운 방식으로 text encoding
- Embedding을 군집화 한 후, 해당 군집 중심에 근접한 단어를 찾아 주제를 찾음
- Top2Vec은 Doc2Vec의 단어 및 문서 표현을 활용하여 공동으로 포함된 주제, document and word vector를 학습
- 주제 표현은 BERT 기반, 군집화는 HDBSCAN을 활용하여 생성
- 방금 언급한 모형은 군집의 중심에 근접한 단어가 해당 군집을 가장 대표하는 주제라고 가정
- 하지만 실제로는 군집에서 군집 주위의 구 내에 항상 있는 것은 아님
- 일관된 topic representation을 생성하기 위해 군집화와 class based variation of TF-IDF을 활용하는 BERTopic

1. 문서 정보를 얻기 위해 사전 훈련된 언어 모형을 사용하여 document embedding
2. 문서를 군집화하기 전에 document embedding의 차원을 줄임
3. class based TF-IDF를 개발하여 각 주제에서 topic representation 추출

- 위의 3 가지 독립적인 단계는 dynamic topic modeling과 같은 다양한 사용 사례에서 사용할 수 있는 유연한 주제 모형을 가능케 함


## BERTopic

### Document embeddings

- 동일한 주제를 포함하는 문서는 의미상 유사하다고 가정
- SBERT(Sentence-BERT) 사용
- SBERT는 문장과 단락을 조밀한 vector representation으로 변환할 수 있음
- 다양한 sentence embedding에서 최첨단 성능을 달성
- 이러한 embedding은 주로 의미상 유사한 문서를 군집화하는데 사용되며, 주제를 생성하는데 직접 사용 X
- 새롭고 향상된 언어 모형이 개발됨에 따라 BERTopic의 군집화 성능이 향상됨

### Document clustering
- 고차원 공간에서 공간 지역성의 개념이 모호해지고 거리 측정이 거의 다르지 않음
- 차원의 저주를 극복하기 위한 접근 방식이 존재하지만, 더 간단한 방식은 embedding dimension을 줄이는 것
- PCA, t-SNE는 차원을 줄이는 잘 알려진 방법이지만 UMAP은 더 낮은 투영 차원에서 고차원 data의 local 및 global 기능을 더 많이 보존
- 차원에 대한 계산 제한이 없음
- UMAP을 사용하여 document embedding의 차원을 줄임
- 차원 축소된 embedding은 HDBSCAN을 사용하여 군집화 
- HDBSCAN : DBSCAN을 계층적 군집화로 변환하여 다양한 밀도의 군집을 찾는 확장 모형 
- Soft clustering approach, noise를 이상치로 확인 가능
- 관련 없는 문서가 특정 군집에 할당되는 것을 방지하고 topic representation을 개선

### Topic Representation 
- 주제 표현은 각 분포에 하나의 주제가 할당되는 각 군집의 문서를 기반으로 모델링
- 각 주제에 대해 군집 분포를 기반으로 한 주제를 다른 주제와 다르게 만드는 것이 무엇인지 확인
- 문서에 대한 단어의 중요성을 나타내는 TF-IDF를 수정하여 주제에 대한 용어의 중요성을 나타내도록 함
- 고전적인 TF-IDF는 두 가지 통계, 용어 빈도 및 문서 빈도의 역수를 결합

![image](https://user-images.githubusercontent.com/80622859/235345877-013af366-0105-4b3b-a391-f80efde2fc2a.png)

- 용어 빈도는 문서 d에서 용어 t의 빈도를 modeling
- 문서 빈도의 역수는 용어가 문서에 제공하는 정보의 양을 측정하며, corpus N의 문서 수 t를 포함하는 총 문서 수로 나눈 log를 취함
- 위의 절차를 군집으로 일반화
- 단순히 문서를 연결하여 군집의 모든 문서를 단일 문서로 취급
- TF-IDF로 문서를 표현

![image](https://user-images.githubusercontent.com/80622859/235346123-f4747993-942c-48ac-8c98-e59d2ba235a5.png)

- c-TF-ICF
- 단일 군집의 모든 문서를 단일 문서로 간주하고 TF-IDF를 적용
- 문서 식별자를 군집 전체로 확장
- 산출된 행렬의 각 행은 해당 군집 내의 단어에 대한 중요도 점수를 갖게 됨. 즉, 군집 별로 가장 중요한 단어를 추출
- 이를 주제애 대한 설명(각 주제에 대한 단어 집합)으로 볼 수 있음
- 각 군집은 문서의 집합이 아닌 단일 문서로 변환
- 그런 다음 class c에서 단어 x의 빈도를 추출
- class-based tf
- Class A당 평균 단어 수를 모든 class에 걸친 단어 x의 빈도로 나눈 log
- 
