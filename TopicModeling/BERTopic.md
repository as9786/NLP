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
- 
