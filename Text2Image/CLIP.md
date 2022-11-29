# CLIP

- 기존 ImageNet보다 방대한 4억개 용량의 이미지 데이터를 사용하여 representational learning 수행
- (이미지, 물체 분류) 데이터가 아닌 (이미지, 텍스트)의 데이터를 사용
- 웹 크롤링을 통해 자동으로 이미지와 그에 연관된 자연어 텍스트를 추출하여 4억개의 쌍을 가진 거대 데이터 셋 구축
- Class label이 없기 때문에 분류 문제로 학습 불가
- 주어진 N개의 이미지들과 N개의 텍스트들 사이의 올바른 연결 관계를 찾는 문제


## 1. Constrastive pre-training

![image](https://user-images.githubusercontent.com/80622859/204523429-70d2aa06-2c39-4f29-8b79-9d559f8d8baf.png)

- 이미지 인코더와 텍스트 인코더가 있음
- 각 인코더를 통과해서 나온 N개의 이미지, 텍스트 특징 벡터들 사이의 올바른 연결 관계를 학습
- 인코더는 모두 transformer 기반

## 2. Create dataset classifier from label text/Zero-shot

![image](https://user-images.githubusercontent.com/80622859/204524147-3e181e27-8802-4ac0-8448-fb79eab50c7b.png)


- Zero-shot learning 가능성을 테스트
- CLIP 모델을 이용해서 ImageNet 데이터의 분류 과정 수행
- N개의 텍스트 특징들 중 이미지 특징과 가장 높은 상관관계를 가지는 텍스트를 입력 이미지의 물체 분류 결과로 선택하여 출력
- 사전 학습 네트워크를 사용하는 방식으로 모델의 마지막 단계에 선형 분류기를 추가한 모델로도 평가 수행
