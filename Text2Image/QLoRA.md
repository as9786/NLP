# QLoRA: Efficient Finetuning of Quantized LLMs

## 양자화
- 모형 가중치와 활성화 함수 출력을 더 작은 단위로 표현하도록 변환
- A model compression technique that slightly reduces the information content and precision of data to improve efficiency by decreasing storage and computational requirements
- 세 가지 방식
  1. 동적 양자화(Dynamic quantization)
  2. 정적 양자화(Static quantization)
  3. 양자화-인지 훈련(Quantization-Aware Training, QAT)
- 동적 양자화 : 가중치만 미리 양자화. 활성화 함수는 추론 시에 동적으로 양자화
- 정적 양자화 : 가중치와 활성화 함수 모두 미리 양자화
- 위 두 방식은 이미 학습이 완료된 모형을 양자화. Avoiding the trade-off between quantization loss and inference latency is challenging.
- 위 단점들을 보완하기 위해 QAT
- Fake quantization simulates post-training quantization during training by iteratively applying quantization and dequantization operations, enabling the model to adapt to quantization effects in advance
- 다만 양자화-인지 훈련은 기존 모형을 재학습해야 함

## QLoRA
- A variant of LoRA augmented with additional quantization techniques

### 4-bit NormalFloat(NF4)
- The weights of the PLM are stored in a 4-bit quantized representation, utilizing the NormalFloat data type

- Quantile quantization
  - Quantization intervals are defined based on the quantiles derived from the cumulative distirbution function of the target data distribution
  - A quantization approach that guarantees an equal allocation of data points across all quantization intervals
- 정규분포를 가정하고, 그 분포의 분위를 이용한 4 bits 양자화
- 값의 간격이 아닌 확률의 간격
- 과정
  1. 정규 분포 가정 : $w \sim N(0, \sigma^2) $
  2. 분위 계산
    - 4 bits -> 16 level
    - CDF : $F(w) = P(W \leq w)$
    - 분위 : $q_i = F^{-1} (\frac{i}{16}) $, i=1,2,...,16
  3. 구간 정의 : $[q_i, q_{i+1}]$ => bin
  4. 대표값 정의 : $c_i=E[w|q_i \leq w \leq q_{i+1}]$
  5. 양자화 : $Q(w)=c_i$

### Dobule quantization
- 양자화에 필요한 scale도 양자화

### Paged optimizer
- GPU memory가 부족할 때, 최적화 상태를 CPU/GPU 사이에서 자동으로 paging
- Paging : 필요한 것만 RAM에 
