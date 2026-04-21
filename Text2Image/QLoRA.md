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

### Quantile quantization
- Quantization intervals are defined based on the quantiles derived from the cumulative distirbution function of the target data distribution
- A quantization approach that guarantees an equal allocation of data points across all quantization intervals
- 
