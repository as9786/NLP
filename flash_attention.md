# Flash attention
- The original transformer's attention computation suffers from inefficiencies related to memoty access
- Flash attention preserves the exact attention computation while being significantly superior in both memory efficiency and speed

<img width="1796" height="1018" alt="image" src="https://github.com/user-attachments/assets/f4bcb3fb-70c8-4217-90e6-be30d84a3600" />

- There were excessive accesses to HBM(the GPU's main memory), making IO communication the bottleneck
- 불러오기 및 쓰기 과정을 줄임. Perform the computation in SRAM(the GPU's register-level cache) in a single pass
- Hardware-aware programming

## 방법

### Tiling
- Q, K, V를 작은 block으로 나눔
- 전체 행렬을 만들지 않ㅇ므

### Streaming softmax
- 부분적으로 계산하면서 softmax를 누적 계산
