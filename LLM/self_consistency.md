# Self-Consistency Imporves Chain of Thought Reasoning in Language Models

## 1. 서론
- 추론을 입증하는 능력은 모형의 크기를 늘리는 것만으로 해결 불가
- CoT -> Greedy decoding.
- Greedy decoding -> Self-Consistency
- Instead of greedily decoding the optimal inference path, use sample-and-marginalize decoding
1. Sample from the language model's decoder to generate diverse reasoning paths2.
2. Each reasoning path may lead to a different final answer, so we marginalize out the sampled reasoning paths
  - 어떤 변수를 제거하면서 전체 확률을 합침
3. 최종 답들에서 가장 일관된 답을 찾아 최적의 답으로 결정
- 인간의 경험과 유사
- 단일 표본이 추출되는 확률성을 완화
- 비지도 방식. 사전 학습된 언어 모형과 즉시 사용 가능
- Self-Ensemble

<img width="952" height="446" alt="image" src="https://github.com/user-attachments/assets/c867c1f6-a6a8-4221-b803-a825c70e6ae1" />

- 인간마다 생각은 다르게 함. 이를 반영
- 

