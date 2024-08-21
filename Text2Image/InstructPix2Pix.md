# InstructPix2Pix: Learning to Follow Image Editing Instructions

## 서론

![image](https://github.com/user-attachments/assets/51a53aee-c39b-47e7-91ef-169adfe67860)

- 영상 편집을 위해 사람이 작성한 명령을 따르도록 생성 모형을 가르치는 방법 제시
- 서로 다른 modality로 학습된 pre-trained GPT3와 stable diffusion을 결합한 쌍으로 dataset을 생성
- 생성된 쌍을 사용하여 입력 영상과 편집 방법에 대한 text 명령이 주어지면 편집된 영상을 생성하는 conditional diffusion model을 학습
- Forward pass에서 영상 편집을 직접 수행하며, 추가 예제 영상, 입력/출력 영상에 대한 전체 설명 또는 예제별 미세 조정이 필요하지 않음
- 
