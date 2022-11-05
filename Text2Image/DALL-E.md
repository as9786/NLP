# DALL-E: Zero-Shot Text-to-Image Generation

- Denosing VAE를 사용하여 pixel 단위의 image를 image token으로 변환
- Large data and large model

## Abstract
- Transformer
- AUto-regressive
- Zero-shot

## Background

- Input : text, output : image
- Dataset : MS-COCO, CUB

### MS-COCO
- Dataset for object detection, segmentation, captioning task
- There are 5 captions each image when using image captioning

![캡처](https://user-images.githubusercontent.com/80622859/200106834-900c1e79-73bc-45e4-bccc-10552613729d.PNG)

### CUB
- 북비 새 200 종에 대한 11,788개의 image dataset
- 각 이미지에 대한 5개의 fine-grained 설명 포함

![캡처](https://user-images.githubusercontent.com/80622859/200106878-d25cd461-1422-4207-a095-403cdc03e4a4.PNG)
