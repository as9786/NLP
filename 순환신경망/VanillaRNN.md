# 기본적인 순환 신경망

- 순차 데이터를 처리

## Memory system

![image](https://user-images.githubusercontent.com/80622859/222882113-6579cb12-1b2c-45cb-a7f1-2dca9a295e11.png)

- 올바른 대답을 하기 위해서는, 입력을 받을 때 마다 그 내용을 기억해야 함
- 이전 입력을 기억하지 않는 system은 memoryless system

## 얕은 신경망(Shallow Neural Network)

![image](https://user-images.githubusercontent.com/80622859/222882142-9c9d8c25-8825-4cdf-a168-3898f8e8a43a.png)

- 대표적인 memoryless system
- n번째 time stemp에 대한 결과가 이전 입력에 영향을 받지 않음

## 기본적인 순환 신경망(Vanilla Recurrent Network)

![image](https://user-images.githubusercontent.com/80622859/222882235-eb717c7d-d3c4-4d80-b0d4-3f4ee431b6f5.png)

- 얕은 신경망 구조에 순환이 추가
- Memory system이므로, RNN의 출력은 이전의 모든 입력에 영향을 받음

## 다중 계층 순환 신경망(Multi-layer RNN)

![image](https://user-images.githubusercontent.com/80622859/222882302-04b588aa-0a29-46e5-8aeb-c154ae7173b9.png)

- 심층 신경망처럼 순환 신경망을 쌓은 것
- 신경망의 구조가 매우 복잡해지고 학습이 잘 되지 않아, 권장되지 않음
