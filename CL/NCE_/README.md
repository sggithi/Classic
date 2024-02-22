### Todo

- [ ] 제대로 이해해서 코드 구현해보기

  https://github.com/lifeitech/nce

  

## ABSTRACT
 observed data와 artificially generated noise를 구분하는 모델

## Introduction
 unknown 확률 밀도 함수 p_d(.)이 있고 p_m(.;a)의 parameterized family에 의해 모델링된다

  본논문에서는 특정 objective function을 최대화하여 관찰된 결과로부터 alpha를 추정하는게 목표!

## 2 Noise-contrastive estimation

  ![](https://velog.velcdn.com/images/seogimin/post/1326ca9d-bc50-4f53-85c7-413221f864f4/image.png)

<img src = "https://velog.velcdn.com/images/seogimin/post/18a43ddb-7d56-41ef-a962-ade6230b4455/image.png" width = 300/>

결국 data에서 나온 확률은 최대화하고, artifical generated noise에서 나올 확률은 최소화하는 문제