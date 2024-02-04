class와 정답을 이용한 deterministic model이 아니라, 확률 밀도 함수를 모델링하는 Generative model

결국 생성은 추정한 확률 밀도 함수로부터 sampling을 통해 일어남

#### VAE
q(z|x)와 p(x|z)를 모델링하고자 함
결국엔 확률 분포를 모델링하는 거지만, 그렇게 모사될 수 있도록 하는 파라미터를 학습시키는 것

#### RBM
visibile layer와 hidden layer의 분포
https://www.youtube.com/watch?v=UcAWwySuUZM
