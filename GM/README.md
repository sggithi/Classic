class와 정답을 이용한 deterministic model이 아니라, 확률 밀도 함수를 모델링하는 Generative model

결국 생성은 추정한 확률 밀도 함수로부터 sampling을 통해 일어남

#### VAE
q(z|x)와 p(x|z)를 모델링하고자 함
결국엔 확률 분포를 모델링하는 거지만, 그렇게 모사될 수 있도록 하는 파라미터를 학습시키는 것

#### RBM
visibile layer와 hidden layer의 분포
https://www.youtube.com/watch?v=UcAWwySuUZM

#### VAE vs RBM
 둘다 확률 분포를 모델링하는 것은 같지만 VAE의 경우 layer를 통해서 generative하고, RBM은 rejected sampling 개념을 차용해서 generative를 한다.
 RBM을 반복을 1회 (visible -> hidden -> hidden)만 하면 VAE와 유사해보이긴 하지만 VAE는 압축하고 압축한 데이터를 복원하는게 주 목적이고
 RBM은 보이는 것으로부터 숨겨진 데이터를 추출하고, 또 hidden으로부터 visible을 유도하는 (... 생각하면 할수록 헷갈림,,,)

 