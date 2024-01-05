## DeepWalk
 DeepWalk = RandomWalk + Word2Vec

 Graph에서 node들 간의 random sequence를 sentence처럼 이해해서, Word2Vec 모델에 학습을 시키면 단어가 서로간의 similarity와 frequency를 반영한 embedding을 갖게 되는 것처럼 graph의 node 역시도 이웃 node와의 similarity와 frequency가 반영된 embedding을 얻을 수 있다.


#### 알면 좋을 개념들
 #### Hierachical Softmax
  word corpus의 크기가 매우 커서 그 모든 단어들에 대해서 softmax를 계산하는 것은 불가능하다.
  -> Hierahcical하게 tree의 가지들로 이해, 마지막 leaves가 words

  #### Walk
  node에서 이웃 node들을 random하게 선택하고, 이를 sequence로 이해하는 것
