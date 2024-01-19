### Abstract
 TransE: multi-relational data를 저차원 벡터 스페이스로 임베딩하는 모델
 
### Introduction
 (h, l, t) : label은 head와 tail 사이의 relationship
 한마디로 Word2Vec과 같이 node들의 관계를 embedding space에서 표현하겠다. (Simple하게!! 이게 핵심)
 ![](https://velog.velcdn.com/images/seogimin/post/1a19963e-5ae1-4824-8bd1-5320a29e4325/image.png)
 신기한점은 앞의 계수가 6
 
 결국엔 학습 자체가 (h+l, t)의 차이가 (h'+l, t')보다 margin보다 더 작게 되도록 (여기서 negative mining이 필요함)
 = 실제 relationship이 l인 애들의 경우 h + l이 t에 가까워야하고, 관계가 아닌 애들은 h+l이 t와 멀어야 함!
  
  
  ### Evaluation
  
  [3]에서 head 지우고, 다른 entity들 dissimilarities 계산해서 지워진 entity의 rank
  => tail을 지워서 predicted rank의 평균을 구하거나 hits@10으로 성능 평가
  
 ### 의문인점은
~~어떻게 mean Rank가 263인데 Hit@10이 75퍼센트가 나오지??대부분이 263등쯤에 정답이 있는 건데 10퍼안에 드는 경우가 75프로??~~

 14,951개가 있으니까 10퍼센트면 1500 정도고 충분히 가능한듯!