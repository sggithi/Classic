## Abstract
 전통적인 추천 시스템은 사용자들의 행동을 iid(독립적이고 동일하게 분포되었다고) 가정해서, user간의 social interaction이나 connection을 무시한다. 하지만 사람의 social network는 Web에서의 personal 행동에 영향을 줄 것이다!
 
 ## Introduction
  #### 기존의 CF 문제점
  1) rating matrix의 sparsity
   user간 공통으로 평가한 item이 있어야만 similar user를 찾을 수 있음
   2) 현실에서는 친구나 속한 company에 의해 movie, music, book 등을 추천 받기도 함
   실제로 추천 시스템이 추천한 item이 high novelty factor를 가지고 있어도 친구의 추천이 더 qualified되었다고 여겨지기도 함
   
   **따라서, social network structure와 user-item rating matrix를 고려한 추천 시스템을 고안함**
   
   ### Trust-based recommender system
   기존의 model들은 user들을 independent and identically distributed되었다고 가정하지만 실제 세상에서는 그렇지 않음. 따라서 trust-based 모델에서는 user들의 reputation에 CF가 영향을 받음. 
   
   **본 논문에서는 PMF를 기반으로 user social network와 user-item matrix를 동시에 적용해 user/item latent feature space를 학습시킨다!**
   
   ### 3. Social Recommendation Framework
 social network graph도 matrix로 표현하고 UTZ, UTV로 matrix factorization을 하는데 이때 U를 공유함
 
 **3.2 Social Network Matrix Factorization**
<img src="https://velog.velcdn.com/images/seogimin/post/6e382c71-8593-4c4b-ab80-da698edf8c5b/image.png" width="400" height="100">

trust에 대한 사후 확률 분포 
<img src="https://velog.velcdn.com/images/seogimin/post/2253487e-f38b-4a7a-a1b1-e74edcea5d7e/image.png" width="400" height="2000">

이미 C는 관측되었고, 그 상황에서 U, Z가 나올 확률이 최대가 되게 하는 U, Z를 찾으면 됨

 cik는 user i가 user k를 trust하는 정도인데, 만약 user i가 다른 여러 user들을 trust한다면 감소하고, user k가 다른 user들에 의해 많이 trusted된다면 증가함
  따라서 이를 보정하기 위해 다음과 같이 변형
  <img src="https://velog.velcdn.com/images/seogimin/post/9068978e-5971-47fd-a6d7-ac194194f36d/image.png" width="250" height="100">
   **3.3 User-Item Matrix Factorization**
    <img src="https://velog.velcdn.com/images/seogimin/post/3f0d0b19-6ffe-4b6e-a5a5-5096a262cda9/image.png" width="400" height="100">
**3.4 Matrix Factorization for Social Recommendation**
  <img src="https://velog.velcdn.com/images/seogimin/post/c39cfe29-490e-4c25-9647-114471ed5d1a/image.png" width="450" height="100">
 결과적으로 U를 공통으로 학습하고 두 식을 합치면 (8)과 같이 됨
 
 **4.4 Impact of Parameter lambdaC**
  lambdaC = 0일 때는 기존의 PMF와 동일하고 inf에 가까울수록 social network 에서만 정보를 추출한 것과 동일함
  실험 결과 0이거나 inf일 때보다 그 사이 [10, 20]일 때가 성능이 제일 좋았고, 이는 fused model의 성능이 뛰어나다는 의미!
  
  **4.5 Performance on Different Users**
   user가 few rating만을 제공한 경우 해당 user의 별점을 예측하는 것이 어려운 문제였음. 이를 실험해보기 위해서 본 논문은 사용자를 각 user가 제공한 rating의 수로 분류하고 성능 평가
   나머지 baseline에 비해 no rating record를 가진 user에 대한 성능이 매우 좋았음
   **4.6**
   흥미로운 점은 lambdaC =0.1일 때는 200에서 300epoch 후에 overfit하는 경향이 있는데 labmda를 10으로 하면 overfitting 문제가 안생김 (WHY??)
   
