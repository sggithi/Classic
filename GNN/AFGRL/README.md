### Abstract
 graph aumentaion 잘못했다가는 성능 망가진다. 따라서 본 논문에서는 augmentation없이 graph에서 self-supervised learning이 가능하게 하는 framework를 제안한다.
 
### Introduction
Graph의 경우 structral information이 있기 때문에 이미지처럼 augmentaion을 막하면 안된다.
 따라서 augmented view의 쌍을 만들기보다는 original graph를 하나의 view로 쓰고, representation space에서 knn을 통해 positive sample 역할을 하는 node를 찾아서 또다른 view를 생성한다. 
  그런데 또 여기서 막 knn을 찾으면 original graph의 semantic을 바꿔버릴 수 있기 때문에 false positive를 필터링하는 매커니즘을 소개한다.
  
   **BYOL**
   
    ~~이게 왜 되지...~~
    augmented view가 다른 두개의 network를 이용 
    (online network & target network)
  
    Online network
    (θ) encoder, projector, predictor로 구성
    
    Target network
    (ξ) 얘가 만든 regression으로 online network 학습
  
    진짜 너무 신기하당 
    두개의 네트워크를 이용한 방식에 대해 더 알아보면 좋을듯

    원리는 바로 두 쌍을 유사하게 만들도록 representation 학습하는 것은 맞는데 학습이 진행되다보면
    결국 모두 같은 답을 내게 만드는 쪽으로 학습이 돼서 붕괴가 일어남!
    그러한 현상을 막으려고 의도적으로 gradient 흐름을 끊고, online network에만 predictor 달아서
    두 network가 완전히 같은 답을 내뱉지 않도록 하는 방향으로 학습이 된다
    
 본 논문에서는 BYOL이 backbone이라서 negative sample 자체가 필요 없고 따라서 sampling bias또한 피할 수 있음
 
 그런데 궁금한점은 view를 바꾼다는게 무슨 의미??
 
 ### 5 Proposed Method
 어쨌든 기존의 방식들은 augmented view 두개 만들고 그 친구들 임베딩해서 걔네끼리 비슷해지게 학습을 해서 negative smaple이 필요가 없다. 그런데 이런 경우는 또 augmentaion scheme에 따라서 성능이 너무 달라짐
 즉, 하이퍼파라미터에 많이 민감해진다는 이야기 !
 
 사실 RBM도 nagative를 음...
 
 노드 vi가 있다 했을 때, 그거에 대해서 online / target representation을 통해 similarity를 구해서 그걸로 knn구하고 얘네들을 Bi라 했을 때, positive candidate가 될 수 있지만 본질적을 noisy할 수밖에 없음 (label이 없으니)
 따라서 본 논문에서는 이 중에 false positive를 필터링 !!
 
 그래서 GCN 통과시킨 임베딩으로, KNN결과의 이웃들 중에 같은 label 갖는 비율 조사하는데 당연히 k커질수록 noisy함,,
 
 **Local Structural Information**
 Representation 영역에서 knn이랑 adjacent matrix에서 이웃인 노드들의 교집합을 *local positive* 
 
 **Global Semantics**
 non-adjacent node중에서 query node랑 glrobal semantic information을 공유하는 애들 찾기 (ex. edge가 없지만 label이 같은 노드)
 target representation에서 K-means 클러스터링해서 같은 클러스터에 속한 애들이 global perpective에서 유사한 노드
 
  k-means는 centorid 초기화에 민감하기 때문에 M번해서 걔네들 합집합을 cluster로
  
  
  아 나 좀 이해됐는데 이게 graph를 augment하는게 아니라 있던 node들의 서로 다른 representation으로 similarity 계산하는 거라서 view가 다르고 augment가 필요 없다라고 하는 듯!!
  
  i, j의 유사도를 계산할 때 online network를 통과한 i랑 target network를 통과한 j의 representation으로 하는데 이게 view가 다르다는 의미인듯?? 이거 아닌 것 같기도 하고
  
  ### Obejctive function
   임의의 node vi에 대해서 "real positive"의 거리를 가깝게
   
  
  
  ## 주의할점
   반드시 Matrix 연산으로 하기!!!!!!!!!!!!!!!!!!!!
