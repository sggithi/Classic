한마디로 기존의 방식들은 inherently transductive하기 때문에 unseen nodes에 일반화하기 어렵기 때문에 본 논문에서는 node embedding을 직접 학습하기 보다는 **embedding을 만드는 함수를 학습시키겠다**

### 1. Introduction
Inductive framework는 node들의 structural property를 인식할 수 있어야 한다.
node feature를 활용해서 embedding function을 배운당
(이러면 완전히 unseen data라고 할 수 있나?? 여튼 node 정보를 일부 활용해서 embedding을 만들긴 하는듯 하긴 이것도 unseen이라할 수 있을듯,,)
 node feature들을 학습 알고리즘에 통합함으로써, 이웃의 node feature 분포를 배우는만큼 각 노드의 이웃들의 topological structure를 배울 수 있다.
 
 **node의 local neighborhood로부터 feature 정보를 받아서 aggregate하는 aggregator function을 배우는 것**
 
 ### 3.1 Embedding generation algorithm
 k개의 loop를 둘면서 k hop 이웃으로 점차멀어지는 순으로 aggeragte
 현재 k step일 때,
  k-1까지 만들어진 이웃 노드들의 representation을 aggregate해서 single vector로 만들고 그거랑 k-1의 현재 노드랑 concat, 이후 FC 통과시키고 normalize
  마지막 k번째까지 다 돌았을 때 나온 output이 최종 (z)
  **mini batch 가능**
  
  흥미로운 점 K = 2일때 높은 성능을 얻을 수 있었다!!
GCN도 2-layer면 충분한 것처럼 이웃은 2-hop까지가 유의미한듯??

### 3.3 Aggergator Architectures
 #### Mean
  단순 이웃 노드들 + 중심 노드의 평균
  -> concat 필요 없음
#### LSTM
 sequential하게 넣기 때문에 symmetric하지 않음
 -> random permutation을 통해 unorder set 이웃
 
#### Pooling
 symmetric and trainable
 max pooling을 통해 이웃들의 다른 점을 포착할 수 있음


 ## Implementaion
  일단 Q나 negative sample loss가 이상하고, epoch을 더 돌리면 loss가 무한대로 발산함,,, 이게 가능한 일인가????????? (아마도 node 전부 써서 메모리 부족해서 생긴 오류인듯??)
  https://github.com/williamleif/GraphSAGE/tree/master/graphsage
  참고해보기
 
