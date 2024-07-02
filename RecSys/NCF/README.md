2017년 WWW 억셉된 논문 사실 모델링이 현란하거나 그러진 않는데, Neural 네트워크를 CF에 적용한 첫 사례인가보다(??) 

## Instruction

 Collaborative filtering, 즉 사용자와 아이템 간의 상호작용의 시그널을 담아내는 방식은 주로 Matric Factorization 방법으로 구현된다.
  이 방법은 user와 item을 latent space로 보내 vector로 표현하는 방식이고, user의 특정 item에 대한 상호작용은 user와 item latent vector의 내적으로 표현된다.

  이때 내적은 단순히 latent feature의 곱이기 때문에 user interaction data의 복잡한 구조를 나타내는데 충분하지 않다 (저자의 주장),,,
  ~~근데 정말,, 추천에 복잡한 정보가 내재되어 있는게 맞을까? 그렇다면 트랜스포머를 통해 성능이 많이 향상되어야 하는 것이 아닐까~~

  본 논문에서는 **DNN**을 써서 interaction function을 배우고자 한다. 지금껏 DNN을 추천에 쓰려는 시도들이 있었지만, 결국엔 MF,,,

 **implicit feedback** 데이터를 활용
 이 논문의 contribution은
 1. Latent feature를 모델링하기 위한 neural network architecture를 제시하고, general framework NCF를 만들었다.

 2. MF가 NCF의 specialization이라는 것을 증명하였고,, MLP를 통해 NCF를 이끌어냈다.

 3. 두 개의 real-world 데이터셋에 대하여 NCF가 CF에 있어서 좋은 접근이라는 것을 증명했다.


 ![](https://velog.velcdn.com/images/seogimin/post/3b297100-7acc-4f85-a8e4-13a96abec17c/image.png)
1이면 유저와 아이템의 상호작용이 관찰된 경우, 0이면 관찰X

0이라고 꼭 부정적인 반응일 것이라고 할 수는 없지만, 당연히 관찰된 아이템보다는 선호도가 떨어질 것이라는 가정 (일단 흥미로워 보였으니 클릭을 했겠지,,, 하지만 클릭했지만 negative sample일수도 있고,,,,, 추천은 진짜 데이터셋이 제일 문제다,,,, 하지만 클릭을 유도하는 것이 추천 시스템의 목적이라면 이런 것도 positive로 분류하는게 맞는 것 같고
 장기적인 사용자의 이용 시간을 증대하고자함이면 true positive만을 잘 골라야 함,, 아닌가 싫어하는 걸 더 잘 골라야 하나?? 그런 것 같기두 )

 ![](https://velog.velcdn.com/images/seogimin/post/50958887-8e17-409f-a64c-a8fa430d8453/image.png)

위에서도 언급했듯이, Matrix Factorization은 다음과 같이 유저와 아이템의 latent vector(Pu, Qi)를 곱해서 나온 값이 상호작용의 점수

**이 논문에서 얘기하고 싶은 것은 inner product의 한계에 대해서 지적하고, 이를 Neural Network를 활용해 개선했다는 것이다**
![](https://velog.velcdn.com/images/seogimin/post/15772fca-b264-489a-ba88-322700e13213/image.png)

 본 그림을 보면 u4의 경우, p1이랑 제일 비슷하다는 이유로 1b의 p4'처럼 그려진다면 나머지 2, 3에 대해서 순서가 맞지 않게 됨
 따라서 MF에서 내적을 통해 최적화를 진행하다보면 latent vector의 최적화가 어려움,,ㅠㅠ,,,

 ~~근데 이게 inner product를 안쓴다고 해결될 문제일까?????,,,,,, 걍 차원을 추가해야하는 거 아닌감,,~~

 맞네 바로 밑에 나옴 ㅋㅋ ㅠㅠ

 이걸 해결하는 방법으로는 차원을 늘리는 방법이 있는데 차원이 커지면 추천 데이터셋은 sparse하기 때문에 오버피팅이 발생할 수밖에 없음T_T
  따라서 본 논문에서는 DNN을 이용해서 interaction function을 배워보고자 함

 ## Neural Collaborative Filtering

 ![](https://velog.velcdn.com/images/seogimin/post/65fa2c6c-d13b-45a1-be44-d000d47667e9/image.png)
 구조는 다음과 같이 생겼는데 Layer1에서 user와 item을 input으로 받는다. 이때 user와 item 페어 대신에 아이템의 컨텐츠 정보를 넣는다든가 하면 content based recommendation이 될 수도 있고, cold start 문제를 해결할 수도 있게 된다.

 사실 이 논문을 다시 읽으면서 뭘 배워야할지.. 약간의 의문이 있었는데 (죄송합니다 죄송합니다,,) 잘 쓴 논문이 되려면 가져야할 요소?에 대해서 생각해볼 수 있었다.
  물론 Matrix Factorization이 아닌 정말 순수한 DNN의 첫 도입이라는 의의도 있긴 하지만 이렇게 general하게 사용될 수 있도록 모델링을 해야한다는 것을 느꼈다.

  ![](https://velog.velcdn.com/images/seogimin/post/9058892d-4cc9-476e-99fe-67f7cefdda68/image.png)
Loss function을 이처럼 pointwise로 계산하고자했지만, 이건 VAE에서 배웠던 것처럼 가우시안 분포를 가정했을 때 loss가 squared loss와 같아짐
 따라서 squared loss는 어찌보면 가우시안 분포에서 나왔다고 추론할 수 있는 여지를 주게 되고 이는 implicit data를 활용한 경우에는 맞지 않음
  왜냐면 1과 0. binary니까.

  ![](https://velog.velcdn.com/images/seogimin/post/0a92a8a3-1cde-48dc-aa3c-aa09189eb913/image.png)
따라서 이 논문에서는 이런 식으로 binary classification처럼 loss를 구상하였다. 
   이게 참,, Log취하니까 덧셈이 되는 거지 그게 아니라 정말 전부 다 곱셈한다고 생각하면 정말 폭력적인 계산량이다.

   그러니까 정말 CPU를 곱셈에 쓰는 건,,, 너무 낭비 같기도 하고

   여튼 이런 간단해보이는 논문이지만 역시 정보이론 같은 수학 공부를 조금 해두어야겠다고 생각했다.

   ## Generalized Matrix Facotirzation
   자 그렇다면 이제 NCF로 MF를 만들 수 있음을 보여보자 layer 한 층인 NCF에 대해서
   ![](https://velog.velcdn.com/images/seogimin/post/f702fe4d-937d-45a6-adc4-773e0c7fa58e/image.png)
   a를 identity function, h를 uniform vector1로 주면 얘는 결국 MF가 됨

   (이 논문의 introduction에서 말했던 MF를 DNN으로 푸는 케이스)

   만약 a를 통해 nonlinearity를 더한다면, 보다 general한 MF가 됨
 이 논무에서는 sigmoid 사용

 근데 대체... 시그모이드 더 뭐가 글케 대단한데.... 왜 자꾸 쓰이는 건데
 미분했을 때 예쁘게 생겨서??

 ## 3.3 Multi-Layer Perceptron
  Layer1에서 user와 item 벡터 두개를 받는데 이때 이 둘을 단순히 concat하는 것은 collaborative signal을 표현하는데 충분하지 않다. 따라서 concat한 벡터에 hidden layer를 추가해서 user와 item의 interaction을 충분히 배우도록 하고자 함

  ![](https://velog.velcdn.com/images/seogimin/post/51df8fb8-fb36-4c0e-a31f-0e215925ab23/image.png)
그래서 대략적으로 이렇게 Layer를 쌓아줍니당

 ## 3.4 Fusion of GMF and MLP
  왜 이런 짓을,, MF 없이 MF를 이기기란 쉽지 않은가보다,,,

  ![](https://velog.velcdn.com/images/seogimin/post/c8f1ec88-4be0-46c1-bbcc-4704f598ff1d/image.png)

   아주 간단하게는 user와 item 임베딩을 공유하면 된다.

   하지만,, 단순 sharing은 어찌보면 fused model의 성능을 제한할 수도 있다. 왜냐면 GMF와 MLP의 optimal embedding size가 같은데 둘이 똑같이 해버리면 한쪽이 손해봄


 ## Experiments
  1. NCF가 정말 다른 CF 방법보다 성능이 좋은지?
  2. 어떻게 추천 태스크에서 동작하는지?
  3. 깊게 쌓는 것이 효과적일까?


 ## Dataset
 ![](https://velog.velcdn.com/images/seogimin/post/bcc680cb-7eb9-464d-9afc-a43f74a979c8/image.png)

## 4.2 Performance Comparison (RQ1)
![](https://velog.velcdn.com/images/seogimin/post/f4e26021-95f2-481e-8afd-461503ceec8c/image.png)
k도 이렇게 바꿔가면서 실험하고 전부 돌린게 신기하다 이게 바로 고전의 힘,,

## 4.2.1 Utility of Pre-training
![](https://velog.velcdn.com/images/seogimin/post/56730f92-78a8-4f65-9aa6-fbb503a47e71/image.png)
pretrain 했을 때 잘하더라

## 4.3 Log Loss with Negative Sampling (RQ2)
대부분 첫 10peoch까지 성능이 잘 향상되었다.

NeuMF > MLP > GMF 순으로 성능이 좋았다 (MLP로 MF 이김!!!!)
pairwise보다 pointwise가 좋은 이유는 당연히 샘플링,,
겨우 negative sampling 한 개만으로는 충분하지 않다.

### 의문인점

근데 또 재미있는 점은 7개보다 더 많이 뽑으면 성능이 떨어지기 시작함,,,, 어렵넹
진짜 이해가 강ㄴ가네.... SASRec이런데서는 3000개씩 뽑고 그랬던 거 같은데,,,
아 아마도 음,,, implicit data의 차이일까? 비슷한거같은데..... 진짜 왜지??????????????

 추천데이터셋을 언어처럼 취급(?)해서 트랜스포머에 돌리면 negative sample 개수가 많아도 되는데, 여기서는 왜 그렇지 않을까?

## 4.4 Is Deep Learning Helpful? (RQ3)
 ![](https://velog.velcdn.com/images/seogimin/post/ace100ec-5753-4177-b014-ccfc3d41a4da/image.png)

 0층 쌓으면 ItemPoP보다 못하더라,, 당연한 거 아닌가 ,, 아닌가.....

 

 