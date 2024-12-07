## Mutual Information

단순히 중요한 정보는 보존하고, 필요 없는 정보는 줄여보자. 라고 생각했으나 사실 compression과 Fitting 두 가지의 관점에서 학습을 바라볼 수가 있다.

어쨌든 I(X;Y)를 추정해야 하는데 이때 주어진 데이터 X의 Data distribution을 알 수가 없으니 Mutual Information의 추정치에 대한 연구가 진행이 되었고, MINE에서는 Lower bound를 구해서 Mutual Information Maximization을 제시하여 GAN에서 mode collapse 문제를 해결하였다.

 Information Bottleneck의 경우는 I(X;T)를 함께 최소화하면서 불필요한 Input Feature에 대한 학습이 진행되지 않도록하여 "정말 필요한 정보"만 학습이 되도록 하는 방법이다.