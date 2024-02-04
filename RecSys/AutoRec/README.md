auto encoder를 추천에 적용시킨 모델

흥미로운 점은 item / user embedding을 만드는게 목표라는 점

즉 특정 item에 대해서 다른 user들이 남긴 평점들을 input으로 넣으면 그걸 hidden layer로 압축(?)시키고 그걸 다시 복원하는 형태의 오토 인코더

### AutoRec vs RBM-CF
1. AutoRec은 분포를 생성하는 RBM-CF와 달리 discriminatrive model이라는 점 (Rating을 직접 비교함)
2. RMSE vs max log likelihood
3. RBM-CF는 contrastive

### 결과
 item-based 모델의 성능이 좋음 왜냐면 데이터가 더 많으니까
 sigmoid를 g에 썼고, nonlinearity 덕에 성능 향상 굿
 k = 500일 때까지 성능이 좋다(?) 제법 큰 디멘젼,,
 deep하게 쌓으면 어느정도 성능이 더 좋아지는듯
 
### 궁금한점은
 RBM-CF랑 VAE랑 뭐가 다른지?
 왜 베이스라인으로 LLORMA를 썼는지?
 
### 같이 나오는 개념들
#### RBM

#### RProp(Resilient Backpropagation)
 단순히 lr을 일괄적으로 가지 않고 이전의 step과 부호가 같다면 더 보폭을 늘리는 식으로 수렴을 빠르게 함
