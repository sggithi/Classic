![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/45a23f37-dde6-434e-80b6-81d09378b371/b79e5bad-67c2-45ea-96c3-207e292466a0/image.png)

여기서는 엔트로피 자체를 행렬의 eigen value를 계산하고, 행렬을 Gram matrix라는 것을 통해 계산함

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/45a23f37-dde6-434e-80b6-81d09378b371/14a6191c-1124-4ed5-b119-e47079dde18b/image.png)

그래서 결과적으로 A와 B사이의 Mutual Information은 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/45a23f37-dde6-434e-80b6-81d09378b371/cdb0e9cd-7e23-40c2-a0a8-1a7a1ed4ffa1/image.png)

이런식으로 계산한당

## Information Bottleneck

그래서 기존의 Information bottleneck method는

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/45a23f37-dde6-434e-80b6-81d09378b371/8b1d2818-2d60-4538-834e-8bbcca6f64ec/image.png)

이 Loss를 최소화하는 것이 목표임

이때

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/45a23f37-dde6-434e-80b6-81d09378b371/be95a18d-981a-4340-a8b9-054aa9f6ddda/image.png)

이렇게 치환할 수 있고, (T가 주어졌을 때 Y의 불확실성을 줄이면 되는 것임)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/45a23f37-dde6-434e-80b6-81d09378b371/898c8b31-11db-4bc9-ac52-8564ab421129/image.png)

이렇게 비례하는 관계를 세울 수 있는데, 이를 유한한 Train dataset에 대해서 식으로 바꿔주면

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/45a23f37-dde6-434e-80b6-81d09378b371/d2374e71-b6f7-42e4-a7eb-1cdd76927f97/image.png)

이렇게 되니까, 결국에는 Cross Entropy와 동일해진다고 볼 수 있겠다.

I(X;T)는 위에서 언급한 이 식을 기반으로 계산함..

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/45a23f37-dde6-434e-80b6-81d09378b371/cdb0e9cd-7e23-40c2-a0a8-1a7a1ed4ffa1/image.png)

### 여기서 Mutual Information 측정 방법
Gram Matrix라는 걸 만들어 놓고, 이를 기반으로 엔트로피를 계산한다.
왜냐면 분산이라는게 결국엔 해당 행렬이 얼마나 퍼져 있느냐(?)인 것이기 때문에 무질서도로 치환해서 생각할 수 있을듯..

