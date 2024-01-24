## Introduction

추천 시스템에서 중요한 것은 **memorization**과 **generalization**

 Generalized linear 모델을 쓰면 학습에서 보았던 정답에
대해서 memorize가 가능하지만 unseen data에 대해서 사용하기 어렵고, Embedding-based model을 쓰면 unseen data에 대해서도 일반화가 가능하지만 학습 데이터가 sparse할 때는 또 효과적으로 학습하기 어려움

 그래서 이 논문에서는 "wide"(학습 데이터를 기억)하고 "deep"(일반화가 가능)한 component를 모두 합쳐서 학습하는 모델을 만들었다.

## Overview
 구글에서 제안한 모델은 대부분 2-stage로 가는듯
 일단 query가 들어오면 모든 걸 다하는 대신 어느정도 골라진 작은 set중에서 후보를 고름!
 
 ## 3. WIDE & DEEP LEARNING
  ### 3.1 The Wide Component
   generalized linear model이고, train data에 대해서 잘 맞추는 부분
  <img src ="https://velog.velcdn.com/images/seogimin/post/17d3d24e-13dc-4d1b-81b0-cf8e6bcb488f/image.png" width = 300>

 정말 각 feature마다 하나하나 cross-product 느낌이라 SVM에 가까운 느낌
 
 ### 3.2 The Deep Component
  우리가 흔히 생각하는 deep learning layer 
  embedding vector 구하는게 목표
    <img src ="https://velog.velcdn.com/images/seogimin/post/bc411d69-cd38-4d58-b079-d61b734ba1c1/image.png" width = 250>

### 3.3 Join Training
 **Ensemble**
  아예 개별 모델을 따로 학습시켜서 추론할 때만 prediction을 합침
  보다 좋은 성능을 위해서는 개별 모델의 size가 커야함
 **Join Training**
  모든 파라미터를 동시에 최적화, training time에 합침
  각각의 모델이 서로의 약한점만 보완해주면 돼서 모델 사이즈가 클 필요가 없음!
  
  
  본 논문에서도 Deep Model의 취약점(cross-product가 적은 경우)에서만 Wide Model이 보완해주면 돼서 full-size wide model일 필요가 없게 됨!!
  
  
  
  ## TODO
   코드 좀 수정해야함 DataLoader 사용하는 쪽으루
