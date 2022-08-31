# 기계학습(MachineLearning)



## 바둑 경기 기력 예측 머신러닝 만들기(predict_Baduk_rating)

- 개요
  + 팀별로 자신만의 바둑 경기 기력 예측 머신러닝 만들기
  + 주어진 train data와 train label(18급~9단)을 이용하여 기력 예측 인공지능을 만들고, test data로 검증
  + 최종적으로, 구현한 기력 예측 인공지능을 이용해 query data를 예측해 label 제출

- 구현 사항
  + 바둑 기보를 50수까지 제한 및 패딩
  + 2차원 이미지(CNN)의 순차적 배열(RNN) → tensorflow의 ConvLSTM2D를 이용해 모델 학습 및 검증
  + 최종 구현한 모델을 이용해 query data 예측

- 정확도
```
  Accuracy : 12.18519%
```

- 퍼블릭 랭킹
```
  Ranking : 1위(14팀 중)
  14팀 평균 Accuracy : 10.88%
```

