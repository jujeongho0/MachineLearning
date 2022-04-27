# 기계학습(MachineLearning)


## 바둑 기사의 기력을 예측하는 인공지능 만들기(predict_Baduk_rating)

- 개요
  + 팀별로 자신만의 바둑 경기 기력을 예측하는 인공지능 만들기
  + 주어진 train data와 train label(18급~9단)을 이용하여 기력 예측 인공지능을 만들고, test data로 검증
  + 최종적으로, 구현한 기력 예측 인공지능을 이용해 query data를 예측해 label 제출
  + 데이터 셋 : https://drive.google.com/drive/folders/1GWnyr1ouSuvtvnxROiqVNaJf3t-8CavT?usp=sharing

- 구현 사항
  + 바둑판은 백 돌('1'), 흑 돌('-1'), 돌이 없는 점('0')으로 구성되어있음 → 백 돌('100'), 흑 돌('200')로 초기화(백 돌과 흑 돌의 분포를 극명하게 하기 위해)
  + 바둑 기보를 50수까지 제한 및 패딩(바둑은 초반에 승패가 갈린다는 통계 참고)
  + 2차원 이미지(CNN)의 순차적 배열(RNN) → 2차원 이미지(19x19 바둑판)의 순차적 배열(바둑 기보) → keras의 'ConvLSTM2D' 모델 이용  
  + 'categorical_crossentropy' 손실 함수 이용
  + 'RMSprop' 알고리즘 이용
  
- 정확도
```
  Accuracy : 12.18519%
```

- 퍼블릭 랭킹

<img src="https://user-images.githubusercontent.com/62659407/121729249-32b8c680-cb29-11eb-9576-6fef5ba68c72.png" width="60%">

```
  Ranking : 공동 1위(13팀 중)
  13팀 평균 Accuracy : 10.88%
```
