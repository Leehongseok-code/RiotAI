# League of Legends 데이터분석 프로젝트

transformer 모델과
CBOW, Skip-gram을 활용하여 챔피언 임베딩을 훈련시켰고, 결과를 시각화한 프로젝트


사용한 데이터: Riot API master 티어 전적 데이터

data preparation example(MatchDatamaster.csv):
(You can prepare dataframe by preprocessing Riot API)


champion0,champion1,champion2,champion3,champion4,champion5,champion6,champion7,champion8,champion9,winner
Yone,Rell,Swain,Aphelios,Rakan,Leblanc,Viego,TwistedFate,Ezreal,Alistar,red
KSante,Belveth,Ryze,Cassiopeia,Swain,Darius,Graves,Akali,Zeri,Rakan,red
Yasuo,Gragas,Zed,Sivir,Soraka,KSante,Rell,Yone,Zeri,Yuumi,blue
...

embedding visualization command
```
python cbow.py
```


참고 자료: https://queez0405.github.io/lol-project/
