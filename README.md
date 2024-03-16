# League of Legends 데이터분석 프로젝트

transformer 모델과
CBOW, Skip-gram을 활용하여 챔피언 임베딩을 훈련시켰고, 결과를 시각화한 프로젝트


사용한 데이터: Riot API master 티어 전적 데이터
목적: 챔피언의 유사도를 반영한 벡터를 얻고, 이를 바탕으로 가장 유사한 챔피언을 찾는 코드


data preparation example(MatchDatamaster.csv):
(You can prepare dataframe by preprocessing Riot API)

```
champion0,champion1,champion2,champion3,champion4,champion5,champion6,champion7,champion8,champion9,winner
Yone,Rell,Swain,Aphelios,Rakan,Leblanc,Viego,TwistedFate,Ezreal,Alistar,red
KSante,Belveth,Ryze,Cassiopeia,Swain,Darius,Graves,Akali,Zeri,Rakan,red
Yasuo,Gragas,Zed,Sivir,Soraka,KSante,Rell,Yone,Zeri,Yuumi,blue


...
```

Transformer-based embedding visualization command


Transformer 모델을 기반으로 승패를 예측하고, 그를 기반으로 Embedding map을 형성합니다. (CBOW에 비해 명확하게 임베딩을 형성하지 않음)


'''
python transformer.py
'''


CBOW-based embedding visualization command


해당 코드를 실행하면 수집한 csv파일을 바탕으로 각 챔피언의 벡터를 추출하고, 추출한 벡터를 기반으로 t-SNE 기반 시각화 자료를 출력합니다.


```
python cbow.py
```

t-SNE Embedding example


<img width="80%" src="https://github.com/Leehongseok-code/RiotAI/assets/52267586/c2a9cf0b-ab2f-4f72-8a33-e5c60a35e5a7"/>



참고 자료: https://queez0405.github.io/lol-project/
