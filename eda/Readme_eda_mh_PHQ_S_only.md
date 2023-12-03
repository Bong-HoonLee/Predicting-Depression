담당 케이스: `mh_PHQ_S` 변수만 쓰는 경우
>팀원 분들 작업  follow-up이 잘 안 되고 있음...😭 일단 할 수 있는 건 다 해두고 미팅 때 만나서 확인 후 빠르게 sync 맞추자. 
- 상준님 feature로 사용
- 봉훈님
	- (DF2_pr) 결측치 1000개 정도 됨
		- 
### Todo 
- [ ] `depressed == 0 & DF2_pr == 1`인 사람들의 치료 여부(`DF2_pt`) 관련 논의
- [ ] 결측치 처리 (KNN 등)
- [ ] 이상치 확인
- [ ] feature 다시 정리해서 선택
- [ ] 너무 직접적인 영향을 주는 변수들 -> 확인 필요함
### 지난 주 금요일 논의 내용

**y 라벨 결정하기**

1. DF2_pr == 1 변수만 쓰는 경우

- 문제점: DF2_pr == 비해당(8)인 경우가 너무 많음.
    - 실질적으로 DF2_pr == 1 수도 있는데, 의사 진단을 받지 않아서 본인이 우울증인 것을 모르는 경우

2. DF2_pr ==1 변수와 mh_PHQ_S를 섞어서 활용

- mh_PHQ_S >= 5 도 DF2_pr ==1로 판단하려면, 다음을 고려해야 함
    - DF2_pr ==1 인 사람들은 전부 다 mh_PHQ_S >=5 인가?
    - 현재 치료 여부도 고려해주어야 함
        - DF2_pr ==1인데, ‘현재치료여부’ == 1이면, mh_PHQ_S < 5 일 수 있음. (약을 먹어서 현재 본인이 느끼기에는 우울정도가 낮은 경우)
- 이후 새로운 y 변수 생성하기

**3) mh_PHQ_S 변수만 쓰는 경우**

- 기준
    - 10점 이상인 사람 = 1257명 (중증) 5%
    - 5점 이상인 사람 = 4309명 (경증) 20%
- 문제점
    - 2020, 2018, 2016, 2014 일부 년도에서만 실행됨.
    - 해당 변수 non-null 수 21,977명로 데이터 자체가 적은 편
    - 완전 정상인 사람들과 완전 우울한 사람들만 모아야 함
        - 확실하게 우울한 사람: DF2_pr == 1
        - 확실하게 정상인 사람: DF2_pr == 0 or mh_PHQ_S < 5
        - 잠재적으로 우울할지도 모르는 사람을 제외해야함 (학습에 방해)
            - DF2_pr == 0 & mh_PHQ_S == NaN
                - 현재 우울증으로 진단 받지는 않았지만, mh_PHQ_S == NaN 인 사람들이 사실 잠재적인 우울한 사람일지 모름
    - 기준을 어떻게 세울 지 결정해야 함
        - **1) Binary로 기준 세우기**
            - **5점으로 할 지, 10점으로 할 지만 결정**
    - 추가 작업 가능
        - BP_PHQ_n 변수를 직접 합산해서 PHQ_sum 변수를 만드는 방법
            - 2)보다 조금 더 데이터를 늘릴 수 있음
        - 4점 미만, 5~9점, 10점 이상으로 구간 나누기
            - 각 구간별 y값 조정해주어야 함

**결측값 처리하기**

- **결측치 어떻게 처리할 지**
    - 1. 가로 기준 - 어떤 사람1이 모든 응답 중에서 10% 이상 null로 대답한 경우 해당 row를 drop
    - 2. 세로 기준 - column 결측치 30% 이상인 경우 해당 column을 drop
- **결측값 처리**
    - 1. [9, 99, 999] = null로 처리
    - 2. NaN + [9, 99, 999] 들 처리해주기
        
        - KNN 방법
    - 




---

---
### 데이터셋

- - 대한민국 질병관리청 국민건강영양조사 제 1기~8기 데이터셋
    - [https://poweb.page.link/EAyxyN4gki6V5Kuw5](https://poweb.page.link/EAyxyN4gki6V5Kuw5) 데이터셋 설명 (hwp - 폴라리스 웹뷰어)
    - [https://knhanes.kdca.go.kr/knhanes/sub03/sub03_02_05.do](https://knhanes.kdca.go.kr/knhanes/sub03/sub03_02_05.do) 원시 데이터 다운로드 링크
- (급한대로) `mh_PHQ_S`변수는 14,16,18,20년도에만 실시되었기 때문에, 해당 년도의 DB만 https://knhanes.kdca.go.kr/knhanes/sub03/sub03_02_05.do 에서 다운 받아 사용함.

### 선택한 변수 집어 넣기

- `features`: 이용지침서를 보고 선택한 변수를 raw dataset에서 빼둔다.
- 이때, 내가 만든 데이터셋에는 포함되지 않은 변수들을 따로 제외하고, 추가적으로 features에 포함되어 있지 않은 y 후보 변수들(`mh_PHQ_S`, `DF2_pr`)을 넣어준다.

### 데이터에서 청소년,소아를 제외하기

우울증 유병 여부나 PHQ 설문은 만 19세 이상인 성인만을 대상으로 시행되었기 때문에, further analysis 전에 미리 데이터에서 만 19세 미만인 사람을 제외한다.
```python
# Filter only adult data using `age` variable
df = df[df['age'] >= 19]
```

### 결측치 처리 - (1)

기존에 생각한 우선적인 결측치 처리 방법은 아래 2가지임.
1) row 기준, 모든 column의 응답에서 30% 이상을 결측치로 응답한 사람은 drop한다.
	-> 10% 이상으로 잡으면 1627개만 남음😱
2) column 기준 결측치를 drop할 필요가 지금 단계에서 있을까?
	-> 일단 skip하고, 팀원 분들이 한 방법 체크해서 다시 시도한다. 가로 기준으로 처리하면 이 경우로 drop하는 경우가 그렇게 많을까 싶긴 한데, 봉훈 님께서 보여주신 그래프로 확인하는 걸로 시각화하면 좋을 것 같다. -> 나중에 보니 아직도 많다...drop 해야 함
```python
threshold = 0.3 # threshold for null proportion
null_percent = df.isnull().mean(axis = 1) # 각 행에 대한 결측값 비율 계산
df = df[null_percent < threshold]
```
이 상태에서, shape은 (21114, 511) 👍(not bad)


### y 변수 

- 건강설문 이환 - 우울증 (p.116)
	- ![[Pasted image 20231130205900.png]]
- `mh_PHQ_S` 변수란?
	- 한글판 PHQ-9 (Patient Health Questionnaire-9) 검사를 평가 도구로 쓰고, 각 문항의 접수를 합산한 변수 
	- 문항은 수검자 본인이 직접 읽고 기입한 자기 기입식 설문 조사
	- 국가건강검진에서 PHQ-9 검사 결과 **9번 문항에서 1점 이상으로 응답한 경우** 및 **총점이 10점 이상인 경우**, 일상생활에서 우울증을 극복할 수 있는 방법을 담은 <우울 증상과 극복 방법>을 제공한다. 또한 가까운 병/의원 또는 정신건강 복지센터에 의뢰해 신속하게 진료 및 치료를 받을 수 있도록 안내하고, 정신건강 위기 상담 전화를 통해 도움받을 수 있는 방법도 함께 안내한다.
		- ❗일단 10점을 threshold로 잡아서 진행하되, 팀원 분들 작업 하신것 보고 빠르게 5점 등으로 수정 가능함!
		- ❗이 변수를 쓴다는 것은, 의사의 판단에 상관없이 본인이 우울하다고 느끼면 우울증 또는 위험군이라고 정의하는 것이다. (이건 ok인 것으로 생각, 실제로 PHQ-9이 약식 검사로 활용될 수 있기 때문)
		- ![[Pasted image 20231130210800.png]]
		![[Pasted image 20231130210815.png]]

## y 변수 케이스 3)에서 확인해야 할 것

### 1) `depressed`를 정의하기

- `BP_PHQ_1` ~`BP_PHQ_9`은 null값이 1,629개인 반면 , `mh_PHQ_S` 는 null값이 9,074나 된다.
- 다만, `BP_PHQ_1` ~`BP_PHQ_9`null값이 아니어도, 각 항목에 8(비해당), 9(모름/무응답)을 선택한 경우는 유효하지 않다. 
- 따라서, 이미 생성된 `mh_PHQ_S`를 활용하되, `BP_PHQ_9`에서 1점 이상을 선택한 사용자도 선별해서 우울증 위험군으로 분류한다.
```python
# Define depression (dependent variable)
cond1 = df['mh_PHQ_S'] >= 10 # total score
cond2 = df['BP_PHQ_9'].isin([1, 2, 3]) # person who chose 1, 2, 3 in 9th question
depressed = df[cond1 | cond2].drop_duplicates() # person with cond1 or cond2
```
```python
# Create a new variable 'depressed'
df.loc[cond1 | cond2, 'depressed'] = 1 # 두 조건 중 하나 이상을 만족하는 행에 'depressed' 변수에 1 값
df.loc[~(cond1 | cond2), 'depressed'] = 0 # 두 조건 중 하나라도 만족하지 않는 행에 0 값
```
- `depressed` 비중 확인
```python
# Check distribution of who those are depressed or not
df.groupby('depressed').count()
```

```python
# Check distribution of who those are depressed or not with two variables `depressed` and `DF2_pr`
df.groupby(['depressed', 'DF2_pr']).count()
```


### 2) 학습에 방해가 될 수 있는 친구들 찾기
[[231130 프로젝트 1 - 우울증 y 라벨]]
![[Pasted image 20231202234424.png]]

위에서 `depressed` == 1의 기준을, 1) PHQ 설문 총점이 10점 이상이거나 or 2) PHQ-9 문항의 응답이 1점 이상인 경우(1,2,3)로 잡았기 때문에, `depressed` == 0 에는 온갖 사람들이 다 들어가 있다.
- 특히, `mh_PHQ_S == NaN` 를 주목해야 함.
	- 내가 맡은 y 라벨 3번 케이스에서는, `DF2_pr` 변수를 함께 사용하고 있진 않지만, 일단 함께 고려해서 생각해보자. (강사님 피드백도 있었던 부분: "현재 우울증으로 진단 받지는 않았지만, 잠재적인 우울한 사람을 골라내야 학습에 방해를 주지 않을것이다"") 
	- 딥러닝 모델을 제대로 학습시키기 위해서는, 극단값 제공을 잘 해주어야 한다고 함. 즉, 확실하게 우울한 사람과 확실하게 우울하지 않은 사람의 데이터를 학습시켜야 함. 
	- **여기까지 하니까, 결국은 `DF2_pr` 혹은 `DF2_dg`변수를 함께 고려해야 할 것 같은데..? 2번 케이스 작업을 확인해보자.
	
	- 일단, ==잠재적으로 우울한 사람의 경우==를 생각해보자.
		- 1) 일단 의사에게 진단을 받은 적이 없음. (`DF2_pr == 8`)
			- 1-1) `mh_PHQ_S == NaN` : PHQ 설문도 응답을 하지 않았기 때문에 이 사람들에 대해서는 아예 정보가 없음. -> drop 해야 함.
			- 1-2) `mh_PHQ_S != NaN`: 이 사람들은 괜찮음. 어떻게던 분류가 될 것
		- 2) `DF2_pr & mh_PHQ_S == NaN`: 우울증에 관련한 정보가 아예 없는 사람
	- 위 2가지 케이스 모두 데이터셋에서 제외해야 함. 
```python
# people who might be depressed

# DF2_pr == 0 & mh_PHQ_S == NaN
# 일단, mh_PHQ_S == NaN인 경우를 분리하고 생각하자 (나중에 drop할때에는 원래 df에서 제외해버리면 됨)
null_PHQ_S = df[pd.isnull(df['mh_PHQ_S'])]

# 1) DF2_pr == 8 (never been diagnosed by doctor) & mh_PHQ_S == NaN
null_PHQ_S[null_PHQ_S['DF2_pr'] == 8].shape # 213개 있음
# 2) DF2_pr & mh_PHQ_S == NaN`: 우울증에 관련한 정보가 아예 없는 사람
null_PHQ_S[['DF2_pr', 'BP_PHQ_9']].isnull().sum() # DF2_pr, BP_PHQ_9(혹시 모르니까)도 null인 케이스를 세보자 # 없음!
```

```python
# Remove people who might be depressed
# DF2_pr == 8 (never been diagnosed by doctor) & mh_PHQ_S == NaN) 인 경우를 제외하려면?
# DF2_pr != 8 or mh_PHQ != NaN (전부 다 ~ 취하기)
df = df[(df['DF2_pr'] != 8 |  ~df['mh_PHQ_S'].isna() )]
```

- [`depressed == 0 (mh_PHQ_S != NaN) & DF2_pr == 1`인 사람들의 치료 여부(`DF2_pt`) 파악](#치료-여부)
	- ([[#y 변수]]) 정의 다시 확인하기
	- ![[Pasted image 20231203003116.png]]
	- 1) DF2_pt == 1: 원래 우울해야 하는데 우울증 치료를 받고 있어서, 우울하지 않은 사람
		- ❗==어떻게 처리하지? ==
			- i) 학습에 방해가 될 수 있다고 판단하고 제외하기
			- ii) 그냥 우울하지 않은 것으로 취급하기 -> 현재 위험 상태는 아닌 것이니, 괜찮지 안을까? 
	- 2) DF2_pt == 0 or 8 or 9: 치료 받고 있지 않거나, 진단 받은 적이 없거나, 응답하지 않은 사 

### 결측치 처리 - (2)

- column 기준으로 모든 행에 대한 결측값이 10% 이상이면 제외하기
```python
# Remove columns if percentage of NaN exceeds 0.1
threshold = 0.1  

null_percent = df.isnull().mean() # 각 열에 대한 결측값 비율 계산
selected_cols = null_percent[null_percent < threshold].index
df_filtered = df[selected_cols]
df_filtered.shape # (20492, 314)
```
[[#y 변수]]

