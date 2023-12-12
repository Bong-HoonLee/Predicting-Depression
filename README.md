## Predicting Depression Using Neural Network and ML Models
- Purpose of the project
	- In a society where there are still psychological and social hurdles to seeking counselling for depression diagnosis, the aim is to predict the risk of depression based on lifestyle habits or regular health checkup results, and subsequently provide guidance on treatment or diagnosis to help the patient. Additionally, identifying factors which might significantly influence depression diagnosis can be helpful to assist in treatment and symptom improvement.
- Research Questions
  1) To predict depression using daily life questionnaires and basic health screening results (Binary Classification)
  2) To find significant features regarding the prediction of depression (Feature Importances)

### Team
- 

### Directory
- `archive`: EDA, data pre-processing
- `bin`
- `config`: 
- `doc`
- `models`: ANN models
- `rf_clf`: Random Forest models
- `requirements.txt`: required libraries and packages 
- `trainer.py`: main train&test logics

### Dataset
- Source: Korea National Health & Nuturition Examination Survey ([link](https://knhanes.kdca.go.kr/knhanes/sub03/sub03_01.do))
	- Only respondents aged over 19 are selected 
	- Respondents are chosen from the population with 0.0002 extraction rate
- Train & Test Set
	- Train Set: survey of 2007 ~ 2019, and 2021 
	- Test Set: survey of 2020
- Features (independent variables, X)
	- 125 features are selected from over 800 features 
		- including the results of multiple blood tests, urine tests to intensity of daily workouts, education level, and even whether to brush their teeth.
	- Tried to use as many as possible features but,
		1) features with a high proportion of NaN values are discarded. 
		2) features which can be representative of similar questions
		3) commonly used features among 2007 ~ 2021, selected year.
- Target (dependent variable, y)
	- `depressed` variables have been defined. `depressed` == 1 if:
		1) `mh_PHQ_S` >= 10 or # `mh_PHQ_S`: total score of PHQ self test
		2) `BP_PHQ_9`.isin([1, 2, 3]) or # `BP_PHQ_9`: 9th question of PHQ self-test, "Have you ever thought about suicide or hurting yourself this year?"
		3) `BP6_10` == 1 or  # `BP6_10`: "Have you ever thought about suicide this year?"
		4) `BP6_31` == 1 or # `BP6_31`: "Have you ever thought about hurting yourself this year?"
		5) `DF2_pr` == 1 & `mh_PHQ_S`== NaN # `DF2_pr==1`: currently experiencing depression and have been diagnosed by a doctor
	- A respondent with no information on depression-related variables has been removed rather than filling in missing values.
	- Reference of definition of depression: National Health Insurance ([link](https://www.nhis.or.kr/static/alim/paper/oldpaper/202001/sub/s01_02.html))

### Preprocessing
- Manual Processing
	- Data Loading, Train/Test Separation, Feature Selection and Grouping, Define Target Label, Drop NaNs -> output to csv
- In Pipeline
	- Fill NaNs (KNN, most frequent values), Encoding/Scaling, Under/Over-sampling
>excalidraw 사진 넣기

### Models
- Pre-processed data (.csv): [drive link](https://drive.google.com/drive/folders/1UjUa46Cx-X8-EDdWtWvQhg5gAbJgRlP3)
	- file name of the csv is corresponded with config.py
- Metrics: Accuracy, Precision, Recall, F1-score, Support, AUROC
- Best Model
- Comparison: Logistic Regression, Random Forest, ANN

### Feature Importances
- Top 5 features based on random forest - feature importances
	- `D_1_1`: Subjective health awareness
	- `ainc`: Monthly average household total income
	- `LQ4_00_1.0`: Presence of (physical) activity limitation (Yes)
	- `edu`: Educational level - Academic background
	- `D_2_1_1.0`: Recent experience of physical discomfort in the past 2 weeks (Yes)
- Excluding any of the above features didn't show a significant change in metrics with Random Forest Model.
- Extra work is needed with the Neural Network Model to identify significant features.

> 그래프 넣기


--------------------------------------------------------------------------
## Pytorch 모델 학습 과정
### Logical Flow


![pytorch-learning-logical-flow](doc/pytorch_learning_logical_flow.png)

### Pseudo Code
```
# 필요한 패키지 import

# hyperparameter 선언
# 선언한 hyperparameter 를 저장

# 데이터 불러오기
# dataset 만들기 & 전처리하는 코드
# dataloader 만들기

# AI 모델 설계도 만들기 (class)
	# init, forward 구현하기

# AI 모델 객체 생성 (과정에서 hyperparameter 사용)
# loss 객체 생성
# optimizer 객체 생성

# ------- 준비단계 끝 -------- 
# ------- 학습단계 시작 -------- 

# loop 돌면서 학습 진행
    # [epoch]을 학습하기 위해 batch 단위로 데이터를 가져와야함
    # 이 과정이 loop 로 진행
        # dataloader 가 넘겨주는 데이터를 받아서

        # ai 모델에게 넘겨주고
        # 출력물을 기반으로 loss 를 계산하고
        # loss 를 바탕으로 Optimization 을 진행

        # 특정 조건을 제시해서, 그 조건이 만족한다면 학습의 중간 과정을 확인
            # 평가를 진행
            # 보고 싶은 수치 확인(loss, 평가 결과 값, 이미지와 같은 meta-data 등)
            # 만약 평가 결과가 괜찮으면
                # 모델 저장
```

### Pseudo Code(k-fold)

```
# 패키지 import

# hyperparameter 설정
# hyperparameter 저장

# 데이터 불러오기
# 전처리하는 코드

# AI 모델 설계도 만들기 (class)
    # init, forward 구현하기

# ------- 준비단계 끝 --------
# ------- 학습단계 시작 --------

# 학습과 평가를 위한 kfold 객체 생성(n_splits)
# n_splits 만큼 AI 모델 객체 생성 (과정에서 hyperparameter 사용)

# loop 돌면서 학습 및 교차검증
    # kfold 객체로부터 train, validation 데이터를 받아옴
    # train, validation dataset 만들기
    # train, validation dataloader 만들기

    # loss 객체 생성
    # optimizer 객체 생성

    # train dataloader 로부터 데이터를 받아서
    # ai 모델에게 넘겨주고 loss 를 계산

    # validation dataloader 로부터 데이터를 받아서
    # (학습이 진행된) ai 모델에게 넘겨주고 loss_val 를 계산

    # loss, loss_val 을 기반으로 평가 진행
        # 보고 싶은 수치 확인(loss, 평가 결과 값, 이미지와 같은 meta-data 등)

# 평가 결과가 괜찮으면(조건 만족)
    # 모델 저장
```

<br>
<br>

## Pytorch 모델 추론 과정

![pytorch-inference-logical-flow](doc/pytorch_inference_logical_flow.png)

-------

## 프로그램 도식

![team4_project_01_flow](doc/team4_project_01_flow.png)

--------

## Runner Sample(launch.json)
[prerequisite]
pip install -r requirements.txt
```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": true
        },
        {
            "name": "Python: train",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}${pathSeparator}bin${pathSeparator}train",
            "console": "integratedTerminal",
            "justMyCode": true,
            "args": [
                "--mode=train",
                "--config-dir=config/20231211_final",
            ]
        },
        {
            "name": "Python: validate",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}${pathSeparator}bin${pathSeparator}train",
            "console": "integratedTerminal",
            "justMyCode": true,
            "args": [
                "--mode=validate",
                "--config-dir=config/20231211_final",
            ]
        },
        {
            "name": "Python: test",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}${pathSeparator}bin${pathSeparator}train",
            "console": "integratedTerminal",
            "justMyCode": true,
            "args": [
                "--mode=test",
                "--config-dir=config/20231211_final",
                "--config-name=train_X_231211_final_col_01_transformed",
                "--model-path=output/train_X_231211_final_col_01_transformed_202312120307.pth",
            ]
        }
    ]
}
```
