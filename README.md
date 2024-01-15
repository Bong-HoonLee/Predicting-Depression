## Predicting Depression Using Neural Network and ML Models
- Purpose of the project
	- In a society where there are still psychological and social hurdles to seeking counselling for depression diagnosis, the aim is to predict the risk of depression based on lifestyle habits or regular health checkup results, and subsequently provide guidance on treatment or diagnosis to help the patient. Additionally, identifying factors which might significantly influence depression diagnosis can be helpful to assist in treatment and symptom improvement.
- Research Questions
  1) To predict depression using daily life questionnaires and basic health screening results (Binary Classification)
  2) To find significant features regarding the prediction of depression (Feature Importances)

### Team
- [chirichidi](https://github.com/chirichidi)
- [Bong-HoonLee](https://github.com/Bong-HoonLee)
- [angie0bb](https://github.com/angie0bb)
- [hantoro123](https://github.com/hantoro123)


### Directory
- `archive`: EDA, data pre-processing
- `bin`: executable file(train) is located 
- `config`: setting files for model parameters
- `doc`: documents, images, reports
- `models`: ANN models
- `rf_clf`: Random Forest models
- `requirements.txt`: required libraries and packages 
- `trainer.py`: main train&test logics

### How to Run & Debug
1) `pip install -r requirements.txt` to install required packages
2) create `./.vscode/launch.json` and copy&paste launch.json at the very end of this readme.md
3) choose a config directory(for example, out best model is located at `./config/20231211_final/`) you want to test  (all of the config.py in the selected folder will be executed at once)
4) now, open `launch.json` to put the config directory path on `"args"`
5) then, choose a mode (train, validate(5-fold), test) on **RUN AND DEBUG**!


### Running Samples
1) Please download the preprocessed sample dataset(transformed 6 csv files) from this [link](https://drive.google.com/drive/folders/1UjUa46Cx-X8-EDdWtWvQhg5gAbJgRlP3) and place it in the ./data directory.
2) In this state, running the command `make sample-test` will sequentially perform model validation, training, and testing using the predefined sample Config File (./config/samples/20231212_transformed) and the files that were previously downloaded.

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
![Pre_processing](https://github.com/Bong-HoonLee/EST_wassup01_TEAM_4/assets/76639910/54a9fb38-71f9-4a13-a8e5-7c28efd41477)
- Manual Processing
	- Data Loading, Train/Test Separation, Feature Selection and Grouping, Define Target Label, Drop NaNs -> output to csv
- In Pipeline
	- Fill NaNs (KNN, most frequent values), Encoding/Scaling, Under/Over-sampling


### Models
- Pre-processed data(.csv): [drive link](https://drive.google.com/drive/folders/1UjUa46Cx-X8-EDdWtWvQhg5gAbJgRlP3)
	- file name of the csv is corresponded with config.py
	- csv files should be located on `./data` 
- Metrics: Accuracy, Precision, Recall, F1-score, Support, AUROC
- Best Model (ANN)
  	- config_path: `./config/20231211_final/config_col_01_transformed.py`
  	- run settings: `./bin/train --mode=validate --config-dir=config/20231211_final`
  	- config settings
  	  	1) Model=ANN
		2) Model Params=ModuleList(
		 (0): Linear(in_features=208, out_features=2, bias=True)
		 (1): BatchNorm1d(2)
		 (2): ReLU()
		 (2): Linear(in_features=2, out_features=1, bias=True)
		3) loss_function=BCEwithLogitsLoss,
		4) optimizer=Adam,
		5) lr=0.001
		6) epochs=50
		7) batch size = 128
		8) lr scheduler = ReduceLROnPlateau
		   ( 'mode': 'min', 'factor': 0.1, 'patience': 5)
  	- results
  	 ![image](https://github.com/Bong-HoonLee/EST_wassup01_TEAM_4/assets/76639910/0bae431d-13b6-4ac4-a7b0-4cd51261fcca)
	 ![image](https://github.com/Bong-HoonLee/EST_wassup01_TEAM_4/assets/76639910/f31752fc-3f9e-4df0-a887-2290869b4fff)

- Comparison: Logistic Regression, Random Forest, ANN
- Detailed results with Korean can be found on /doc/20231212_report/20231212_report.ipynb

### Feature Importances
- Top 5 features based on random forest - feature importances
	- `D_1_1`: Subjective health awareness (0.0439)
	- `ainc`: Monthly average household total income (0.0263)
	- `LQ4_00_1.0`: Presence of (physical) activity limitation (Yes) (0.0240)
	- `edu`: Educational level - Academic background (0.0229)
	- `D_2_1_1.0`: Recent experience of physical discomfort in the past 2 weeks (Yes) (0.0214)
- Excluding any of the above features didn't show a significant change in metrics with Random Forest Model.
- Extra work is needed with the Neural Network Model to identify significant features.


--------------------------------------------------------------------------
## Pytorch 모델 학습 과정
### Logical Flow


![pytorch-learning-logical-flow](./doc/pytorch_learning_logical_flow.png)

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

![pytorch-inference-logical-flow](./doc/pytorch_inference_logical_flow.png)

-------

## 프로그램 도식

![team4_project_01_flow](./doc/team4_project_01_flow.png)

--------

## Runner Sample(launch.json)
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
                "--target-config-name=train_X_231211_final_col_01_transformed",
                "--target-model-path=output/train_X_231211_final_col_01_transformed_202312120307.pth",
            ]
        }
    ]
}
```
