"""
This module defines base data processing operations
"""

import torch
import warnings
import pymorphy2
import numpy  as np
import pandas as pd
from tqdm import tqdm
from langdetect import detect
from sklearn.metrics import r2_score
from nltk.stem import WordNetLemmatizer 
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

class IQMethod():

    def __init__(self, data, column):
        """
        Method for initializing class. Initialization requires only two parameters:
        [pandas.DataFrame]: data (data for outliers cleaning);
        [string]: column (column on which processing will be).
        """

        self.data = data

        if not isinstance(column, str):
            raise ValueError("'column' should be string")
        else:
            self.column = column

    def apply(self):
        """
        Method is intended for cleaning inputted data using values of inputted column. Base is interquartile method.
        Return:
        [pandas.DataFrame]: cleaned data.
        """

        self.datalength = self.data.shape[0]
        self.iqr = self.data[self.column].quantile(0.75) - self.data[self.column].quantile(0.25)
        self.q1 = self.data[self.column].quantile(0.25)
        self.q3 = self.data[self.column].quantile(0.75)

        self.lowLimit = self.q1 - 1.5 * self.iqr
        self.highLimit = self.q3 + 1.5 * self.iqr

        self.outliersIndex = self.data[(self.data[self.column] <= self.lowLimit) | (self.data[self.column] >= self.highLimit)].index
        self.newData = self.data.drop(self.outliersIndex, axis=0)
        self.newData.reset_index(drop=True)
        self.outlierPercentage = round(len(self.outliersIndex) / self.datalength * 100, 5)
        print(f"Percentage of outliers that was dropped: {self.outlierPercentage} %")

        return self.newData, self.outlierPercentage

def inverseBoxCox(x, l):
    """
    Method realizes inverse Box Cox transformation.
    Input:
    [int, float]: x (value for inverse transformation);
    [float]: l (necessary parameter for inverse transformation).
    """

    return round(np.exp(np.log(l * x + 1) / l))

class Embeddings():

    def __init__(self, data, languageModel, column, device, maxLen=128):
        """
        Method for initializing class. Initialization requires following parameters:
        [pandas.DataFrame]: data (data that contains column for creation text embeddings);
        [string]: model_name (name of language model);
        [string]: column (column for which text embeddings will be calculated);
        [torch.device]: device (device which takes all processing load);
        [int]: maxLen (restriction of an embedding vector max length).
        """

        self.data = data
        self.device = device

        if not isinstance(languageModel, str):
            raise ValueError("'languageModel' should be string")
        else:
            self.languageModel = languageModel
        
        if not isinstance(column, str):
            raise ValueError("'column' should be string")
        else:
            self.column = column

        if not isinstance(maxLen, int):
            raise ValueError("'maxLen' should be string")
        else:
            self.maxLen = maxLen
            
    def apply(self):
        """
        Method is intended for calculating text embeddings and
        conversion it results into pandas dataframe.
        Return:
        [pandas.DataFrame]: data with text embeddings features.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(self.languageModel)
        self.model = AutoModel.from_pretrained(self.languageModel).to(self.device)
        
        self.colName = self.languageModel.split("/")[1]
        self.featues = []

        for text in tqdm(self.data[self.column]):
            self.encoded = self.tokenizer([text], padding="max_length", add_special_tokens=True,
                                          truncation=True, max_length=self.maxLen, return_tensors='pt')

            with torch.no_grad():
                self.output = self.model(input_ids=self.encoded['input_ids'].to(self.device))

            self.featues.append(self.output[0].detach().to("cpu").sum(dim=1).numpy()[0])

        self.res = pd.DataFrame(columns = [f"{self.colName}_f{i}" for i in range(len(self.featues[0]))],
                                data=self.featues)
        
        return self.res

def createTimeFeatures(data):
    """
    Method is intended for creating time features. Method requires following parameters:
    [pandas.DataFrame]: data (data for which it's necessary to create time features);
    [string]: column (column that contains datetime values).
    """

    data["created"] = pd.to_datetime(data["created"])
    data["time"] = data["created"].apply(lambda x: x.time())
    data["month"] = data["created"].apply(lambda x: x.month)
    data["week"] = data["created"].apply(lambda x: x.week)
    data["day"] = data["created"].apply(lambda x: x.day)
    data["dayOfWeek"] = data["created"].apply(lambda x: x.dayofweek)
    data["day_year"] = data["created"].apply(lambda x: x.dayofyear)
    data["year"] = data["created"].apply(lambda x: x.year)
    data["hour"] = data["time"].apply(lambda x: x.hour)

    return data

def createAuxiliaryFeatures(data):
    """
    Method is intended for creating auxiliary features. Method requires following parameters:
    [pandas.DataFrame]: data (data for which it's necessary to create Auxiliary features).
    """

    data["len1"] = data["summary"].apply(len)
    data["len2"] = data["summary"].str.split(" ").apply(len)
    data["key_val"] = data["key"].str.split("-").apply(lambda x: x[1]).astype("int16")
    data["is_the_same"] = (data["assignee_id"] == data["creator_id"]).astype("int8")

    data.drop(["key"], axis=1, inplace=True)

    return data

def frequencyTables(dataframe):
    """
    Method is intended for creating frequency features
    Return:
    [pandas.DataFrame]: data with new features. 
    """

    projectsInMonth = dataframe.groupby(["project_id", "month"]).count().reset_index()\
    .iloc[:, :3].rename(columns={"id": "amnt_projectsInMonth"})
    projectsInMonth["amnt_projectsInMonth"] = np.log1p(projectsInMonth["amnt_projectsInMonth"])
    dataframe = dataframe.merge(projectsInMonth, on=["project_id", "month"], how="left")

    projectsInMonthByAssignee = dataframe.groupby(["project_id", "month", "assignee_id"]).count().reset_index()\
    .iloc[:, :4].rename(columns={"id": "amnt_projectsInMonthByAssignee"})
    projectsInMonthByAssignee["amnt_projectsInMonthByAssignee"] = np.log1p(projectsInMonthByAssignee["amnt_projectsInMonthByAssignee"])
    dataframe = dataframe.merge(projectsInMonthByAssignee, on=["project_id", "month", "assignee_id"], how="left")

    projectsInMonthByCreator = dataframe.groupby(["project_id", "month", "creator_id"]).count().reset_index()\
    .iloc[:, :4].rename(columns={"id": "amnt_projectsInMonthByCreator"})
    projectsInMonthByCreator["amnt_projectsInMonthByCreator"] = np.log1p(projectsInMonthByCreator["amnt_projectsInMonthByCreator"])
    dataframe = dataframe.merge(projectsInMonthByCreator, on=["project_id", "month", "creator_id"], how="left")

    projectsInWeekDay = dataframe.groupby(["project_id", "dayOfWeek"]).count().reset_index()\
    .iloc[:, :3].rename(columns={"id": "amnt_projectsInWeekDay"})
    projectsInWeekDay["amnt_projectsInWeekDay"] = np.log1p(projectsInWeekDay["amnt_projectsInWeekDay"])
    dataframe = dataframe.merge(projectsInWeekDay, on=["project_id", "dayOfWeek"], how="left")

    projectsInWeekDayByAssignee = dataframe.groupby(["project_id", "dayOfWeek", "assignee_id"]).count().reset_index()\
    .iloc[:, :4].rename(columns={"id": "amnt_projectsInWeekDayByAssignee"})
    projectsInWeekDayByAssignee["amnt_projectsInWeekDayByAssignee"] = np.log1p(projectsInWeekDayByAssignee["amnt_projectsInWeekDayByAssignee"])
    dataframe = dataframe.merge(projectsInWeekDayByAssignee, on=["project_id", "dayOfWeek", "assignee_id"], how="left")

    projectsInWeekDayByCreator = dataframe.groupby(["project_id", "dayOfWeek", "creator_id"]).count().reset_index()\
    .iloc[:, :4].rename(columns={"id": "amnt_projectsInWeekDayByCreator"})
    projectsInWeekDayByCreator["amnt_projectsInWeekDayByCreator"] = np.log1p(projectsInWeekDayByCreator["amnt_projectsInWeekDayByCreator"])
    dataframe = dataframe.merge(projectsInWeekDayByCreator, on=["project_id", "dayOfWeek", "creator_id"], how="left")

    projectsInHour = dataframe.groupby(["project_id", "hour"]).count().reset_index()\
    .iloc[:, :3].rename(columns={"id": "amnt_projectsInHour"})
    projectsInHour["amnt_projectsInHour"] = np.log1p(projectsInHour["amnt_projectsInHour"])
    dataframe = dataframe.merge(projectsInHour, on=["project_id", "hour"], how="left")

    projectsInHourByAssignee = dataframe.groupby(["project_id", "hour", "assignee_id"]).count().reset_index()\
    .iloc[:, :4].rename(columns={"id": "amnt_projectsInHourByAssignee"})
    projectsInHourByAssignee["amnt_projectsInHourByAssignee"] = np.log1p(projectsInHourByAssignee["amnt_projectsInHourByAssignee"])
    dataframe = dataframe.merge(projectsInHourByAssignee, on=["project_id", "hour", "assignee_id"], how="left")

    projectsInHourByCreator = dataframe.groupby(["project_id", "hour", "creator_id"]).count().reset_index()\
    .iloc[:, :4].rename(columns={"id": "amnt_projectsInHourByCreator"})
    projectsInHourByCreator["amnt_projectsInHourByCreator"] = np.log1p(projectsInHourByCreator["amnt_projectsInHourByCreator"])
    dataframe = dataframe.merge(projectsInHourByCreator, on=["project_id", "hour", "creator_id"], how="left")

    projectInYear = dataframe.groupby(["project_id", "year"]).count().reset_index()\
    .iloc[:, :3].rename(columns={"id": "amnt_projectInYear"})
    projectInYear["amnt_projectInYear"] = np.log1p(projectInYear["amnt_projectInYear"])
    dataframe = dataframe.merge(projectInYear, on=["project_id", "year"], how="left")

    projectInYearByAssignee = dataframe.groupby(["project_id", "year", "assignee_id"]).count().reset_index()\
    .iloc[:, :4].rename(columns={"id": "amnt_projectInYearByAssignee"})
    projectInYearByAssignee["amnt_projectInYearByAssignee"] = np.log1p(projectInYearByAssignee["amnt_projectInYearByAssignee"])
    dataframe = dataframe.merge(projectInYearByAssignee, on=["project_id", "year", "assignee_id"], how="left")

    projectInYearBycreator = dataframe.groupby(["project_id", "year", "creator_id"]).count().reset_index()\
    .iloc[:, :4].rename(columns={"id": "amnt_projectInYearBycreator"})
    projectInYearBycreator["amnt_projectInYearBycreator"] = np.log1p(projectInYearBycreator["amnt_projectInYearBycreator"])
    dataframe = dataframe.merge(projectInYearBycreator, on=["project_id", "year", "creator_id"], how="left")

    dataframe.drop(["month", "dayOfWeek", "hour", "year"], axis=1, inplace=True)
    
    return dataframe

def diffFeaturesByGroups(dataframe):
    """
    Method is intended for calculating difference features grouped by projects&assignee and projects&creator.
    Return:
    [pandas.DataFrame]: data with new features. 
    """
   
    uniqueProjects = dataframe["project_id"].unique()
    uniqueAssignee = dataframe["assignee_id"].unique()
    uniqueCreator = dataframe["creator_id"].unique()

    dataframe["diff1ByAssignee"], dataframe["diff2ByAssignee"] = 0, 0
    dataframe["diff3ByAssignee"], dataframe["diff4ByAssignee"], dataframe["diff5ByAssignee"] = 0, 0, 0

    for project in tqdm(uniqueProjects):
        for assignee in uniqueAssignee:

            indexes = dataframe[dataframe["project_id"].isin([project]) & dataframe["assignee_id"].isin([assignee])].index
            if len(indexes) != 0:

                for idx in range(len(indexes)-1):
                    dataframe["diff1ByAssignee"].loc[indexes[idx+1]] = (dataframe["created"].loc[indexes[idx+1]] -\
                                                                        dataframe["created"].loc[indexes[idx]]).seconds

                for idx in range(len(indexes)-2):
                    dataframe["diff2ByAssignee"].loc[indexes[idx+2]] = (dataframe["created"].loc[indexes[idx+2]] -\
                                                                        dataframe["created"].loc[indexes[idx]]).seconds
                for idx in range(len(indexes)-3):
                    dataframe["diff3ByAssignee"].loc[indexes[idx+3]] = (dataframe["created"].loc[indexes[idx+3]] -\
                                                                        dataframe["created"].loc[indexes[idx]]).seconds

                for idx in range(len(indexes)-4):
                    dataframe["diff4ByAssignee"].loc[indexes[idx+4]] = (dataframe["created"].loc[indexes[idx+4]] -\
                                                                        dataframe["created"].loc[indexes[idx]]).seconds

                for idx in range(len(indexes)-5):
                    dataframe["diff5ByAssignee"].loc[indexes[idx+5]] = (dataframe["created"].loc[indexes[idx+5]] -\
                                                                        dataframe["created"].loc[indexes[idx]]).seconds

    dataframe["diff1ByCreator"], dataframe["diff2ByCreator"]  = 0, 0
    dataframe["diff3ByCreator"], dataframe["diff4ByCreator"], dataframe["diff5ByCreator"] = 0, 0, 0

    for project in tqdm(uniqueProjects):
        for creator in uniqueCreator:

            indexes = dataframe[dataframe["project_id"].isin([project]) & dataframe["creator_id"].isin([creator])].index
            if len(indexes) != 0:

                for idx in range(len(indexes)-1):
                    dataframe["diff1ByCreator"].loc[indexes[idx+1]] = (dataframe["created"].loc[indexes[idx+1]] -\
                                                                       dataframe["created"].loc[indexes[idx]]).seconds

                for idx in range(len(indexes)-2):
                    dataframe["diff2ByCreator"].loc[indexes[idx+2]] = (dataframe["created"].loc[indexes[idx+2]] -\
                                                                       dataframe["created"].loc[indexes[idx]]).seconds
                for idx in range(len(indexes)-3):
                    dataframe["diff3ByCreator"].loc[indexes[idx+3]] = (dataframe["created"].loc[indexes[idx+3]] -\
                                                                       dataframe["created"].loc[indexes[idx]]).seconds

                for idx in range(len(indexes)-4):
                    dataframe["diff4ByCreator"].loc[indexes[idx+4]] = (dataframe["created"].loc[indexes[idx+4]] -\
                                                                       dataframe["created"].loc[indexes[idx]]).seconds

                for idx in range(len(indexes)-5):
                    dataframe["diff5ByCreator"].loc[indexes[idx+5]] = (dataframe["created"].loc[indexes[idx+5]] -\
                                                                       dataframe["created"].loc[indexes[idx]]).seconds
    
    return dataframe

def diffFeatures(dataframe):
    """
    Method is intended for calculating difference features.
    Return:
    [pandas.DataFrame]: data with new features. 
    """
    
    uniqueProjects = dataframe["project_id"].unique()

    dataframe["diff1"], dataframe["diff2"] = 0, 0
    dataframe["diff3"], dataframe["diff4"], dataframe["diff5"] = 0, 0, 0

    for project in tqdm(uniqueProjects):

        indexes = dataframe[dataframe["project_id"].isin([project])].index
        if len(indexes) != 0:

            for idx in range(len(indexes)-1):
                    dataframe["diff1"].loc[indexes[idx+1]] = (dataframe["created"].loc[indexes[idx+1]] -\
                                                              dataframe["created"].loc[indexes[idx]]).total_seconds()

            for idx in range(len(indexes)-2):
                    dataframe["diff2"].loc[indexes[idx+2]] = (dataframe["created"].loc[indexes[idx+2]] -\
                                                              dataframe["created"].loc[indexes[idx]]).total_seconds()
            for idx in range(len(indexes)-3):
                    dataframe["diff3"].loc[indexes[idx+3]] = (dataframe["created"].loc[indexes[idx+3]] -\
                                                              dataframe["created"].loc[indexes[idx]]).total_seconds()

            for idx in range(len(indexes)-4):
                    dataframe["diff4"].loc[indexes[idx+4]] = (dataframe["created"].loc[indexes[idx+4]] -\
                                                              dataframe["created"].loc[indexes[idx]]).total_seconds()

            for idx in range(len(indexes)-5):
                    dataframe["diff5"].loc[indexes[idx+5]] = (dataframe["created"].loc[indexes[idx+5]] -\
                                                              dataframe["created"].loc[indexes[idx]]).total_seconds()
    
    return dataframe

def posFreqEstimation(data):
    """
    Method is intended for position frequency estimation taking in account positions of assignee and positions of creator.
    Return:
    [pandas.DataFrame] data with new features.
    """

    posOfAssigneeInProject = data.groupby(["project_id", "assignee_id", "pos"]).count().reset_index()\
    .iloc[:, :4].rename(columns={"id": "amnt_posOfAssignee_in_project"})
    posOfCreatorInProject = data.groupby(["project_id", "creator_id", "pos"]).count().reset_index()\
    .iloc[:, :4].rename(columns={"id": "amnt_posOfCreator_in_project"})
    data = data.merge(posOfAssigneeInProject, on=["project_id", "assignee_id", "pos"], how="left")
    data = data.merge(posOfCreatorInProject, on=["project_id", "creator_id", "pos"], how="left")

    return data

def coeffPositionsInProjects(data):
    """
    Method is intended for calculation position coefficient in project
    [pandas.DataFrame] data with new features.
    """

    coeffPositionsInProjects = data.groupby(["project_id", "pos"]).count().reset_index()\
    .iloc[:, :3].rename(columns={"id": "coeffPositionsInProjects"})

    uniqueProjects = coeffPositionsInProjects["project_id"].unique()

    for project in uniqueProjects:
        
        val = data[data["project_id"].isin([project])].shape[0]
        indexes = coeffPositionsInProjects[coeffPositionsInProjects["project_id"].isin([project])].index
        coeffPositionsInProjects["coeffPositionsInProjects"].loc[indexes] = coeffPositionsInProjects["coeffPositionsInProjects"]\
        .loc[indexes] / val
        
    data = data.merge(coeffPositionsInProjects, on=["project_id", "pos"], how="left")

    return data


def validateModel(model, data, target, fitParams, nFold=5, KFold_randomState=42):
    """
    Method is intended for validating model.
    Return:
    [none].
    """

    kFold = KFold(n_splits=nFold, shuffle=True, random_state=KFold_randomState)

    X = data.copy()
    y = X[target]
    X.drop([target], axis=1, inplace=True)
    print(f"[X]: {X.shape} \n[y]: {y.shape}")

    cv = []
    for i, idx in enumerate(kFold.split(X)):
    
        X_train, X_valid = X.loc[idx[0]], X.loc[idx[1]]
        y_train, y_valid = y.loc[idx[0]], y.loc[idx[1]]

        try:
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], **fitParams)
        except:
            model.fit(X_train, y_train)
    
        r2 = r2_score(y_valid, model.predict(X_valid))
        cv.append(r2)
        print(f"[{i}]: {r2}")

    print(f"[mean]: {np.mean(cv)}")

def lemmatizeSummary(data):
    """
    Method is intended for providing basic processing which is lemmatizing for 'summary' column.
    Return:
    [pandas.DataFrame] data with lemmatized 'summary' column.
    """

    morph = pymorphy2.MorphAnalyzer()
    lemmatizer = WordNetLemmatizer()

    data["summary_l"] = 0
    for idx, seq in tqdm(enumerate(data["summary"])):
        if detect(seq) == "en":
            data["summary_l"].loc[idx] = " ".join([lemmatizer.lemmatize(word) for word in seq.split(" ")])
        elif detect(seq) == "ru":
            data["summary_l"].loc[idx] = " ".join([morph.parse(word)[0].normal_form for word in seq.split(" ")])
        else:
            data["summary_l"].loc[idx] = seq

    return data