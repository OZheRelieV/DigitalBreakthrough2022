"""
This module is intended for "comments.csv" data file processing
"""

import pandas as pd
import numpy as np
from scipy import stats

class CommentsPreparation():

    def __init__(self, path):
        """
        Method for initializing class. Initialization requires only one string parameter - path to the data file
        """

        if not isinstance(path, str):
            raise ValueError("'path' should be string")
        else:
            self.path = path
            if "/" not in self.path:
                self.path = self.path.replace("\\", "/")
    
    def __load(self):
        """
        Private method for loading data file
        [return]: pandas.DataFrame 
        """

        return pd.read_csv(self.path)

    def apply(self):
        """
        Method is intended for calculating frequency features and main statistics of the text length
        [return]: pd.DataFrame
        Note that the first return value is the dataframe 'byAuthor' and the second one is dataframe 'byIssue',
        the last one is dataframe that contains information about 'text' feature lengths.
        """

        self.dfComments = self.__load()

        self.byAuthor = self.dfComments.groupby("author_id").count().reset_index()[["author_id", "comment_id"]]
        self.byAuthor.rename(columns={"comment_id": "amnt_by_author", "author_id": "assignee_id"}, inplace=True)

        self.dfComments["text"] = self.dfComments["text"].str.lower()
        self.dfComments["len_text"] = self.dfComments["text"].apply(lambda x: len(x.split(" ")))
        self.dfComments["len_w"] = self.dfComments["text"].apply(len)

        self.temp1 = self.dfComments.groupby("author_id")["len_text"].max().to_frame().reset_index()\
        .rename(columns={"author_id": "assignee_id", "len_text": "text_max"})

        self.temp2 = self.dfComments.groupby("author_id")["len_text"].min().to_frame().reset_index()\
        .rename(columns={"author_id": "assignee_id", "len_text": "text_min"})

        self.temp3 = self.dfComments.groupby("author_id")["len_text"].median().to_frame().reset_index()\
        .rename(columns={"author_id": "assignee_id", "len_text": "text_med"}).astype("int32")

        self.temp4 = self.dfComments.groupby("author_id")["len_text"].mean().apply(np.ceil).to_frame().reset_index()\
        .rename(columns={"author_id": "assignee_id", "len_text": "text_mean"}).astype("int32")

        self.temp5 = self.dfComments.groupby("author_id")["len_text"].sum().to_frame().reset_index()\
        .rename(columns={"author_id": "assignee_id", "len_text": "text_sum"}).astype("int32")

        self.temp6 = self.dfComments.groupby("author_id")["len_text"].agg(stats.mode).to_frame().reset_index()\
        .rename(columns={"author_id": "assignee_id", "len_text": "text_mode"})
        self.temp6["text_mode"] = self.temp6["text_mode"].apply(lambda x: x[0][0])

        self.temp7 = self.dfComments.groupby("author_id")["len_w"].mean().apply(np.ceil).astype("int16")\
        .to_frame().reset_index().rename(columns={"author_id": "assignee_id"})

        self.byAuthor = self.byAuthor.merge(self.temp1, on="assignee_id", how="left").merge(self.temp2, on="assignee_id", how="left")\
        .merge(self.temp3, on="assignee_id", how="left").merge(self.temp4, on="assignee_id", how="left")\
        .merge(self.temp5, on="assignee_id", how="left").merge(self.temp6, on="assignee_id", how="left")\
        .merge(self.temp7, on="assignee_id", how="left")

        self.byIssue = self.dfComments.groupby("issue_id").count().reset_index()[["issue_id", "comment_id"]]
        self.byIssue.rename(columns={"comment_id": "amnt_by_issue", "issue_id": "id"}, inplace=True)

        return self.byAuthor, self.byIssue, self.dfComments[["issue_id", "len_w", "len_text"]].rename(columns={"len_w": "len_w_by_issue",
                                                                                                               "len_text": "len_t_by_issue"})