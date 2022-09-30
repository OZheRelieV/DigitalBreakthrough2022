"""
This module is intended for "employees.csv" data file processing
"""

import pandas as pd

class EmployeesPreparation():

    def __init__(self, path, naLimit=0.5):
        """
        Method for initializing class. Initialization requires:
        [string]: path (path to a data file);
        [float]: naLimit (limit of na content in data)
        """

        if not isinstance(path, str):
            raise ValueError("'path' should be string")
        else:
            self.path = path
            if "/" not in self.path:
                self.path = self.path.replace("\\", "/")

        if not isinstance(naLimit, float):
            raise ValueError("'naLimit' should be float")
        else:
            self.naLimit = naLimit
    
    def __load(self):
        """
        Private method for loading data file
        [return]: pandas.DataFrame 
        """

        return pd.read_csv(self.path)

    def apply(self):
        """
        Method includes following data processing operations:
        - 'na' dropping by limit value na in the data;
        - one-hot encoding such features as 'position' and 'hiring_type' after 'na' filling by 'none' value;
        - summing up all numerical features except 'sex' feature into new data feature 'rating'.
        [return]: pandas.DataFrame 
        """
        
        self.data = self.__load()

        self.tempDf = ((self.data.isna().sum() / self.data.shape[0]) > self.naLimit).to_frame(name="f")
        self.columnsToDrop = self.tempDf[self.tempDf["f"].isin([True])].index.to_list()
        self.columnsToDrop.append("full_name")
        self.columnsToDrop.append("passport") # ATTENTION. HERE WAS CHANGING
        self.data.drop(self.columnsToDrop, axis=1, inplace=True)
        print(f"{self.columnsToDrop} were dropped")

        self.data["position"].fillna("none", inplace=True)
        self.data["hiring_type"].fillna("none", inplace=True)
        self.data["payment_type"].fillna("none", inplace=True)

        self.data["pos"] = self.data["position"]
        self.data = pd.get_dummies(self.data, columns=["hiring_type", "position", "payment_type"], dtype='int8')
        self.data.drop(["hiring_type_none", "position_none"], axis=1, inplace=True)

        self.numCols = [col for col in list(self.data)[1:] if self.data[col].dtype != "object"]
        if "sex" in self.numCols:
            self.numCols.remove("sex")

        if self.data[self.numCols].isna().sum().sum() == 0:
            self.data["rating"] = 0
            for col in self.numCols:
                self.data["rating"] += self.data[col]
                self.data.drop(col, axis=1, inplace=True) # ATTENTION. HERE WAS CHANGING
        else:
            print("'na' is in data. Not all. \n!!!ATTENTION!!! Not all operations under data was performed")

        return self.data