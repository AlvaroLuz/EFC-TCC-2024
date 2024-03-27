import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from log import Log

ENV = "/home/alvaro/tcc/experiments"
DATASETS_PATH = f"{ENV}/data/"

class Dataset(Log):
    #Create, read, delete, update
    #rea 
    def get_full_df(self, ds_name:str):#retorna os dados de um dataset
        ds_path = DATASETS_PATH+ds_name+"/"
        return pd.read_csv(ds_path+f"full-{ds_name.lower()}.csv", index_col=None)
    

    def get_split_df(self, ds_name:str, test_size = 0.3):#retorna os dados de um dataset
        ds_path = DATASETS_PATH+ds_name+"/"

        self.log(f"Reading file: {ds_path+f'full-{ds_name.lower()}.csv'}")
        df = pd.read_csv(ds_path+f"full-{ds_name.lower()}.csv", index_col=None)
        

        df[" Label"] = np.where(df[" Label"] != "BENIGN", 1, 0)
        df = df.drop("label", axis=1)
        # Replace infinite or large values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        self.log(f"Dataset head:\n{df.head()}")
        
        self.log("Splitting the dataset...") 
        labels= df[' Label']
        return train_test_split(
            df.drop(' Label', axis=1), labels, 
            random_state=42, stratify = labels, shuffle=True, test_size=test_size
        )

    def create_unified_csv(self, ds_name: str):
        ds_path = DATASETS_PATH+ds_name+"/"
        #merges all csv files from cicids into one big csv
        all_files = glob.glob(os.path.join(ds_path , "*.csv"))
        li = []
        header=True
        for filename in all_files:
            print(filename)
            df = pd.read_csv(filename, index_col=None, header=0)
            df.to_csv(ds_path + f"full-{ds_name.lower()}.csv",
                      mode = "a", index= False, header=header
                    )
            header = False
  

        
#     def ds_open(self):
#         self.df = pd.read_csv(self.path+f"full-{self.path.split()[2]}.csv", index_col=None)

# #df = pd.concat(li, axis=0, ignore_index=True)

# #df.to_csv(path+"full-cicids.csv")
