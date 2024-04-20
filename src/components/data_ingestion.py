import os
import sys #to use CustomException
from src.exception import CustomException #a customised class in exception.py
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
"""
The @dataclass decorator is used here to automatically generate special methods like __init__(), __repr__(), __eq__(), and others for the class, based on the class attributes defined.
The dataclass is particularly useful in cases like this because it automatically generates many utility methods and makes the class easy to use, especially for storing 'plain' data structures. 
It reduces the amount of boilerplate code you have to write to create classes that are mainly containers for data fields.
"""
@dataclass
class DataIngestionConfig:#This class is designed to hold configuration paths for different datasets
    train_data_path: str=os.path.join('artifacts',"train.csv") #the training data file (train.csv) is expected to be in an artifacts directory relative to the location where this script is run.
    test_data_path: str=os.path.join('artifacts',"test.csv") #The use of os.path.join is crucial as it constructs a file path that is operating system agnostic, ensuring that the file paths are 
    raw_data_path: str=os.path.join('artifacts',"data.csv") #correctly specified on platforms like Windows, Linux, or MacOS 

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))



