import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation: #To encapsulate the process of creating a data transformation pipeline that includes handling of both numerical and categorical data.
    def __init__(self): #to initialize the class with an instance of DataTransformationConfig
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self): #This method constructs and returns a preprocessing pipeline for data transformation using sklearn's pipeline functionality.
        '''
        This code defines a  class named DataTransformation that is used to create a preprocessing pipeline for data transformation in machine learning projects.
        The DataTransformation class provides a structured way to assemble a data preprocessing pipeline that handles both numerical and categorical data using scikit-learn's Pipeline and ColumnTransformer classes. This setup ensures that all necessary data transformations (imputation, encoding, scaling) are performed in a way that is suitable for feeding into a machine learning model. The use of custom logging and exception handling helps in maintaining robustness and traceability of the process.
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),#filling missing values with the median
                ("scaler",StandardScaler()) #standardizing features by removing the mean and scaling to unit variance

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),#replacing missing values with the most frequent value
                ("one_hot_encoder",OneHotEncoder()),#transforming categorical variables into a form that could be provided to ML algorithms
                ("scaler",StandardScaler(with_mean=False))#especially important to apply scaling after one-hot encoding to prevent features with larger categories dominating those with fewer categories
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(#Combines both the numerical and categorical pipelines into a single transformer object that applies appropriate transformations to each column type
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)#If an exception is caught, it raises a CustomException, which is a custom-defined exception class for the project.
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(#saves pickel file

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
