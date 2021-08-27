
"""
    This gets the training data from the data folder and saves it to a large
    csv file for ease of use
"""
import pandas as pd
import os

# locations of data

training_loc_b = "/../medical data/training_setB/"
training_loc_a = "/../medical data/training/"

def copyData(data_folder_location, csv_file_location):
    DATA = pd.DataFrame()
    old_percent = 0
    df = pd.DataFrame()

    for index, filename in enumerate(os.listdir(data_folder_location)):
        temp_df = pd.read_csv(data_folder_location + filename, sep='|')
        df = pd.concat([df, temp_df]).reset_index(drop=True)

        percent = int((index/20336)*100)
        if percent > old_percent:
            old_percent = percent
            print(str(percent) + "%")


    df.to_csv(data_folder_location + csv_file_location,
        encoding='utf-8', index=False)

    return DATA

data_folder_location = os.getcwd() + training_loc_a
csv_file_location = '/../concated_training_data_A.csv'

if not os.path.isfile(data_folder_location + csv_file_location):
    TRAIN_DATA_A = copyData(data_folder_location, csv_file_location)

data_folder_location = os.getcwd() + training_loc_b
csv_file_location = '/../concated_training_data_B.csv'

if not os.path.isfile(data_folder_location + csv_file_location):
    TRAIN_DATA_B = copyData(data_folder_location, csv_file_location)


else:
    TRAIN_DATA_A = pd.read_csv(os.getcwd() + '/../medical data/concated_training_data_A.csv')
    TRAIN_DATA_B = pd.read_csv(os.getcwd() + '/../medical data/concated_training_data_B.csv')











