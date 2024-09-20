import pandas as pd
import numpy as np

#Usually save_preds function in main.py already outputs the CSV file of (predictions,ground_truth)
#calculate the MAE metric
def calculate_mae(actual, predicted):

    actual_cleaned = clean_data(actual)
    predicted_cleaned = clean_data(predicted)
    

    mae = ((actual_cleaned - predicted_cleaned).abs()).mean()
    return mae

def clean_data(data):

    if isinstance(data, (pd.DataFrame, pd.Series)):
        cleaned_data = data.apply(pd.to_numeric, errors='coerce')
    else:
        try:
            cleaned_data = pd.to_numeric(data, errors='coerce')
        except TypeError:
            raise TypeError("Data should be a pandas DataFrame or Series.")
    return cleaned_data


def read_csv_file(file_path):
    return pd.read_csv(file_path)


def write_to_txt_file(txt_file_path, file_name, mae_value):
    with open(txt_file_path, 'a') as file:
        file.write(f"{file_name},{mae_value}\n")

if __name__ == "__main__":
    csv_file_path = "/your_csv_result_path/dist_multi_comma_multitask_none_True.csv" 
    txt_file_path = '/your_MAE_record/MAE.txt'  

    df = read_csv_file(csv_file_path)

    if df.shape[1] < 2:
        raise ValueError("CSV must contains 2 columns")

    actual_values = df.iloc[:, 1]
    predicted_values = df.iloc[:, 0]

    mae = calculate_mae(actual_values, predicted_values)
    file_name = csv_file_path.split('/')[-1]
    print(f"MAE (Mean Absolute Error) for {file_name}: {mae}")

    write_to_txt_file(txt_file_path, file_name, mae)