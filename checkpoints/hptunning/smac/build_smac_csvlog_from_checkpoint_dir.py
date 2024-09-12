import json
import pandas as pd

def json_to_csv(json_filepath, csv_filepath):
    # Load the JSON data from the file
    with open(json_filepath, 'r') as file:
        json_data = json.load(file)
    
    # Extract relevant data and create a DataFrame
    data = []
    for entry in json_data['data']:
        id = entry[0]
        budget = entry[3]
        cost = entry[4]
        starttime = entry[7]
        endtime = entry[8]
        data.append([id, budget, cost, starttime, endtime])

    # Create a DataFrame with the required columns
    df = pd.DataFrame(data, columns=['id', 'budget', 'cost', 'start_time', 'end_time'])

    # Write the DataFrame to a CSV file
    df.to_csv(csv_filepath)

json_to_csv("checkpoints/hptunning/smac/ETTh1_96_96/123/runhistory.json", "checkpoints/hptunning/smac/ETTh1_96_96/results.csv")
json_to_csv("checkpoints/hptunning/smac/ETTh1_96_192/123/runhistory.json", "checkpoints/hptunning/smac/ETTh1_96_192/results.csv")
json_to_csv("checkpoints/hptunning/smac/ETTh1_96_336/123/runhistory.json", "checkpoints/hptunning/smac/ETTh1_96_336/results.csv")
json_to_csv("checkpoints/hptunning/smac/ETTh1_96_720/123/runhistory.json", "checkpoints/hptunning/smac/ETTh1_96_720/results.csv")