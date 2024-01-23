import numpy as np
import pandas as pd
import pickle

df_all_stations = pd.read_csv("../Data/Save/Pressure_graph_stations_2366738.csv")
df_stations = pd.read_csv("../Data/stations.csv")
with open("../Data/euclidean_adj_matrix.pkl", "rb") as f:
    adj_matrix = pickle.load(f)

# get_station_index function
def get_station_index(station_name):
    """
    Returns the index of the station in the DataFrame.
    """
    return df_stations[df_stations["Name"] == station_name].index[0]

def adj_matrix_of_nearby_stations(station_name, radius=3):
    """
    Return: distance values of nearby stations from given station within a given radius
    """
    station_index = get_station_index(station_name)
    return adj_matrix[station_index].sort_values()[1:radius + 1]

def calculate_station_weights(station_distances, p=2):
    """
    Calculate Theta, i.e weights of nearby stations and return a list of weights
    """
    theta = []
    for index, value in station_distances.items():
        theta.append((1 / value ** p) / sum(1 / station_distances.values ** p))
    return theta

# columns = ["Temperature", "Rain", "Humidity", "WindDirection", "WindSpeed", "Pressure"]
columns = ["Temperature"]
radius = 3  # Radius of 3 stations

for column in columns:
    print(f"Processing column: {column}")
    count = 0
    for index, row in df_all_stations.iterrows():
        if pd.isnull(row[column]):
            count += 1
            # print(f"Index: {index}, Count: {count}")

            if count == 1000:
                print(f"Index: {index}, Count: {count}, Saving to CSV file...")
                df_all_stations.to_csv("../Data/Save/" + str(column) + "_graph_stations_" + str(index) + ".csv", index=False)
                count = 0
            
            # Get adjacency matrix values of nearby stations
            near_by_stations_adj = adj_matrix_of_nearby_stations(row['StasName'], radius)
            # Calculate weights of nearby stations
            near_by_stations_weights = calculate_station_weights(near_by_stations_adj)
            # print(f"Nearby Stations Adj: {near_by_stations_adj}")
            # print(f"Nearby Stations Weights: {near_by_stations_weights}")

            # Fo every station adjacency index value, get its name and use it to find column value at that StasName and DateT
            Col_value = [] # E.g Temperature, Humidit, etc
            # print(f"Nearby Stations: {near_by_stations_adj}")
            for indx in near_by_stations_adj.index:
                # print(f"Col_value: {Col_value}")
                # print(f"indx: {indx}")
                # Assign variable date_time depending on If df_all_stations['DateT'] == row['DateT'], else add 1 hour to row['DateT']
                new_date_time = row['DateT'] if df_all_stations[(df_all_stations['StasName'] == df_stations.loc[indx, "Name"]) & (df_all_stations['DateT'] == row['DateT'])].shape[0] > 0 else pd.to_datetime(row['DateT']) + pd.Timedelta(hours=1)
                # print(f"new_date_time: {new_date_time}")
                # print(f"ROW: {df_all_stations.loc[index]}")
                # print(f"STATION: {df_all_stations[(df_all_stations['StasName'] == df_stations.loc[indx, 'Name']) & (df_all_stations['DateT'] == new_date_time)]}")
                station_df = df_all_stations[(df_all_stations['StasName'] == df_stations.loc[indx, 'Name']) & (df_all_stations['DateT'] == new_date_time)]
                if station_df.empty:
                    value = 0
                    # print("Station empty")
                else:
                    value = station_df[column].values[0]
                # if value is NAN append 0, else value
                Col_value.append(0 if pd.isnull(value) else value)

            # Calculate Value_hat, e.g new value for Temperature, Humidity, etc
            Value_hat = 0
            for i in range(len(Col_value)):
                Value_hat += near_by_stations_weights[i] * Col_value[i]

            # Round accordingly
            if not isinstance(Value_hat, int):
                Value_hat = np.round(Value_hat, decimals=1)

            # Write the new value to the original DataFrame to replace the NaN value    
            df_all_stations.at[index, column] = Value_hat
            # break

df_all_stations.to_csv("../Data/Save/" + str(column) + "_graph_stations_" + str(index) + ".csv", index=False)