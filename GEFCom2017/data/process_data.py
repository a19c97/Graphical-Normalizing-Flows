import pickle
from time import strptime

import numpy as np
import pandas as pd


def init_load_df(raw_load):
    dates = raw_load.date.unique()

    row_inds = []
    for d in dates:
        d = d.replace("/", "-")
        for i in range(1, 25):
            row_inds.append("{}-{}".format(d, i))

    col_inds = raw_load.meter_id.unique()
    col_inds = ["meter_{}".format(i) for i in col_inds]
    
    load_df = pd.DataFrame(index=row_inds, columns=col_inds)
    return load_df


def to_list(r):
    return list(r[2:])


def flatten(r):
    return [i for l in r for i in l]


def convert_time(row):
    date = row.date
    d = date[:2]
    m = str(strptime(date[2:5],'%b').tm_mon)
    if len(m) < 2:
        m = "0" + m
    y = date[5:]
    h = row.hr
    
    ind = "{}-{}-{}-{}".format(m, d, y, h)
    return ind


def append_hierarchical_sums(df, hierarchy):
    df = df.copy()
    for mid_level, meters in hierarchy.groupby("mid_level")["meter_id"]:
        cols = ["meter_{}".format(m) for m in meters]
        cols = list(set(cols).intersection(set(df.columns)))
        if len(cols) == 0:
            continue
        df[mid_level] = df[cols].sum(axis=1)
    
    for aggregate, mid_levels in hierarchy.groupby("aggregate")["mid_level"]:
        mid_levels = mid_levels.unique()        
        mid_levels = list(set(mid_levels).intersection(set(df.columns)))
        df[aggregate] = df[mid_levels].sum(axis=1)
    return df


def set_adj_hierarchy(adj_mat, df, hierarchy):
    for mid_level, data in hierarchy.groupby("mid_level"):
        aggregate = data["aggregate"].unique()
        assert(len(aggregate) == 1)
        aggregate = aggregate[0]
        meters = data.meter_id

        cols = ["meter_{}".format(m) for m in meters]
        cols = list(set(cols).intersection(set(df.columns)))
        if len(cols) == 0:
            continue
        
        inds = [df.columns.get_loc(c) for c in cols]
        agg_ind = df.columns.get_loc(aggregate)
        mid_ind = df.columns.get_loc(mid_level)

        for i in inds:
            # Meters depend on other meters within clusters
            adj_mat[i][inds] = 1
            # Meters also depend on aggregate stats
            adj_mat[i][agg_ind] = 1
            adj_mat[i][mid_ind] = 1
            adj_mat[mid_ind][inds] = 1
            adj_mat[agg_ind][inds] = 1

        # Aggregate stats depend on each other
        adj_mat[mid_ind][agg_ind] = 1
        adj_mat[agg_ind][mid_ind] = 1

        
def get_adj_mat(data_df_wm):
    n_var = data_df_wm.columns.shape[0]
    adj_mat = np.zeros((n_var, n_var))

    # Meters depend on weather stations
    # Note that we don't set the other direction
    adj_mat[56:, :56] = 1

    # Weather is probably correlated
    adj_mat[:56, :56] = 1

    set_adj_hierarchy(adj_mat, data_df_wm, hierarchy)
    return adj_mat


def minmax_norm(df):
    return (df-df.min())/(df.max()-df.min())


if __name__ == "__main__":
    hierarchy_f = open("./hierarchy.csv", "r")
    load_f = open("./load.csv", "r")
    temp_f = open("./temperature.csv", "r")
    relative_humidity_f = open("./relative_humidity.csv", "r")

    hierarchy = pd.read_csv(hierarchy_f)
    raw_load = pd.read_csv(load_f)
    raw_temp = pd.read_csv(temp_f)
    raw_humidity = pd.read_csv(relative_humidity_f)

    load_df = init_load_df(raw_load)

    load_l = raw_load.iloc[:, :2]
    load_l["hourly"] = raw_load.apply(to_list, axis=1)
    hourly_load = load_l.groupby("meter_id")["hourly"].apply(list)
    hourly_load = hourly_load.apply(flatten)

    # Drops incomplete meters
    hourly_load = hourly_load[hourly_load.apply(len) == 61344]

    # Fills into dataframe
    for ind, meter in hourly_load.items():
        load_df.loc[:, "meter_" + str(ind)] = meter

    # Drops empty meters
    load_df = load_df.dropna(axis=1, how="all")

    temp = raw_temp.set_index(raw_temp.apply(convert_time, axis=1))
    temp = temp.drop(columns=["date", "hr"])

    humidity = raw_humidity.set_index(raw_humidity.apply(convert_time, axis=1))
    humidity = humidity.drop(columns=["date", "hr"])

    data_df = load_df.join(humidity).join(temp)

    meter_cols = [c for c in data_df.columns if c[:5] == "meter"]
    non_meter_cols = [c for c in data_df.columns if c[:5] != "meter"]
    data_df = data_df[non_meter_cols + meter_cols]
    data_df.dropna(inplace=True)

    data_df_wm = append_hierarchical_sums(data_df, hierarchy)
    data_df_norm = minmax_norm(data_df_wm)
    adj_mat = get_adj_mat(data_df_wm)


    dataset = {
        "df": data_df_norm,
        "adj_mat": adj_mat
    }

    with open("gefcom2017_processed.pkl", "wb") as f:
        pickle.dump(dataset, f)
