import pandas as pd
import numpy as np

def shift(arr, num, fill_value=0):
    #print(arr)
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result = arr
    return result

def mvts_to_xy(*tslist, n_x=1, n_y=1, x_idx=None, y_idx=None):
    n_ts = len(tslist)
    if n_ts == 0:
        raise ValueError('At least one timeseries required as input')

    #TODO: Validation of other options

    result = []

    for ts in tslist:
        ts_cols = 1 if ts.ndim==1 else ts.shape[1]
        if x_idx is None:
            x_idx = range(0,ts_cols)
        if y_idx is None:
            y_idx = range(0,ts_cols)

        n_x_vars = len(x_idx)
        n_y_vars = len(y_idx)

        ts_rows = ts.shape[0]
        n_rows = ts_rows - n_x - n_y + 1

        dataX=np.empty(shape=(n_rows, n_x_vars * n_x),dtype=np.float32)
        dataY=np.empty(shape=(n_rows, n_y_vars * n_y),dtype=np.float32)
        x_cols, y_cols, names = list(), list(), list()

        # input sequence x (t-n, ... t-1)
        from_col = 0
        for i in range(n_x, 0, -1):
            dataX[:,from_col:from_col+n_x_vars]=shift(ts[:,x_idx],i)[n_x:ts_rows-n_y+1]
            from_col = from_col+n_x_vars

        # forecast sequence (t, t+1, ... t+n)
        from_col = 0
        for i in range(0, n_y):
            #y_cols.append(shift(ts,-i))
            dataY[:,from_col:from_col+n_y_vars]=shift(ts[:,y_idx],-i)[n_x:ts_rows-n_y+1]
            from_col = from_col + n_y_vars

        # put it all together
        #x_agg = concat(x_cols, axis=1).dropna(inplace=True)
        #y_agg = concat(y_cols, axis=1).dropna(inplace=True)

        #dataX = np.array(x_cols,dtype=np.float32)
        #dataY = np.array(y_cols,dtype=np.float32)

        result.append(dataX)
        result.append(dataY)
    return result

def getDataset():
    filepath = "dataset/datasets_1392_2506_international-airline-passengers.csv"

    dataframe = pd.read_csv(filepath)
    dataset_ = dataframe.values

    num_train = 97
    num_test = 48
    train = np.zeros((97, 1))
    test = np.zeros((48, 1))
    dataset = np.zeros((145, 1))
    for i, value in enumerate(dataset_):
        if i < 97:
            train[i] = value[1]
        elif i < 145:
            test[i-97] = value[1]
        dataset[i] = value[1]

    # reshape into X=t-1,t and Y=t+1
    n_x=2
    n_y=1

    X_train, Y_train, X_test, Y_test = mvts_to_xy(train,test,n_x=n_x,n_y=n_y)
    return dataset, X_train, Y_train, X_test, Y_test


