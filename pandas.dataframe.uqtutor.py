import random
import numpy as np
import pandas as pd

# generate multidim array
data_np = np.random.randint(low=0, high=100, size=(3, 4))

def rand_array(row, col):
    return [[random.randrange(start=0, stop=100) for _ in range(col)] for _ in range(row)]

data  = np.array(rand_array(3, 4))
columns = ['Eleanor', 'Chidi', 'Tahani', 'Jason']
dataframe = pd.DataFrame(data=data, columns=columns)
# test 
print(dataframe)
print("\nSecond row of the Eleanor column: %d\n" % dataframe['Eleanor'][1]) # use column - then row

dataframe['Janet'] = dataframe['Tahani'] + dataframe['Jason']
#test
print(dataframe)


# modify cell:
dataframe.at[1, 'Jason'] = dataframe['Jason'][1] + 2
print(dataframe)