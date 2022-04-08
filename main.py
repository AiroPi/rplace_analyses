import time
import datetime
import math
from os import path

import pandas as pd
from PIL import Image
import numpy as np

colors = [(255, 255, 255), (255, 248, 184), (255, 214, 53), (255, 180, 112), (255, 168, 0), (255, 153, 170), (255, 69, 0), (255, 56, 129), (228, 171, 255), (222, 16, 127), (212, 215, 217), (190, 0, 57), (180, 74, 192), (156, 105, 38), (148, 179, 255), (137, 141, 144), (129, 30, 159), (126, 237, 86), (109, 72, 47), (109, 0, 26,), (106, 92, 255), (81, 233, 244), (81, 82, 82), (73, 58, 193), (54, 144, 234), (36, 80, 164), (0, 204, 192), (0, 204, 120), (0, 163, 104), (0, 158, 170), (0, 117, 111), (0, 0, 0,)]

moment = datetime.datetime.fromtimestamp(1648822500000/1000)
moment += datetime.timedelta(days=1)

step = 0
start = time.time()
def show_perf(): 
    global step, start
    print(f'Step', (step:=step+1), ':', -start+(start:=time.time()))


print("Let's start !")

if not path.exists('data.hdf'):
    df = pd.read_csv('combined_sort_timestamp.csv', nrows=60000000, dtype={'x_coordinate': np.uint16, 'y_coordinate': np.uint16, 'color': np.uint8})
    print(f'Step', (step:=step+1), ':', -start+(start:=time.time()))
    df.to_hdf('data.hdf', 'data')
    print(f'SubStep', step, ':', -start+(start:=time.time()))
else:
    df = pd.read_hdf('data.hdf', 'data')
show_perf()
# df = df.loc[(df['ts'] < int(moment.timestamp()) * 1000)]

array = df['ts'].to_numpy()
value = moment.timestamp() * 1000

idx = np.searchsorted(array, value, side="left")
if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
    idx-=1

df = df[:idx]

show_perf()

# df.drop_duplicates(subset=['x_coordinate', 'y_coordinate'], keep='last', inplace=True)
# show_perf()

new = df.pivot_table(index='y_coordinate', columns='x_coordinate', values='color', aggfunc='first').to_numpy()
# new = df.set_index(['y_coordinate', 'x_coordinate']).unstack().values
show_perf()
rgb_dtype = np.dtype([('r', np.int8), ('g', np.int8), ('b', np.int8)])
data = np.full(new.shape, 255, dtype=rgb_dtype)
show_perf()

for i, color in enumerate(colors):
    data[new == i] = color
show_perf()
img = Image.fromarray(data, 'RGB')       # Create a PIL image
show_perf()
img = img.quantize(colors=256)
show_perf()
img.save(f'{datetime.datetime.now().strftime("%d-%m-%y_%H-%M-%S")}.png', optimize=True)
show_perf()
img.show()
