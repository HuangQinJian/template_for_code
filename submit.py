#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: HuangQinJian
@LastEditors: HuangQinJian
@Date: 2019-05-04 13:07:38
@LastEditTime: 2019-05-24 09:00:35
'''
import pandas as pd
import math
import numpy as np

mapF = {0: 0, 1: 6, 2: 15, 3: 2, 4: 5, 5: 19, 6: 9, 7: 13, 8: 10, 9: 8, 10: 20,
        11: 3, 12: 14, 13: 11, 14: 7, 15: 18, 16: 16, 17: 1, 18: 17, 19: 12, 20: 4}

submit = pd.read_csv('submit.csv')

submit['type'] = submit['type'].map(mapF)

submit['X1'] = submit['X1'].apply(lambda x:math.ceil(x))
submit['Y1'] = submit['Y1'].apply(lambda x:math.ceil(x))

submit['X2'] = submit['X2'].apply(lambda x:math.ceil(x))
submit['Y2'] = submit['Y2'].apply(lambda x:math.ceil(x))


submit['X3'] = submit['X3'].apply(lambda x:math.ceil(x))
submit['Y3'] = submit['Y3'].apply(lambda x:math.ceil(x))

submit['X4'] = submit['X4'].apply(lambda x:math.ceil(x))
submit['Y4'] = submit['Y4'].apply(lambda x:math.ceil(x))


# submit['X1'] = submit['X1'].apply(lambda x:np.round(x))
# submit['Y1'] = submit['Y1'].apply(lambda x:np.round(x))

# submit['X2'] = submit['X2'].apply(lambda x:np.round(x))
# submit['Y2'] = submit['Y2'].apply(lambda x:np.round(x))


# submit['X3'] = submit['X3'].apply(lambda x:np.round(x))
# submit['Y3'] = submit['Y3'].apply(lambda x:np.round(x))

# submit['X4'] = submit['X4'].apply(lambda x:np.round(x))
# submit['Y4'] = submit['Y4'].apply(lambda x:np.round(x))

submit.to_csv('sub.csv', index=None)
