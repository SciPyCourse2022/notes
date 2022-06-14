## Problem 1:

'''
In [1]: len(t), len(x1), len(x2)
Out[1]: (20, 20, 20) # yes, they match in length
'''

## Problem 2:

y = []
for x1val, x2val in zip(x1, x2):
    y.append(x1val * x2val)
y = np.array(y)
'''
In [2]: y
Out[2]:
array([ 0.00000000e+00,  5.87785252e+00,  1.42658477e+01, -4.75528258e+00,
        1.76335576e+01,  4.28626380e-15, -2.35114101e+01, -4.27975432e+01,
       -4.75528258e+01, -2.93892626e+01, -1.22464680e-14,  2.93892626e+01,
                   nan,  4.27975432e+01,  2.35114101e+01,  1.28587914e-14,
       -1.76335576e+01, -2.37764129e+01, -1.42658477e+01,             nan])

In [3]: y.dtype
Out[3]: dtype('float64')
'''

## Problem 3:

# convert all 3 to arrays for later convenience:
x1, x2, t = np.array(x1), np.array(x2), np.array(t)
y = x1 * x2
'''
In [4]: y
Out[4]:
array([ 0.00000000e+00,  5.87785252e+00,  1.42658477e+01, -4.75528258e+00,
        1.76335576e+01,  4.28626380e-15, -2.35114101e+01, -4.27975432e+01,
       -4.75528258e+01, -2.93892626e+01, -1.22464680e-14,  2.93892626e+01,
                   nan,  4.27975432e+01,  2.35114101e+01,  1.28587914e-14,
       -1.76335576e+01, -2.37764129e+01, -1.42658477e+01,             nan])
'''

## Problem 4:

# x1 is float64, x2 is int64, expect float64 from their product, float is more flexible than int:
'''
In [19]: y.dtype
Out[19]: dtype('float64')
'''
# each is of length 20, and each array is of dtype float64 or int64. Both dtypes are 64 bits
# long = 8 bytes long, so all 4 arrays should be 20 * 8 =160 bytes
'''
In [21]: t.dtype
Out[21]: dtype('float64')

In [22]: x1.nbytes
Out[22]: 160

In [23]: x2.nbytes
Out[23]: 160

In [24]: y.nbytes
Out[24]: 160

In [25]: t.nbytes
Out[25]: 160
'''

## Problem 5:

# get bool index array of only the valid data points that we want to keep:
keepis = ~np.isnan(x1) & (x2 != -1) # vectorized NOT and AND operations
x1, x2, t = x1[keepis], x2[keepis], t[keepis]
'''
In [5]: x1
Out[5]:
array([ 0.00000000e+00,  2.93892626e+00,  4.75528258e+00,  2.93892626e+00,
        6.12323400e-16, -2.93892626e+00, -4.75528258e+00, -4.75528258e+00,
       -2.93892626e+00, -1.22464680e-15,  2.93892626e+00,  4.75528258e+00,
        2.93892626e+00,  1.83697020e-15, -2.93892626e+00, -4.75528258e+00,
       -4.75528258e+00])
In [6]: x2
Out[6]: array([ 0,  2,  3,  6,  7,  8,  9, 10, 10, 10, 10,  9,  8,  7,  6,  5,  3])
In [7]: t
Out[7]:
array([0. , 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.3, 1.4,
       1.5, 1.6, 1.7, 1.8])
In [8]: len(t), len(x1), len(x2)
Out[8]: (17, 17, 17) # 17 timepoints left
'''

## Problem 6:

y = x1 * x2
'''
In [6]: y
Out[6]:
array([ 0.00000000e+00,  5.87785252e+00,  1.42658477e+01,  1.76335576e+01,
        4.28626380e-15, -2.35114101e+01, -4.27975432e+01, -4.75528258e+01,
       -2.93892626e+01, -1.22464680e-14,  2.93892626e+01,  4.27975432e+01,
        2.35114101e+01,  1.28587914e-14, -1.76335576e+01, -2.37764129e+01,
       -1.42658477e+01])
In [7]: len(y)
Out[7]: 17 # yes, aligned with t
'''

## Problem 7:

i = x2.argmax() # 7
'''
In [8]: x1[i], x2[i], t[i]
Out[8]: (-4.75528258, 10, 0.8)
'''

## Problem 8:

'''
In [17]: np.where(x2 == 10)[0]
Out[17]: array([ 7,  8,  9, 10]) # not unique, argmax() only returns index of first max
'''
