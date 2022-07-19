# generate stats.csv:

import numpy as np
import pandas as pd
import scipy.stats as stats

control = np.random.normal(loc=0, scale=0.5, size=500)
t1 = np.random.normal(loc=-2, scale=1, size=300)
t2 = np.random.normal(loc=2, scale=1.1, size=300)
treatment = np.concatenate([t1, t2])
np.random.shuffle(treatment) # mix 'em up

f, ax = plt.subplots()
edges = np.arange(-5, 5, 0.25)
ax.hist(control, bins=edges)
ax.hist(treatment, bins=edges)

df = pd.concat([pd.DataFrame({'control':control}),
                pd.DataFrame({'treatment':treatment})], axis=1)
df.to_csv('stats.csv', index=False) # has NaNs


# getting the data back is realistically annoying:

df = pd.read_csv('stats.csv')
control = df['control'].dropna().values # filter out any NaNs, convert to numpy array
treatment = df['treatment'].dropna().values
f, ax = plt.subplots()
edges = np.arange(-5, 5, 0.25)
ax.hist(control, bins=edges, label='control') # looks normal
ax.hist(treatment, bins=edges, label='treatment') # doesn't look normal
ax.legend()
f.savefig('stats.png')

stats.ks_2samp(control, treatment) # appropriate
stats.ttest_ind(control, treatment, equal_var=False) # Welch's, inappropriate

stats.kstest(control, 'norm', args=(control.mean(), control.std())) # can't reject, normal
stats.kstest(treatment, 'norm', args=(treatment.mean(), treatment.std())) # reject, not normal

# could also keep the NaNs in while specifying kwarg nan_policy='omit'
