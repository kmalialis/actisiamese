import pandas as pd

df = pd.read_csv('circle_init50.csv', header=None)
classes = sorted(list(set(df[2].values)))
df_init = pd.DataFrame()
init = 10
for c in classes:
    df_temp = df[df[2] == c].reset_index(drop=True)
    df_init = pd.concat([df_init, df_temp[:init]]).reset_index(drop=True)
df_init = df_init.sample(frac=1).reset_index(drop=True)
df_init.to_csv('circle_init' + str(init) + '.csv', header=None,index=False)