import numpy as np
import os as os
import glob as glob # glob.glob I like how this sounds
import pandas as pd
import requests

files = glob.glob("log*")
uuid = ["306726","306727","306728","3067270","3067280","306733","306735","306736","306760","306756","306757","306758","306738","306740","306741","306730","306731","306732","306743","306744","306745","306749","306751","306752"]
i = 0

for uu in uuid:
    ff = "log_"+uu
    f = glob.glob(ff)
    f = f[0]
    df = pd.read_csv(f, sep='\t', lineterminator="\n", dtype={'iter': np.int, 'bmin': np.float64, 'bbest':np.float64, 'bmax': np.float64, 'chi2': np.float64, 'aic': np.float64,'bic': np.float64 ,'acceptance-fraction': np.float64, 'ESSmin': np.float64})
    ii = np.argmin(df["aic"])
    #ii = 2
    if i == 0:
        print("uuid\titer\tBmin\tBBest\t<BBest>\tBBest-<BBest>\tBmax\tchi²\tAIC\tBIC\tacceptance fraction\tESSmin")
    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t".format(uu,df.loc[ii,"iter"],df.loc[ii,"bmin"],df.loc[ii,"bbest"]," "," ",df.loc[ii,"bmax"],df.loc[ii,"chi2"],df.loc[ii,"aic"],df.loc[ii,"bic"],df.loc[ii,"acceptance-fraction"],df.loc[ii,"ESSmin"]))
    #print(df.loc[ii])
    i+=1
