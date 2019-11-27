import numpy as np
import pandas as pd
import json

def predict(tree,data):
    prediction=[]
    for i in range(len(data)):
        node=tree
        new_data=data.iloc[i,:]
        while type(node) is list:
            v=new_data[node[0]]
            node=node[1][str(v)[:-2]]
        prediction.append(node)

    return prediction


def accuracy(prediction,target):
    prediction=np.array(prediction,dtype=float)
    return (len(target[target==prediction])/len(target))*100

def main(fname):
    data=np.loadtxt("../data/test.txt")
    data=data.T
    columnsData=["RISK","AGE", "CRED_HIS","INCOME","RACE","HEALTH"]
    data=pd.DataFrame(data,columns=columnsData)
    with open(r'../data/'+fname,'r') as f:
        tree=json.load(f)
    prediction=predict(tree,data)
    accuracy_ptg=accuracy(prediction,data["RISK"])
    return accuracy_ptg
