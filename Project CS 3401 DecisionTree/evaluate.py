import numpy as np
import pandas as pd
import json

def predict(tree,data,columnsData):
    node=tree
    prediction=[]
    for i in range(len(data)):
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
    data=np.loadtxt("test.txt")
    data=data.T
    columnsData=["RISK","AGE", "CRED_HIS","INCOME","RACE","HEALTH"]
    data=pd.DataFrame(data,columns=columnsData)
    with open(fname,"r") as f:
        tree=json.load(f)
    prediction=predict(tree,data,columnsData)
    accuracy_ptg=accuracy(prediction,data["RISK"])
    print(accuracy_ptg)
    return accuracy_ptg
main("treeFileFull.txt")
