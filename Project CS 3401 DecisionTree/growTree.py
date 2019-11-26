import numpy as np
import pandas as pd
import json
def id3(data, target, columns, tree,dataDomain,parent):
    unique_targets = np.unique(data[target])

    if len(data)==0:
        tree[0]=str(parent[target].value_counts().argmax())[:-2]
        return 
    elif len(unique_targets) == 1:
        # Insert code here that assigns the "label" field to the node dictionary
        if 1 in unique_targets:
            tree[0]=str(1)
        else:
            tree[0]=str(2)
        return
    elif len(columns)==0:
        tree[0]=str(data[target].value_counts().argmax())[:-2]
        return
    else:
        parent=data
        best_column = find_best_column(data, target, columns)
        # Insert code here that assigns the "column" and "median" fields to the node dictionary
        split_dict = []
        for i in dataDomain[best_column].keys():
            split_dict.append([i,data[data[best_column] == float(i)]])
        tree[0]=best_column

        for name, split in split_dict:
            tree[1][name] = [None,{}]
            id3(split, target, columns[columns!=best_column], tree[1][name],dataDomain,parent)

def find_best_column(data,target,columns):
    best_columns=np.array([])
    for i in columns:
        gain = entropy(data[target])-mutualInfo(data[target],data[i])
        best_columns=np.append(best_columns, gain)
    maxIndex=np.argmax(best_columns)
    bestColumn=columns[maxIndex]
    return bestColumn



def mutualInfo(r1, r2):
    mutualDep=0
    S=len(r1)
    entropyR1=entropy(r1)
    for i in np.unique(r1):
        for j in np.unique(r2):
            Sab=len(r1[r2==j][r1[r2==j]==i])
            Sb=len(r2[r2==j])
            mutualDep+=(-(Sab)/S)*np.log2((Sab)/(Sb))
    return mutualDep


#Calculate the entropy for a, which is a set of values for a single feature.
def entropy(a):
    lenOfUnqElem=[]
    unique_elem=np.unique(a)
    for i in unique_elem:
        lenUniq=len(a[a==i])
        lenOfUnqElem.append(lenUniq)
    entropyS=0
    len_a=len(a)
    for j in lenOfUnqElem:
        entropyS+=((-j/len_a)*np.log2(j/len_a))
    return entropyS

    #Fill in the missing part here
# Create a dictionary to hold the tree  
# It has to be outside of the function so we can access it later
tree = [None,{}]
data=np.loadtxt("train.txt")
data=data.T
data=pd.DataFrame(data,columns=["RISK","AGE", "CRED_HIS","INCOME","RACE","HEALTH"])
with open("deDomain.txt","r") as g:
    dataDomain=json.load(g)
id3(data, "RISK", np.array(["AGE", "CRED_HIS","INCOME","RACE","HEALTH"]),tree,dataDomain,parent=None)
print(tree)
