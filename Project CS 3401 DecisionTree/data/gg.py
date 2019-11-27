import numpy as np
import pandas as pd
data=np.loadtxt("train.txt")
data=data.T
data=pd.DataFrame(data,columns=["RISK","AGE", "CRED_HIS","INCOME","RACE","HEALTH"])
y=data["RISK"].astype(str)
x=data[data.columns[data.columns!="RISK"]]
for i in x.columns:
    x[i]=x[i].astype(str)
x.dtypes
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x,y)
print(dt.get_depth())
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png("iris.png")
