import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis()


class_1 = np.ones(4)
class_0 = np.zeros(3)
classes = np.concatenate((class_0, class_1))

classes=np.array([1,1,1,1,2,2,2])

#importing the data
#df = pd.DataFrame.from_csv("SCLC_study_output_filtered_2.csv")
df = pd.DataFrame.from_csv("inputdata1.csv")
data_matrix = df.as_matrix()

print(data_matrix)
clf.fit(data_matrix,classes)

print("Prediction:")
print(clf.predict([[2.81, 5.46]]))

#print(clf.predict_proba([[2.81, 5.46]]))

