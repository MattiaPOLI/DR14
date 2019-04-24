import IPython
import numpy as np
import pandas as pd
import plotly
import io
import requests
import plotly.graph_objs as go
import plotly.figure_factory as ff
from imblearn.over_sampling import SVMSMOTE
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url = "https://raw.githubusercontent.com/MattiaPOLI/DR14/master/Sky.csv"
path = requests.get(url).content
df = pd.read_csv(io.StringIO(path.decode('utf-8')))
df["run"].astype(np.float64)
df["rerun"].astype(np.float64)
df["camcol"].astype(np.float64)
df["field"].astype(np.float64)
df["plate"].astype(np.float64)
df["mjd"].astype(np.float64)
df["fiberid"].astype(np.float64)

#colorscale used for scatterplot
pl_colorscale = [[0.0, '#08085e'],
                [0.333, '#08085e'],
                [0.333, '#4ef037'],
                [0.666, '#4ef037'],
                [0.666, '#24bddf'],
                [1, '#24bddf']]

#set of functions to return a specific filtered datateset
def get_raw_dataset():
  y = df["class"]
  X = df.drop(columns = ["class"])
  X = StandardScaler().fit_transform(X)
  return X, y

def get_most_relevant_dataset():
  X = df.drop(columns = ["objid", "ra", "dec", "run", "rerun", "camcol", "field", "class", "fiberid"])
  y = df["class"]
  X = StandardScaler().fit_transform(X)
  return X, y
              
def get_meaningful_dataset():
  X = df.drop(columns = ["objid", "ra", "dec", "run", "rerun", "camcol", "field", "class", "fiberid", "specobjid", "plate", "mjd"])
  y = df["class"]
  X = StandardScaler().fit_transform(X)
  return X, y
  
def get_pca_dataset():
  y = df["class"]
  X = df.drop(columns = ["class"])
  X = variance_pca(X, False)
  X = X[:, 0:5]
  return X, y
  
#function used to view the plot in google colab too; must be used before iplot
def enable_plotly_in_cell():
  display(IPython.core.display.HTML('''<script src="/static/components/requirejs/require.js"></script>'''))
  plotly.offline.init_notebook_mode(connected=True)

#function that takes as input data required to perform PCA. If graph is True then it show also the bar-graph explaining how much variance
#each component is carrying. It returns the dataset projected in the new space.
def variance_pca(df, graph):
  dfStandard = StandardScaler().fit_transform(df)
  pca = PCA()
  data_PCA = pca.fit_transform(dfStandard)
  if graph == True: 
    cum_sum = pca.explained_variance_ratio_.cumsum()
    cum_sum = cum_sum * 100
    bars = [go.Bar(y = cum_sum)]
    enable_plotly_in_cell()
    plotly.offline.iplot(bars, filename="cumVariance")
  return data_PCA

#function to plot the confusion matrix
def confusion_matrix_(true, predicted):
  confuMatrix = confusion_matrix(true, predicted)
  for j in range(3):
    confuMatrix[0][j], confuMatrix[2][j] = confuMatrix[2][j], confuMatrix[0][j]
  confuMatrix = np.ndarray.round(confuMatrix.astype(float) / confuMatrix.sum(axis = 1)[:, np.newaxis], 3)
  cmColorScale = [[0.0, "rgb(255, 255, 255)"], [1.0, "rgb(0, 0, 255)"]]
  z = confuMatrix
  x = ["Galaxy", "QSO", "Star"]
  y = ["Star", "QSO", "Galaxy"]
  figure = ff.create_annotated_heatmap(z, x = x, y = y, colorscale= cmColorScale, showscale = True)
  figure['layout']['xaxis']['title']['text'] = "Predicted Label"
  figure['layout']['xaxis']['side'] = "bottom"
  figure['layout']['yaxis']['title']['text'] = "True Label"
  figure['layout']['title'] = "Normalized Confusion Matrix"
  enable_plotly_in_cell()
  plotly.offline.iplot(figure, filename="ConfusionMatrix")
  
  
#function to balance the dataset, standard strategy is all. In this case a SVM SMOTE approach is used because the SVM classifier is
#returning the best result.
def balance_dataset(df, dfLabel, strategy = "all"):
  sm = SVMSMOTE(random_state = 42, sampling_strategy = strategy)
  trainOver, labelOver = sm.fit_sample(df, dfLabel)
  return trainOver, labelOver
