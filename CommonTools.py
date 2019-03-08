import IPython
import plotly
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#colorscale used for scatterplot
pl_colorscale = [[0.0, '#08085e'],
                [0.333, '#08085e'],
                [0.333, '#4ef037'],
                [0.666, '#4ef037'],
                [0.666, '#24bddf'],
                [1, '#24bddf']]

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
