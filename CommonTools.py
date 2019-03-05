import IPython
import Plotly
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#function used to view the plot in google colab too; must be used before iplot
def enable_plotly_in_cell():
  display(IPython.core.display.HTML('''<script src="/static/components/requirejs/require.js"></script>'''))
  plotly.offline.init_notebook_mode(connected=True)

#show a bar graph explaining the variance associated to each component
def variance_pca():
  dfStandard = StandardScaler().fit_transform(df)
  PCA = PCA()
  data_PCA = PCA.fit_transform(dfStandard)
  cum_sum = PCA.explained_variance_ratio_.cumsum()
  cum_sum = cum_sum * 100
  bars = [go.Bar(y = cum_sum)]
  enable_plotly_in_cell()
  plotly.offline.iplot(bars, filename="cumVariance")
