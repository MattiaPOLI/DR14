import IPython
import numpy as np
import pandas as pd
import plotly
import io
import requests
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.plotly import plot, iplot
from imblearn.over_sampling import SVMSMOTE
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, learning_curve

url = "https://raw.githubusercontent.com/MattiaPOLI/DR14/master/Sky.csv"
path = requests.get(url).content
df = pd.read_csv(io.StringIO(path.decode('utf-8')))
df["run"] = df["run"].astype(np.float64)
df["rerun"] = df["rerun"].astype(np.float64)
df["camcol"] = df["camcol"].astype(np.float64)
df["field"] = df["field"].astype(np.float64)
df["plate"] = df["plate"].astype(np.float64)
df["mjd"] = df["mjd"].astype(np.float64)
df["fiberid"] = df["fiberid"].astype(np.float64)

# colorscale used for scatterplot
pl_colorscale = [[0.0, '#08085e'],
                 [0.333, '#08085e'],
                 [0.333, '#4ef037'],
                 [0.666, '#4ef037'],
                 [0.666, '#24bddf'],
                 [1, '#24bddf']]
# colors for learning curve
colors = plotly.colors.DEFAULT_PLOTLY_COLORS

# transparent colors
def get_color_with_opacity(color, opacity):
    return "rgba(" + color[4:-1] + ", %.2f)" % opacity


# set of functions to return a specific filtered datateset
def get_raw_dataset():
	y = df["class"]
	X = df.drop(columns=["class"])
	X = StandardScaler().fit_transform(X)
	return X, y


def get_most_relevant_dataset():
	X = df.drop(columns=["objid", "ra", "dec", "run", "rerun", "camcol", "field", "class", "fiberid"])
	y = df["class"]
	X = StandardScaler().fit_transform(X)
	return X, y


def get_meaningful_dataset():
	X = df.drop(
		columns=["objid", "ra", "dec", "run", "rerun", "camcol", "field", "class", "fiberid", "specobjid", "plate",
		         "mjd"])
	y = df["class"]
	X = StandardScaler().fit_transform(X)
	return X, y


def get_pca_dataset():
	y = df["class"]
	X = df.drop(columns=["class"])
	X = variance_pca(X, False)
	X = X[:, 0:5]
	return X, y


# function used to view the plot in google colab too; must be used before iplot
def enable_plotly_in_cell():
	display(IPython.core.display.HTML('''<script src="/static/components/requirejs/require.js"></script>'''))
	plotly.offline.init_notebook_mode(connected=True)


# function that takes as input data required to perform PCA. If graph is True then it show also the bar-graph explaining how much variance
# each component is carrying. It returns the dataset projected in the new space.
def variance_pca(df, graph):
	dfStandard = StandardScaler().fit_transform(df)
	pca = PCA()
	data_PCA = pca.fit_transform(dfStandard)
	if graph == True:
		cum_sum = pca.explained_variance_ratio_.cumsum()
		cum_sum = cum_sum * 100
		bars = [go.Bar(y=cum_sum)]
		enable_plotly_in_cell()
		plotly.offline.iplot(bars, filename="cumVariance")
	return data_PCA


# function to plot the confusion matrix
def confusion_matrix_(true, predicted):
	confuMatrix = confusion_matrix(true, predicted)
	for j in range(3):
		confuMatrix[0][j], confuMatrix[2][j] = confuMatrix[2][j], confuMatrix[0][j]
	confuMatrix = np.ndarray.round(confuMatrix.astype(float) / confuMatrix.sum(axis=1)[:, np.newaxis], 3)
	cmColorScale = [[0.0, "rgb(255, 255, 255)"], [1.0, "rgb(0, 0, 255)"]]
	z = confuMatrix
	x = ["Galaxy", "QSO", "Star"]
	y = ["Star", "QSO", "Galaxy"]
	figure = ff.create_annotated_heatmap(z, x=x, y=y, colorscale=cmColorScale, showscale=True)
	figure['layout']['xaxis']['title']['text'] = "Predicted Label"
	figure['layout']['xaxis']['side'] = "bottom"
	figure['layout']['yaxis']['title']['text'] = "True Label"
	figure['layout']['title'] = "Normalized Confusion Matrix"
	enable_plotly_in_cell()
	plotly.offline.iplot(figure, filename="ConfusionMatrix")


# function to balance the dataset, standard strategy is all. In this case a SVM SMOTE approach is used because the SVM classifier is
# returning the best result.
def balance_dataset(df, dfLabel, strategy="all"):
	sm = SVMSMOTE(random_state=42, sampling_strategy=strategy)
	trainOver, labelOver = sm.fit_sample(df, dfLabel)
	return trainOver, labelOver


# function to plot the learning curve of a classifier,with respect to the training set size
def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
	"""
	Generate a simple plot of the test and training learning curve.

	Parameters
	----------
	estimator : object type that implements the "fit" and "predict" methods
		An object of that type which is cloned for each validation.

	title : string
		Title for the chart.

	X : array-like, shape (n_samples, n_features)
		Training vector, where n_samples is the number of samples and
		n_features is the number of features.

	y : array-like, shape (n_samples) or (n_samples, n_features), optional
		Target relative to X for classification or regression;
		None for unsupervised learning.

	cv : int, cross-validation generator or an iterable, optional
		Determines the cross-validation splitting strategy.
		Possible inputs for cv are:
		  - None, to use the default 3-fold cross-validation,
		  - integer, to specify the number of folds.
		  - :term:`CV splitter`,
		  - An iterable yielding (train, test) splits as arrays of indices.

		For integer/None inputs, if ``y`` is binary or multiclass,
		:class:`StratifiedKFold` used. If the estimator is not a classifier
		or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

		Refer :ref:`User Guide <cross_validation>` for the various
		cross-validators that can be used here.

	n_jobs : int or None, optional (default=None)
		Number of jobs to run in parallel.
		``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
		``-1`` means using all processors. See :term:`Glossary <n_jobs>`
		for more details.

	train_sizes : array-like, shape (n_ticks,), dtype float or int
		Relative or absolute numbers of training examples that will be used to
		generate the learning curve. If the dtype is float, it is regarded as a
		fraction of the maximum size of the training set (that is determined
		by the selected validation method), i.e. it has to be within (0, 1].
		Otherwise it is interpreted as absolute sizes of the training sets.
		Note that for classification the number of samples usually have to
		be big enough to contain at least one sample from each class.
		(default: np.linspace(0.1, 1.0, 5))
	"""
	
	train_sizes, train_scores, test_scores = learning_curve(
		estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="accuracy", random_state=42)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	
	trace1 = go.Scatter(
		x=train_sizes,
		y=train_scores_mean - train_scores_std,
		showlegend=False,
		mode="lines",
		name="",
		hoverlabel=dict(
			namelength=20
		),
		line=dict(
			width=0.1,
			color=get_color_with_opacity(colors[0], 0.4),
		),
	)
	trace2 = go.Scatter(
		x=train_sizes,
		y=train_scores_mean + train_scores_std,
		showlegend=False,
		fill="tonexty",
		mode="lines",
		name="",
		hoverlabel=dict(
			namelength=20
		),
		line=dict(
			width=0.1,
			color=get_color_with_opacity(colors[0], 0.4),
		),
	)
	trace3 = go.Scatter(
		x=train_sizes,
		y=train_scores_mean,
		showlegend=True,
		name="Train score",
		line=dict(
			color=colors[0],
		),
	)
	
	trace4 = go.Scatter(
		x=train_sizes,
		y=test_scores_mean - test_scores_std,
		showlegend=False,
		mode="lines",
		name="",
		hoverlabel=dict(
			namelength=20
		),
		line=dict(
			width=0.1,
			color=get_color_with_opacity(colors[1], 0.4),
		),
	)
	trace5 = go.Scatter(
		x=train_sizes,
		y=test_scores_mean + test_scores_std,
		showlegend=False,
		fill="tonexty",
		mode="lines",
		name="",
		hoverlabel=dict(
			namelength=20
		),
		line=dict(
			width=0.1,
			color=get_color_with_opacity(colors[1], 0.4),
		),
	)
	trace6 = go.Scatter(
		x=train_sizes,
		y=test_scores_mean,
		showlegend=True,
		name="Test score",
		line=dict(
			color=colors[1],
		),
	)
	
	data = [trace1, trace2, trace3, trace4, trace5, trace6]
	layout = go.Layout(
		title=title,
		autosize=True,
		yaxis=dict(
			title='Accuracy',
		),
		xaxis=dict(
			title="#Training samples",
		),
		legend=dict(
			x=0.8,
			y=0,
		),
	)
	enable_plotly_in_cell()
	fig = go.Figure(data=data, layout=layout)
	return plotly.offline.iplot(fig, filename=title)


# function to perform grid search for parameter tuning and return
def grid_search(estimator, parameters, X_train, y_train, cv=5):
	# create empty object
	gridsearch_obj = GridSearchCV(estimator=estimator, param_grid=parameters, cv=cv, return_train_score=False,
	                              refit=True, error_score=0, n_jobs=-1)
	# actually perform grid search on given data
	gridsearch_obj.fit(X_train, y_train)
	
	# return the best alternative
	return gridsearch_obj.best_estimator_

# function to return the 'accuracy' model evaluation index
def accuracy(estimator, X_test, y_test):
	y_pred = estimator.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	return acc