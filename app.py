# Base Package
import streamlit as st
import pandas as pd
import numpy as np
# Modeling Packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
# Viz Packages
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import warnings
warnings.filterwarnings("ignore")

st.title("Semi-Auto Machine Learning App")


@st.cache(allow_output_mutation=True)
def upload(uploadfile_):
    data = pd.read_csv(uploadfile_)
    return data


# EDA
def eda(df):
    st.header('Exploratory Data Analysis')
    if st.checkbox(label='Dimensions of Data Frame'):
        st.write(f'Number of Rows in this data frame are {df.shape[0]}')
        st.write(f'Number of Columns in this data frame are {df.shape[1]}')

    if st.checkbox(label='Summary of Data Frame'):
        st.dataframe(df.describe())

    if st.checkbox(label='Data Types of Each Columns'):
        st.dataframe(uploaded_file.dtypes.rename('Data Types'))

    if st.checkbox(label='Missing Value Count in Each Column'):
        st.dataframe(df.isnull().sum().rename('Missing Value Counts'))

    if st.checkbox(label='Check Category Count'):
        col = st.selectbox(label='Select a Column', options=col_name)
        st.dataframe(df[col].value_counts())

    if st.checkbox(label='View Correlation'):
        st.dataframe(df.corr())
        sns.heatmap(pd.DataFrame(df.corr()), annot=True, cmap="viridis")
        st.pyplot()


# EVALUATION method for Regression on Metrics Like MSE, MAE, R2
def regression_eval(y_test_set, y_prediction):
    st.success(f"Mean Absolute Error is : {metrics.mean_absolute_error(y_test_set, y_prediction)}")
    st.success(f"Mean Squared Error is : {metrics.mean_squared_error(y_test_set, y_prediction)}")
    st.success(f"Root Mean Squared Error is : {np.sqrt(metrics.mean_squared_error(y_test_set, y_prediction))}")
    st.success(f"R-squared value is : {metrics.r2_score(y_test_set, y_prediction)}")


# EVALUATION method for Classification on Metrics Like Accuracy, Confusion Matrix
# noinspection PyPep8Naming
def classification_eval(X_test, clf, y_train, y_test, y_pred):
    y = clf.predict_proba(X_test)
    result_df = pd.DataFrame(data=y, columns=list(np.unique(y_train)))
    st.header('Probability Matrix For Test Set')
    st.dataframe(result_df)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred, labels=list(np.unique(y_train)))
    fig, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="viridis", xticklabels=list(np.unique(y_train)),
                yticklabels=list(np.unique(y_train)))
    plt.title('Confusion Matrix')
    ax.xaxis.set_label_position("top")
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    st.pyplot()
    st.write("Classification Report")
    st.table(metrics.classification_report(y_test, y_pred, output_dict=True, target_names=list(np.unique(y_train))))
    st.success(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")


# noinspection PyPep8Naming
# Splitting
def df_split(df):
    independent_Col = st.multiselect(label='Select Independent Columns', options=col_name)
    dependent_Col = st.multiselect(label='Select Dependent Columns', options=col_name)
    x = df[independent_Col]
    y = df[dependent_Col]
    testsize = st.slider('Select Test Size in Percentage', 10, 100)
    testsize = testsize / 100
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=testsize, random_state=42)
    return X_train, X_test, y_train, y_test


# Regression
# noinspection PyPep8Naming
def linear_reg(df):
    X_train, X_test, y_train, y_test = df_split(df)
    if st.checkbox(label='See The Model Result'):
        lrr = LinearRegression()
        clf = lrr.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        regression_eval(y_test_set=y_test, y_prediction=y_pred)


# noinspection PyPep8Naming
def knn_reg(df):
    X_train, X_test, y_train, y_test = df_split(df)
    n_neighbors = st.slider("Select Number of K", 1, 10)
    weights = st.selectbox(label='Select Weight for ', options=['uniform', 'distance'], index=0)
    algorithm = st.selectbox(label='Select Algorithm from list', options=['auto', 'kd_tree', 'ball_tree',
                                                                          'brute_force'], index=0)
    leaf_size = st.slider('Select Leaf Size for BallTree or KDTree', 1, 100)
    metric = st.selectbox(label='Select Distance Metric', options=['euclidean', 'manhattan', 'chebyshev',
                                                                   'minkowski', 'wminkowski', 'seuclidean',
                                                                   'mahalanobis'], index=3)
    if st.checkbox(label='See The Model Result'):
        knnr = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size,
                                   metric=metric, )
        clf = knnr.fit(X_train, np.ravel(y_train))
        y_pred = clf.predict(X_test)
        regression_eval(y_test_set=y_test, y_prediction=y_pred)
        

# noinspection PyPep8Naming
def lasso_reg(df):
    X_train, X_test, y_train, y_test = df_split(df)
    alpha = st.slider("Select The Value of Penalty Value or Alpha default is 1", 1, 10)
    if st.checkbox(label='See The Model Result'):
        lasso = Lasso(alpha=alpha, random_state=44)
        clf = lasso.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        regression_eval(y_test_set=y_test, y_prediction=y_pred)
        

# noinspection PyPep8Naming
def ridge_reg(df):
    X_train, X_test, y_train, y_test = df_split(df)
    alpha = st.slider("Select The Value of Penalty Value or Alpha default is 1", 1, 10)
    if st.checkbox(label='See The Model Result'):
        ridreg = Ridge(alpha=alpha, random_state=44)
        clf = ridreg.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        regression_eval(y_test_set=y_test, y_prediction=y_pred)
        

# noinspection PyPep8Naming
def random_forest_reg(df):
    X_train, X_test, y_train, y_test = df_split(df)
    n_estimators = st.number_input(label='Enter Number of Estimator (Integer)', min_value=10, max_value=None)
    criterion = st.selectbox(label='Select the Split Criteria', options=['mse', 'mae'], index=0)
    max_depth = st.number_input(label='Enter Depth of Tree (Integer)', min_value=1, max_value=None)
    max_features = st.selectbox(label='Number of Features to consider at split', options=['auto', 'sqrt', 'log2'],
                                index=0)
    max_leaf_nodes = st.number_input(label='Enter Number of leaf Nodes', min_value=3, max_value=None)
    if st.checkbox(label='See The Model Result'):
        randforreg = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                           max_features=max_features, max_leaf_nodes=max_leaf_nodes, random_state=44,
                                           n_jobs=-1)
        clf = randforreg.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        regression_eval(y_test_set=y_test, y_prediction=y_pred)


# noinspection PyPep8Naming
def support_vector_reg(df):
    X_train, X_test, y_train, y_test = df_split(df)
    C = st.slider("Select Value of C", 1, 10)
    gamma = st.selectbox(label=' Select Kernel coefficient for Poly, rbf and sigmoid', options=['scale', 'auto'],
                         index=0)
    kernel = st.selectbox(label='Select Kernel', options=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], index=2)
    degree_check_box = st.checkbox(label='If Kernel is Poly Select Degree of Polynomial')

    degree = 3
    if degree_check_box:
        degree = st.number_input(label='Enter the Degree of Polynomial', min_value=2, max_value=None)
        degree = int(degree)
        if st.checkbox(label='See The Model Result'):
            svr = SVR(C=C, kernel=kernel, degree=degree, gamma=gamma)
            clf = svr.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            regression_eval(y_test_set=y_test, y_prediction=y_pred)

    if st.checkbox(label='See The Model Result'):
        svr = SVR(C=C, kernel=kernel, degree=degree, gamma=gamma)
        clf = svr.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        regression_eval(y_test_set=y_test, y_prediction=y_pred)
        

# noinspection PyPep8Naming
def decision_tree_reg(df):
    X_train, X_test, y_train, y_test = df_split(df)
    criterion = st.selectbox(label='Select Split Criteria', options=['mse', 'friedman_mse', 'mae'], index=0)
    max_depth = st.number_input(label='Enter Depth of Tree', min_value=10, max_value=None)
    max_features = st.selectbox(label='Number of Features to consider at split', options=['auto', 'sqrt', 'log2'],
                                index=0)
    max_leaf_nodes = st.number_input(label='Enter Number of Leaf Nodes', min_value=3, max_value=None)
    if st.checkbox(label='See The Model Result'):
        dectreereg = DecisionTreeRegressor(criterion=criterion, random_state=44, max_depth=max_depth,
                                           max_features=max_features, max_leaf_nodes=max_leaf_nodes)
        clf = dectreereg.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        regression_eval(y_test_set=y_test, y_prediction=y_pred)
        

# Classification
# noinspection PyPep8Naming
def log_reg(df):
    st.write('''

    For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
    For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; 
    ‘liblinear’ is limited to one-versus-rest schemes.
    ‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty
    ‘liblinear’ and ‘saga’ also handle L1 penalty
    ‘saga’ also supports ‘elasticnet’ penalty
    ‘liblinear’ does not support setting penalty='none'
''')
    penalty = st.selectbox(label='Select Penalty Norm', options=['l1', 'l2', 'elasticnet', 'none'], index=1)
    solver = st.selectbox(label='Select Solver Method', options=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                          index=1)
    multi_class = st.selectbox(label='Select Class Type', options=['auto', 'ovr', 'multinomial'], index=0)
    X_train, X_test, y_train, y_test = df_split(df)
    if st.checkbox(label='See The Model Result'):
        logr = LogisticRegression(penalty=penalty, solver=solver, multi_class=multi_class, random_state=0)
        clf = logr.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        classification_eval(clf=clf, X_test=X_test, y_train=y_train, y_test=y_test, y_pred=y_pred)
        

# noinspection PyPep8Naming
def knn_classification(df):
    X_train, X_test, y_train, y_test = df_split(df)
    n_neighbors = st.slider("Select Number of K", 1, 10)
    weights = st.selectbox(label='Select Weight for ', options=['uniform', 'distance'], index=0)
    algorithm = st.selectbox(label='Select Algorithm from list', options=['auto', 'kd_tree', 'ball_tree',
                                                                          'brute_force'], index=0)
    leaf_size = st.slider('Select Leaf Size for BallTree or KDTree', 1, 100)
    metric = st.selectbox(label='Select Distance Metric', options=['euclidean', 'manhattan', 'chebyshev',
                                                                   'minkowski', 'wminkowski', 'seuclidean',
                                                                   'mahalanobis'], index=3)
    st.dataframe(y_test)
    if st.checkbox(label='See The Model Result'):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size,
                                   metric=metric)
        clf = knn.fit(X_train, np.ravel(y_train))
        y_pred = clf.predict(X_test)
        classification_eval(clf=clf, X_test=X_test, y_train=y_train, y_test=y_test, y_pred=y_pred)
        

# noinspection PyPep8Naming
def decision_tree_classification(df):
    X_train, X_test, y_train, y_test = df_split(df)
    criterion = st.selectbox(label='Select the split criteria', options=['gini', 'entropy'], index=0)
    max_depth = st.number_input(label='Enter Depth of Tree', min_value=10, max_value=None)
    max_features = st.selectbox(label='Number of Features to consider at split', options=['auto', 'sqrt', 'log2'],
                                index=0)
    max_leaf_nodes = st.number_input(label='Enter Number of Leaf Nodes', min_value=3, max_value=None)
    if st.checkbox(label='See The Model Result'):
        dec_tree_clf = DecisionTreeClassifier(criterion=criterion, random_state=44, max_depth=max_depth,
                                              max_features=max_features, max_leaf_nodes=max_leaf_nodes)
        clf = dec_tree_clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        classification_eval(clf=clf, X_test=X_test, y_train=y_train, y_test=y_test, y_pred=y_pred)
        

# noinspection PyPep8Naming,PyTypeChecker
def random_forest_classification(df):
    X_train, X_test, y_train, y_test = df_split(df)
    n_estimators = st.number_input(label='Enter Number of Estimator (Integer)', min_value=10, max_value=None)
    criterion = st.selectbox(label='Select the split criteria', options=['gini', 'entropy'], index=0)
    max_depth = st.number_input(label='Enter Depth of Tree (Integer)', min_value=1, max_value=None)
    max_features = st.selectbox(label='Number of Features to consider at split', options=['auto', 'sqrt', 'log2'],
                                index=0)
    max_leaf_nodes = st.number_input(label='Enter Number of leaf Nodes', min_value=3, max_value=None)
    if st.checkbox(label='See The Model Result'):
        randforclf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                            max_features=max_features, max_leaf_nodes=max_leaf_nodes, random_state=44)
        clf = randforclf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        classification_eval(clf=clf, X_test=X_test, y_train=y_train, y_test=y_test, y_pred=y_pred)
        

# noinspection PyPep8Naming,PyTypeChecker
def support_vector_classification(df):
    X_train, X_test, y_train, y_test = df_split(df)
    C = st.slider("Select the Value of C", 1, 10)
    kernel = st.selectbox(label='Select Kernel', options=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], index=2)
    gamma = st.selectbox(label=' Select Kernel coefficient for Poly, rbf and sigmoid', options=['scale', 'auto'],
                         index=0)
    degree_check_box = st.checkbox(label='If Kernel is Poly Select Degree of Polynomial')
    degree = 3
    if degree_check_box:
        degree = st.number_input(label='Enter the Degree of Polynomial', min_value=1, max_value=None)
        degree = int(degree)
        if st.checkbox(label='See The Model Result'):
            svc = SVC(C=C, degree=degree, kernel=kernel, gamma=gamma, probability=True)
            clf = svc.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            classification_eval(clf=clf, X_test=X_test, y_train=y_train, y_test=y_test, y_pred=y_pred)

    if st.checkbox(label='See The Model Result'):
        svc = SVC(C=C, degree=degree, kernel=kernel, gamma=gamma, probability=True)
        clf = svc.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        classification_eval(clf=clf, X_test=X_test, y_train=y_train, y_test=y_test, y_pred=y_pred)
        

def model(df):
    st.header('Choose Model Specifications')
    problemtype = st.sidebar.selectbox(label='Select Problem Type', options=['Regression', 'Classification'])
    if problemtype == 'Regression':
        methodlist = st.sidebar.selectbox(label='Select Algorithm', options=['Linear Regression',
                                                                             'KNN Regression',
                                                                             'lasso Regression',
                                                                             'Ridge Regression',
                                                                             'Random Forest Regression',
                                                                             'Support Vector Regression',
                                                                             'Decision Tree Regression'],
                                          key='regressmethod')
        if methodlist == 'Linear Regression':
            linear_reg(df)
        elif methodlist == 'KNN Regression':
            knn_reg(df)
        elif methodlist == 'lasso Regression':
            lasso_reg(df)
        elif methodlist == 'Ridge Regression':
            ridge_reg(df)
        elif methodlist == 'Random Forest Regression':
            random_forest_reg(df)
        elif methodlist == 'Support Vector Regression':
            support_vector_reg(df)
        else:
            decision_tree_reg(df)
    else:
        methodlist = st.sidebar.selectbox(label='Select Algorithm', options=['KNN Classification',
                                                                             'Logistic Regression',
                                                                             'Decision Tree Classification',
                                                                             'Random Forest Classification',
                                                                             'Support Vector Classification'],
                                          key='classificationmethod')
        if methodlist == 'Logistic Regression':
            log_reg(df)
        elif methodlist == 'KNN Classification':
            knn_classification(df)
        elif methodlist == 'Decision Tree Classification':
            decision_tree_classification(df)
        elif methodlist == 'Random Forest Classification':
            random_forest_classification(df)
        else:
            support_vector_classification(df)


def visualisation(df):
    x_axis = st.selectbox(label='Select Column X-Axis', options=col_name, key='x_axis')
    y_axis = st.selectbox(label='Select Column Y-Axis', options=col_name, key='y_axis')
    color = st.selectbox(label='Select Categorical Column', options=col_name, key='color')
    chart_type = st.sidebar.selectbox(label='Select Chart Type', options=['Pair Plot',
                                                                          'Scatter Plot',
                                                                          'Line Chart',
                                                                          'Pie Chart',
                                                                          'Strip Plot',
                                                                          'Violin Plot',
                                                                          'Histogram',
                                                                          'Distribution Plot',
                                                                          'Sunburst Chart',
                                                                          'Box Plot'], index=1)
    if chart_type == 'Pair Plot':
        fig1 = ff.create_scatterplotmatrix(df, diag='histogram', height=800, width=800)
        st.plotly_chart(fig1)
        fig2 = ff.create_scatterplotmatrix(df, index=color, diag='box', height=800, width=800, colormap_type='cat')
        st.plotly_chart(fig2)

    elif chart_type == 'Scatter Plot':
        st.header(f'{x_axis} vs {y_axis}')
        fig = px.scatter(df, x=x_axis, y=y_axis)
        st.plotly_chart(fig)
        st.header(f'{x_axis} vs {y_axis} and legends are {color} columns value')
        fig1 = px.scatter(df, x=x_axis, y=y_axis, color=color, hover_data=df)
        st.plotly_chart(fig1)

    elif chart_type == 'Line Chart':
        st.header(f'{x_axis} vs {y_axis}')
        fig = px.line(df, x=x_axis, y=y_axis, hover_data=df)
        st.plotly_chart(fig)
        st.header(f'{x_axis} vs {y_axis} and legends are {color} columns value')
        fig1 = go.Figure()
        for color, groupdf in df.groupby(color):
            fig1.add_trace(go.Scatter(x=groupdf[x_axis], y=groupdf[y_axis], name=color, mode='markers'))
            st.plotly_chart(fig1)

    elif chart_type == 'Pie Chart':
        st.header(f'Pie Chart of {color} column')
        fig = px.pie(df, names=df[color])
        st.plotly_chart(fig)
        st.header(f'Donut Chart of {color} column')
        fig1 = px.pie(df, names=df[color], hole=0.3)
        st.plotly_chart(fig1)

    elif chart_type == 'Strip Plot':
        st.header(f'{x_axis} vs {y_axis}')
        fig = px.strip(df, x=df[x_axis], y=df[y_axis])
        st.plotly_chart(fig)
        st.header(f'{x_axis} vs {y_axis} and legends are {color} columns value')
        fig1 = px.strip(df, x=df[x_axis], y=df[y_axis], color=df[color])
        st.plotly_chart(fig1)

    elif chart_type == 'Violin Plot':
        st.header(f'{x_axis} vs {y_axis}')
        fig = px.violin(df, x=df[x_axis], y=df[y_axis], points='all')
        st.plotly_chart(fig)
        st.header(f'{x_axis} vs {y_axis} and legends are {color} columns value')
        fig1 = px.violin(df, x=df[x_axis], y=df[y_axis], color=df[color], points='all')
        st.plotly_chart(fig1)

    elif chart_type == 'Histogram':
        fig = px.histogram(df, x=df[x_axis])
        st.plotly_chart(fig)
        if st.checkbox(label='Y-Axis is Sum'):
            fig1 = px.histogram(df, x=df[x_axis], y=df[y_axis], histfunc='sum')
            st.plotly_chart(fig1)
            fig2 = px.histogram(df, x=df[x_axis], y=df[x_axis], color=df[color], histfunc='sum')
            st.plotly_chart(fig2)
        if st.checkbox(label='Y-Axis is Count'):
            fig3 = px.histogram(df, x=df[x_axis], y=df[y_axis], histfunc='count')
            st.plotly_chart(fig3)
            fig4 = px.histogram(df, x=df[x_axis], y=df[x_axis], color=df[color], histfunc='count')
            st.plotly_chart(fig4)
        if st.checkbox(label='Y-Axis is Average'):
            fig5 = px.histogram(df, x=df[x_axis], y=df[y_axis], histfunc='avg')
            st.plotly_chart(fig5)
            fig6 = px.histogram(df, x=df[x_axis], y=df[x_axis], color=df[color], histfunc='avg')
            st.plotly_chart(fig6)

    elif chart_type == 'Sunburst Chart':
        path = st.multiselect(label='Select The Path', options=df.columns)
        fig = px.sunburst(df, path=path, color=color, values=x_axis)
        st.plotly_chart(fig)

    elif chart_type == 'Box Plot':
        fig = px.box(df, x=df[x_axis], y=df[y_axis], points='all`')
        st.plotly_chart(fig)
        fig1 = px.box(df, x=df[x_axis], y=df[y_axis], points='all', color=df[color])
        st.plotly_chart(fig1)

    else:
        fig1 = px.histogram(df, x=df[x_axis], marginal="box")
        st.plotly_chart(fig1)
        fig2 = px.histogram(df, x=df[x_axis], y=df[y_axis], marginal="box")
        st.plotly_chart(fig2)
        fig3 = px.histogram(df, x=df[x_axis], y=df[y_axis], color=df[color], marginal="box")
        st.plotly_chart(fig3)


def main():
    if side_bar == 'EDA':
        eda(uploaded_file)
    elif side_bar == 'Modeling':
        model(uploaded_file)
    elif side_bar == 'Visualisation':
        visualisation(uploaded_file)


if __name__ == '__main__':
    file = st.file_uploader("Upload a CSV file", type="csv")
    side_bar = st.sidebar.selectbox(label='What do you want to do?', options=['EDA', 'Visualisation', 'Modeling'])

    if file is not None:
        uploaded_file = upload(file)
        col_name = uploaded_file.columns
        if st.checkbox(label='View Dataset'):
            DataFrame = st.dataframe(uploaded_file)
        main()

