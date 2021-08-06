import pandas as pd
from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(df2.drop('classification_yes', 1), df2['classification_yes'], test_size = .2, random_state=10)

import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt




class liver:
    def __init__(self):
        pass
    def selectkbest(self,indep_X,dep_Y):
        test = SelectKBest(score_func=chi2, k=5)
        fit1= test.fit(indep_X,dep_Y)
        # summarize scores
        features = indep_X.columns.values.tolist()
        np.set_printoptions(precision=2)
        print(features)
        print(fit1.scores_)
        #plt.figure(figsize=(12,3))
        #plt.bar(fit1.scores_,height=0.6)
        feature_series = pd.Series(data=fit1.scores_,index=features)
        feature_series.plot.bar()

        selectk_features = fit1.transform(indep_X)
        return selectk_features
    def pca(self,features,dep_Y):
        pca = PCA(n_components=5)
        fit2 = pca.fit(features)
        pca_feature=fit2.transform(features)
        return pca_feature
    def svm(self,features,indep_X,dep_Y):
        X_train, X_test, y_train, y_test = train_test_split(features, dep_Y, test_size = 0.25, random_state = 0)
        #X_train, X_test, y_train, y_test = train_test_split(rfe_feature, dep_Y, test_size = 0.25, random_state = 0)
        #X_train, X_test, y_train, y_test = train_test_split(pca_feature, dep_Y, test_size = 0.25, random_state = 0)
        #X_train, X_test, y_train, y_test = train_test_split(feature_import, dep_Y, test_size = 0.25, random_state = 0)
        #X_train, X_test, y_train, y_test = train_test_split(indep_X,dep_Y, test_size = 0.25, random_state = 0)

        #Feature Scaling
        #from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Fitting K-NN to the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'rbf', random_state = 0,probability=True)
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)

        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)

        from sklearn.metrics import accuracy_score
        from sklearn.metrics import classification_report
        #from sklearn.metrics import confusion_matrix
        #cm = confusion_matrix(y_test, y_pred)

        Accuracy=accuracy_score(y_test, y_pred )

        report=classification_report(y_test, y_pred)
        return  classifier,Accuracy,report,X_test,y_test,cm
