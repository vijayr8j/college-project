import pandas as pd

dataset=pd.read_csv("Prepocessed_HCC.csv",index_col=None)



import pandas as pd
#dataset1=pd.read_csv("prep.csv",index_col=None)

df2=dataset

from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import train_test_split 
#X_train, X_test, y_train, y_test = train_test_split(df2.drop('classification_yes', 1), df2['classification_yes'], test_size = .2, random_state=10)
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, classification_report, r2_score, make_scorer, roc_curve, auc
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score, StratifiedKFold, KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression


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
from sklearn.svm import SVC
#df2 = pd.get_dummies(df2, drop_first=True)

def selectkbest(indep_X,dep_Y):
        test = SelectKBest(score_func=chi2, k=10)
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
    
def rfeFeature(indep_X,dep_Y):
        #model=SVR(kernel="linear")
        model = LogisticRegression(solver='lbfgs')
        rfe = RFE(model,10)
        fit3 = rfe.fit(indep_X, dep_Y)
        rfe_feature=fit3.transform(indep_X)
        features = indep_X.columns.values.tolist()
        #feature_series = pd.Series(data=rfe_feature,index=features)
        #feature_series.plot.bar()
        return rfe_feature
def pca(features,dep_Y):
        pca = PCA(n_components=3)
        fit2 = pca.fit(features)
        pca_feature=fit2.transform(features)
        return pca_feature
        
def svm(features,indep_X,dep_Y):
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
        classifier = SVC(kernel = 'rbf', random_state = 0)
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
   

    
def naives(features,indep_X,dep_Y):
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
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
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
def Decision(features,indep_X,dep_Y):
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
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
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
def knn(features,indep_X,dep_Y):
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
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
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

def random(features,indep_X,dep_Y):
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
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
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
    

def logistics(features,indep_X,dep_Y):
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
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(solver='lbfgs',
                                        random_state=42,
                                        C=0.1,
                                        multi_class='ovr',
                                        penalty='l2')
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
def kfoldd(classifier,indep_X,n_split):
        #sc = StandardScaler()
        #feature = sc.fit_transform(feature)
        #X_test = sc.transform(X_test)
        kfold = StratifiedKFold(
            n_splits=n_split, 
            shuffle=True, 
            random_state=42
        )
        
        predicted = cross_val_predict(
            classifier,
            indep_X,
            #feature,
           # rfe_feature, 
            dep_Y, 
            cv=kfold
        )
        
        scores = cross_val_score(
            classifier, 
            indep_X,
            #feature, 
            #rfe_feature,
            dep_Y,      
            cv=kfold,
            scoring='f1'
        )
        return scores
  
import warnings
#import warnings
warnings.filterwarnings("ignore")
#warnings.simplefilter(action='ignore', category=FutureWarning, UndefinedMetricWarning)
#warnings.simplefilter(action='ignore', category=Conve)
    
indep_X=df2.drop('class_attribute', 1)

#std_scaler = StandardScaler() #StandardScaler() # RobustScaler
#indep_X = std_scaler.fit_transform()

dep_Y=df2['class_attribute']

selectk_feature=selectkbest(indep_X,dep_Y)
rfe_feature=rfeFeature(indep_X,dep_Y)

selectk_pca=pca(selectk_feature,dep_Y)
rfe_pca=pca(rfe_feature,dep_Y)

"""SVM"""
classifier,Accuracy,report,X_test,y_test,cm=svm(selectk_feature,indep_X,dep_Y)
print("Svm_sk:",Accuracy)
scores=kfoldd(classifier,selectk_feature,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))


classifier,Accuracy,report,X_test,y_test,cm=svm(rfe_feature,indep_X,dep_Y)
print("svm_rfe:",Accuracy)
scores=kfoldd(classifier,rfe_feature,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))

classifier,Accuracy,report,X_test,y_test,cm=svm(selectk_pca,indep_X,dep_Y)
print("svm_sk_pca:",Accuracy)
scores=kfoldd(classifier,selectk_pca,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))
classifier,Accuracy,report,X_test,y_test,cm=svm(rfe_pca,indep_X,dep_Y)
print("svm_rfe_pca:",Accuracy)
scores=kfoldd(classifier,rfe_pca,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))

"""Random Forest"""
classifier,Accuracy,report,X_test,y_test,cm=random(selectk_feature,indep_X,dep_Y)
print("Random_sk:",Accuracy)
scores=kfoldd(classifier,selectk_feature,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))

classifier,Accuracy,report,X_test,y_test,cm=random(rfe_feature,indep_X,dep_Y)
print("Random_rfe:",Accuracy)
scores=kfoldd(classifier,rfe_feature,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))

classifier,Accuracy,report,X_test,y_test,cm=random(selectk_pca,indep_X,dep_Y)
print("Random_sk_pca:",Accuracy)
scores=kfoldd(classifier,selectk_pca,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))
classifier,Accuracy,report,X_test,y_test,cm=random(rfe_pca,indep_X,dep_Y)
print("Random_rfe_pca:",Accuracy)
scores=kfoldd(classifier,rfe_pca,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))

"""Decision Tree"""

classifier,Accuracy,report,X_test,y_test,cm=Decision(selectk_feature,indep_X,dep_Y)
print("Decision_sk:",Accuracy)
scores=kfoldd(classifier,selectk_feature,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))

classifier,Accuracy,report,X_test,y_test,cm=Decision(rfe_feature,indep_X,dep_Y)
print("Decision_rfe:",Accuracy)
scores=kfoldd(classifier,rfe_feature,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))

classifier,Accuracy,report,X_test,y_test,cm=Decision(selectk_pca,indep_X,dep_Y)
print("Decision_sk_pca:",Accuracy)
scores=kfoldd(classifier,selectk_pca,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))
classifier,Accuracy,report,X_test,y_test,cm=Decision(rfe_pca,indep_X,dep_Y)
print("Decision_rfe_pca:",Accuracy)
scores=kfoldd(classifier,rfe_pca,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))

"""Navies bay"""

classifier,Accuracy,report,X_test,y_test,cm=naives(selectk_feature,indep_X,dep_Y)
print("Naives_sk:",Accuracy)
scores=kfoldd(classifier,selectk_feature,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))
classifier,Accuracy,report,X_test,y_test,cm=naives(rfe_feature,indep_X,dep_Y)
print("Naives_rfe:",Accuracy)
scores=kfoldd(classifier,rfe_feature,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))

classifier,Accuracy,report,X_test,y_test,cm=naives(selectk_pca,indep_X,dep_Y)
print("Naives_sk_pca:",Accuracy)
scores=kfoldd(classifier,selectk_pca,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))
classifier,Accuracy,report,X_test,y_test,cm=naives(rfe_pca,indep_X,dep_Y)
print("Naives_rfe_pca:",Accuracy)
scores=kfoldd(classifier,rfe_pca,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))

"""knn"""

classifier,Accuracy,report,X_test,y_test,cm=knn(selectk_feature,indep_X,dep_Y)
print("knn_sk:",Accuracy)
scores=kfoldd(classifier,selectk_feature,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))

classifier,Accuracy,report,X_test,y_test,cm=knn(rfe_feature,indep_X,dep_Y)
print("knn_rfe:",Accuracy)
scores=kfoldd(classifier,rfe_feature,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))

classifier,Accuracy,report,X_test,y_test,cm=knn(selectk_pca,indep_X,dep_Y)
print("knn_sk_pca:",Accuracy)
scores=kfoldd(classifier,selectk_pca,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))
classifier,Accuracy,report,X_test,y_test,cm=knn(rfe_pca,indep_X,dep_Y)
print("knn_rfe_pca:",Accuracy)
scores=kfoldd(classifier,rfe_pca,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))

"""Logistc"""


classifier,Accuracy,report,X_test,y_test,cm=logistics(selectk_feature,indep_X,dep_Y)
print("Logistics_sk:",Accuracy)
scores=kfoldd(classifier,selectk_feature,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))

classifier,Accuracy,report,X_test,y_test,cm=logistics(rfe_feature,indep_X,dep_Y)
print("Logistics_rfe:",Accuracy)
scores=kfoldd(classifier,rfe_feature,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))

classifier,Accuracy,report,X_test,y_test,cm=logistics(selectk_pca,indep_X,dep_Y)
print("Logistics_sk_pca:",Accuracy)
scores=kfoldd(classifier,selectk_pca,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))
classifier,Accuracy,report,X_test,y_test,cm=logistics(rfe_pca,indep_X,dep_Y)
print("Logistics_rfe_pca:",Accuracy)
scores=kfoldd(classifier,rfe_pca,5)
print('Cross-validated scores: {}\n'.format(scores))
print("5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))




