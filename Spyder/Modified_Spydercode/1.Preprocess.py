import operator
import keras
from fancyimpute import KNN 
from sklearn.preprocessing import LabelBinarizer
import math
from operator import itemgetter 
import numpy as np
import pandas as pd
import seaborn as sns
#from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, classification_report, r2_score, make_scorer, roc_curve, auc
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score, StratifiedKFold, KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression

columns = [
    # nominal
    'gender', #0-1
    'symptoms', #0-1
    'alcohol', #0-1
    'hepatitis b surface antigen', #0-1
    'hepatitis b e antigen', #0-1
    'hepatitis b core antibody', #0-1
    'hepatitis c virus antibody', #0-1
    'cirrhosis', #0-1
    'endemic countries', #0-1
    'smoking', #0-1
    'diabetes', #0-1
    'obesity', #0-1
    'hemochromatosis', #0-1
    'arterial hypertension', #0-1
    'chronic renal insufficiency', #0-1
    'human immunodeficiency virus', #0-1
    'nonalcoholic steatohepatitis', #0-1
    'esophageal varices', #0-1
    'splenomegaly', #0-1
    'portal hypertension', #0-1
    'portal vein thrombosis', #0-1
    'liver metastasis', #0-1
    'radiological hallmark', #0-1
    
    # integer
    'age', # age at diagnosis
    
    # continuous
    'grams of alcohol per day',
    'packs of cigarets per year',
    
    # ordinal
    'performance status',
    'encephalopathy degree',
    'ascites degree',
     
    # continuous   
    'international normalised ratio',
    'alpha-fetoprotein',
    'haemoglobin',
    'mean corpuscular volume',
    'leukocytes',
    'platelets',
    'albumin',
    'total bilirubin',
    'alanine transaminase',
    'aspartate transaminase',
    'gamma glutamyl transferase',
    'alkaline phosphatase',
    'total proteins',
    'creatinine',
    
    # integer
    'number of nodules',
    
    # continuous
    'major dimension of nodule cm',
    'direct bilirubin mg/dL',
    'iron',
    'oxygen saturation %',
    'ferritin',
        
    #nominal
    'class attribute', #0-1
]
columns = list([x.replace(' ', '_').strip() for x in columns])

df = pd.read_csv('hcc-data.csv', names=columns, header=None, na_values=['?'])


data = df.copy()

print("Null values colunt for each column")

data.isnull().sum(axis=0)

def prepare_missing_values_for_nans(df=None, columns=None):
    """
    Looking for the most frequent value for both decision classes outputs - 0,1.
    """
    
    to_update_nans_dict = {}
    
    if columns:
        for decision_class in [0, 1]:
            for column in columns:
                vals = df[df.class_attribute == decision_class][column].value_counts()
                
                to_update_nans_dict['{decision_class}_{column}'.format(
                    decision_class=decision_class,
                    column=column
                )] = vals.idxmax()
                
        return to_update_nans_dict
    
def replace_missing_values(df=None, columns=None, to_update_nans_dict=None):
    """
    Replacing NaN with the most frequent values for both decission classes outputs - 0,1.
    """
    
    df_list = []
    
    if columns:
        for decision_class in [0, 1]:
            _df = df[df.class_attribute == decision_class].reset_index(drop=True)

            for column in columns:        
                _df[column] = _df[column].fillna(
                    to_update_nans_dict['{}_{}'.format(decision_class, column)]
            )

            df_list.append(_df)

        return df_list
    
nominal_indexes = [
    1, 3, 4, 5, 
    6, 8, 9, 10, 
    11, 12, 13, 
    14, 15, 16, 
    17, 18, 19, 
    20, 21, 22
]

nominal_columns_to_discretize = list(itemgetter(*nominal_indexes)(columns))

cons = data.loc[:, :]

cons['null_values'] = cons.isnull().sum(axis=1)



data2 = data.drop(columns=['null_values'])

nominal_dict = prepare_missing_values_for_nans(df=data2, columns=nominal_columns_to_discretize)

missing_nominal_values_list = replace_missing_values(
    df=data2,
    columns=nominal_columns_to_discretize,
    to_update_nans_dict=nominal_dict

)

data2 = pd.concat(missing_nominal_values_list).reset_index(drop=True)



continuous_indexes = [
    24,25,29,30,
    31,32,33,34,
    35,36,37,38,
    39,40,41,42,
    44,45,46,47,
    48]


continuous_columns_to_discretize = list(
    itemgetter(*continuous_indexes)(columns)
)

continuous_data = data2[continuous_columns_to_discretize].as_matrix()



X_filled_knn = KNN(k=3).fit_transform(continuous_data)

data2[continuous_columns_to_discretize] = X_filled_knn

X_filled_knn.shape

integer_columns = ['age', 'number_of_nodules']

# prepare missing integer values
integer_dict = prepare_missing_values_for_nans(
    df=data2, 
    columns=integer_columns
)

missing_integer_values_list = replace_missing_values(
    df=data2,
    columns=integer_columns,
    to_update_nans_dict=integer_dict

)

data2 = pd.concat(missing_integer_values_list).reset_index(drop=True)

data2['ascites_degree'].value_counts()

ordinal_columns = ['encephalopathy_degree', 'ascites_degree', 'performance_status']

ordinal_dict = prepare_missing_values_for_nans(
    df=data2, 
    columns=ordinal_columns
)

missing_ordinal_values_list = replace_missing_values(
    df=data2,
    columns=ordinal_columns,
    to_update_nans_dict=ordinal_dict

)

data2 = pd.concat(missing_ordinal_values_list).reset_index(drop=True)

data2[data2.isnull().any(axis=1)]

ordinal_columns

binarized_data = []

for c in ordinal_columns:
    lb = LabelBinarizer()
    
    lb.fit(data2[c].values)
    
    binarized = lb.transform(data2[c].values)
    binarized_data.append(binarized)

binarized_ordinal_matrix_data = pd.DataFrame(np.hstack(binarized_data))

list(set(data2.number_of_nodules.values))

lb = LabelBinarizer()

lb.fit(data2.number_of_nodules.values)

binarized_number_of_nodules = pd.DataFrame(lb.transform(data2.number_of_nodules.values))


data2['age_'] = data2.age.apply(lambda x: x / data2.age.max())


data2['age_'].head(10)

age_ = data2.age_.values.reshape(-1,1)

to_drop_columns = [
    'age', 
    'encephalopathy_degree', 
    'ascites_degree', 
    'performance_status', 
    'number_of_nodules'
]

columns_set = set(columns)

columns_ = list(columns_set.difference(to_drop_columns))

len(columns)
#len(_columns)

data2.to_csv("Prepocessed_HCC.csv",index=False)

X = pd.DataFrame(data2[columns_].as_matrix())
y = pd.DataFrame(data2.class_attribute.values)

binary_columns=['encephalopathy_degree_1','encephalopathy_degree_2','encephalopathy_degree_3',
                'ascites_degree_1','ascites_degree_2','ascites_degree_3','performance_status_1',
                'performance_status_2','performance_status_3','performance_status_4','performance_status_5']
nodules_coulmns=['number_of_nodule_0','number_of_nodule_1','number_of_nodule_2',
                 'number_of_nodule_3','number_of_nodule_4','number_of_nodule_5']

X.columns=columns_
binarized_ordinal_matrix_data.columns=binary_columns


binarized_number_of_nodules.columns=nodules_coulmns

cc=[columns_,binary_columns,binarized_number_of_nodules,age_,]
ccc=['hepatitis_c_virus_antibody',
  'class_attribute',
  'platelets',
  'international_normalised_ratio',
  'hepatitis_b_surface_antigen',
  'packs_of_cigarets_per_year',
  'direct_bilirubin_mg/dL',
  'haemoglobin',
  'portal_vein_thrombosis',
  'arterial_hypertension',
  'total_bilirubin',
  'human_immunodeficiency_virus',
  'total_proteins',
  'chronic_renal_insufficiency',
  'major_dimension_of_nodule_cm',
  'leukocytes',
  'smoking',
  'symptoms',
  'endemic_countries',
  'obesity',
  'aspartate_transaminase',
  'creatinine',
  'ferritin',
  'diabetes',
  'cirrhosis',
  'iron',
  'alkaline_phosphatase',
  'nonalcoholic_steatohepatitis',
  'mean_corpuscular_volume',
  'alpha-fetoprotein',
  'hemochromatosis',
  'portal_hypertension',
  'alanine_transaminase',
  'gamma_glutamyl_transferase',
  'hepatitis_b_core_antibody',
  'oxygen_saturation_%',
  'grams_of_alcohol_per_day',
  'hepatitis_b_e_antigen',
  'esophageal_varices',
  'radiological_hallmark',
  'albumin',
  'liver_metastasis',
  'gender',
  'splenomegaly',
  'alcohol','encephalopathy_degree_1','encephalopathy_degree_2','encephalopathy_degree_3',
                'ascites_degree_1','ascites_degree_2','ascites_degree_3','performance_status_1',
                'performance_status_2','performance_status_3','performance_status_4','performance_status_5',
                'number_of_nodule_0','number_of_nodule_1','number_of_nodule_2',
                 'number_of_nodule_3','number_of_nodule_4','number_of_nodule_5','agee']
                

X_new = pd.DataFrame(np.hstack((X, binarized_ordinal_matrix_data,binarized_number_of_nodules, age_)))
#X_new = pd.DataFrame([X, binarized_ordinal_matrix_data,binarized_number_of_nodules, age_])

X_new.columns=ccc

df2=X_new
X_new.to_csv("finalpre.csv",index=False)

print("After preprocessing : filled null values")
df2.isnull().sum()

X_new=X_new.drop('class_attribute', 1)

#y=df2['class_attribute']

X_new.to_csv("finalpre.csv",index=False)
X_new.shape

std_scaler = StandardScaler() #StandardScaler() # RobustScaler
X_new = std_scaler.fit_transform(X_new)



X_train, X_test, y_train, y_test = train_test_split(
    X_new,
    y,
    random_state=42,
    test_size=0.20
)
log_reg = LogisticRegression(
    solver='lbfgs',
    random_state=42,
    C=0.1,
    multi_class='ovr',
    penalty='l2',
)
log_reg.fit(X_train, y_train)

log_reg_predict = log_reg.predict(X_test)


log_reg.score(X_test, y_test)

preds = log_reg.predict(X_test)
print('\nLogistic Regression Accuracy: {:.2f}%'.format(accuracy_score(y_test, log_reg_predict) * 100))
print('Logistic Regression AUC: {:.2f}%'.format(roc_auc_score(y_test, log_reg_predict) * 100))
print('Logistic Regression Classification report:\n\n', classification_report(y_test, log_reg_predict))

kfold = StratifiedKFold(
    n_splits=3, 
    shuffle=True, 
    random_state=42
)

predicted = cross_val_predict(
    log_reg, 
    X_new, 
    y, 
    cv=kfold
)

scores = cross_val_score(
    log_reg, 
    X_new, 
    y, 
    cv=kfold,
    scoring='f1'
)

print('Cross-validated scores: {}\n'.format(scores))

print(classification_report(y, predicted))

print("LogisticRegression: F1 after 5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))