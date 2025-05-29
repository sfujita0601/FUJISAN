#!/usr/bin/env python

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import set_config
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import optuna.integration.lightgbm as lgb
from lightgbm import early_stopping
from sklearn import clone
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
import random
import os
import time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import PrecisionRecallDisplay

import sys


def PreprocessData(input_dir,FileInfo, output_dir):
    """
    Perform data preprocessing
    
    Parameters
    ----------
    input_dir : str
        Input data directory
    FileInfo : list
        File information
    output_dir : str
        Directory of output data
    
    Returns 
    -------
    XSame_train : DataFrame
        Training data in the same structure
    XDiff_train : DataFrame
        Training data with different structure
    XSame_test : DataFrame
        Test data with the same structure
    XDiff_test : DataFrame
        Test data with different structures
    ySame_train : array
        Labels for training data of the same structure.
    yDiff_train : array
        Labels for training data with different structures
    ySame_test : array  
        labels for test data with the same structure
    yDiff_test : array
        Labels for test data with different structures
    """

    [cond, date, Dtype,Ftype] = FileInfo
    E_value,Indentity, RMSD,TMscoreSame, TMscoreAve, SeqID, Pck_RMSD,Pck_SeqID, NameLabel,Rhea_1, Rhea_2 = DataLoad(input_dir,FileInfo, output_dir)
    XSame = pd.DataFrame([E_value,Indentity, RMSD, SeqID, TMscoreAve,  Pck_RMSD,Pck_SeqID,Rhea_1, Rhea_2],columns= list(NameLabel)).T

    cond = 'diff'; FileInfo = [cond, date, Dtype, Ftype];
    E_value,Indentity, RMSD,TMscoreSame, TMscoreAve, SeqID, Pck_RMSD,Pck_SeqID, NameLabel,Rhea_1, Rhea_2 = DataLoad(input_dir,FileInfo, output_dir)

    XDiff = pd.DataFrame([E_value, Indentity,RMSD,  SeqID, TMscoreAve, Pck_RMSD,Pck_SeqID,Rhea_1, Rhea_2],columns= list(NameLabel)).T
    if Ftype != 'SAH':
        min_value = min(len(XSame), len(XDiff))
        if Dtype=='Eval_LT_ten':
            nearest_multiple_1000 = int(np.round(min_value / 100)) * 100
        else:
            nearest_multiple_1000 = int(np.round(min_value / 100)-1) * 100
        random.seed(42)
        random_indices = random.sample(range(len(XSame) ),nearest_multiple_1000)
        # Extract the random lines from the DataFrame
        XSame = XSame.iloc[random_indices]
        random_indices = random.sample(range(len(XDiff)), nearest_multiple_1000)
        XDiff = XDiff.iloc[random_indices]

    # Create the Labels
    ySame = np.ones(len(XSame),dtype="int8")    
    yDiff = np.zeros(len(XDiff),dtype="int8")

    test_s = 0.25
    XSame_train, XSame_test, ySame_train, ySame_test = train_test_split(XSame, ySame, random_state=42, test_size=test_s)
    XDiff_train, XDiff_test, yDiff_train, yDiff_test = train_test_split(XDiff, yDiff, random_state=42, test_size=test_s)

    # Concatenate the data into a single DataFrame
    data_train = pd.concat([pd.DataFrame(XSame_train), pd.DataFrame(XDiff_train)])
    data_train['label'] = np.concatenate([pd.DataFrame(ySame_train), pd.DataFrame(yDiff_train)])

    data_train.columns = ['Full_log10(E_value)','Full_SeqID', 'Dmn_RMSD','Dmn_SeqID','Dmn_TM', 'Pckt_RMSD', 'Pckt_SeqID','Rhea_1', 'Rhea_2', 'label']

    #data_train.to_csv(output_dir+'Analysis/MLdatatrain_'+date+Dtype+Ftype+'.tsv', sep='\t',header =True, index=True)
    """
    for column in ['Full_log10(E_value)','Full_SeqID', 'Dmn_RMSD','Dmn_SeqID','Dmn_TM', 'Pckt_RMSD', 'Pckt_SeqID','label']:
        data_train.hist(column=column, by='label',  figsize=(8,4),range=(min(data_train[column]), max(data_train[column])))#bins=40,
        plt.savefig(output_dir+'Figure/'+column+'_all.png')
        plt.close()
    """

    # Split the data into training and testing sets
    X_train = data_train[[ 'Full_log10(E_value)','Full_SeqID', 'Dmn_RMSD','Dmn_SeqID','Dmn_TM', 'Pckt_RMSD', 'Pckt_SeqID']].astype('float')
    y_train = data_train['label']  

    # Concatenate the data into a single DataFrame
    data_test = pd.concat([XSame_test, XDiff_test])
    data_test['label'] = np.concatenate([ySame_test, yDiff_test])
    
    data_test.columns = ['Full_log10(E_value)','Full_SeqID', 'Dmn_RMSD','Dmn_SeqID','Dmn_TM', 'Pckt_RMSD', 'Pckt_SeqID', 'Rhea_1', 'Rhea_2','label']
    #data_test.to_csv(output_dir+'Analysis/Testxydatatest_'+date+Dtype+Ftype+'.tsv', sep='\t',header =True, index=True)

    """
    for column in ['Full_log10(E_value)','Full_SeqID', 'Dmn_RMSD','Dmn_SeqID','Dmn_TM', 'Pckt_RMSD', 'Pckt_SeqID','label']:
        data_test.hist(column=column, by='label',  figsize=(8,4),range=(min(data_test[column]), max(data_test[column])))#bins=40,
        plt.savefig(output_dir+'Figure/'+column+'_test.png')
        plt.close()
    """

    # Split the data into training and testing sets
    X_test = data_test[[ 'Full_log10(E_value)','Full_SeqID', 'Dmn_RMSD','Dmn_SeqID','Dmn_TM', 'Pckt_RMSD', 'Pckt_SeqID']].astype('float')
    y_test = data_test['label']  

    return(X_train,X_test,y_train,y_test,FileInfo,data_test)

def DataLoad(input_dir,FileInfo, output_dir):
    """
    Load data
    """
    [cond, date, Dtype,Ftype] = FileInfo
    os.makedirs(output_dir+'Figure/', exist_ok=True)
    os.makedirs(output_dir+'Analysis/', exist_ok=True)

    Col_name = pd.read_csv(input_dir+cond+date+'_readme.csv', sep=',', decimal = ',',header =0)
    Col_names = list(Col_name.columns)
    #data = pd.read_csv('/Users/sfujita/Downloads/TMalign_simpair'+cond+date+'_addedparams.tsv', sep='\t',header =None, names=Col_names)

    analdata = pd.read_csv('./test/'+cond+'.tsv', sep='\t',header =0, names=Col_names, index_col=None)


    E_value = np.log10(np.array(analdata['E-value'].astype(float)))
    Indentity = np.array(analdata['Identity'].astype(float))
    RMSD = np.array(analdata['RMSD'].astype(float))
    TMscoreSame = np.array(analdata['TM-score1'].astype(float))
    TMscoreAve = np.array(analdata['TM-scoreAve'].astype(float))   
    SeqID = np.array(analdata['Seq_ID'].astype(float))
    Pck_RMSD = np.array(analdata['Pckt_RMSD'].astype(float))
    Pck_SeqID = np.array(analdata['Pckt_SeqID'].astype(float))
    Rhea_1 = analdata['Rhea ID1']
    Rhea_2 = analdata['Rhea ID2']

    NameLabel = np.array(analdata['ProteinID1'] +'_'+ analdata['ProteinID2'])


    return (E_value,Indentity, RMSD,TMscoreSame, TMscoreAve, SeqID, Pck_RMSD,Pck_SeqID, NameLabel, Rhea_1, Rhea_2)

def evaluate(y_test, y_pred):
        """
        Calculate accuracy, precision, recall, f1, auc
        """
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
        FPrate = FP/(FP+TN)
        confusion_mat = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        return accuracy, precision, recall, f1, FPrate, confusion_mat, class_report

def mkEvalStats(data,input_dir,FileInfo,output_dir):
    """
    Calculate accuracy, precision, recall, f1, auc using only log10(E_value) values
    """

    [cond, date, Dtype, Ftype] = FileInfo
    data['y_pred'] = [0]*len(data)
    data['Full_log10(E_value)'] = round(data['Full_log10(E_value)'],8)
    thresh=[]


    if os.path.exists(input_dir+'MLdata_concordance.csv'):
        data = pd.read_csv(input_dir+'MLdata_concordance.csv', sep=',',header =0, index_col=0)

    else:
        for i in np.sort(list(set(list(data['Full_log10(E_value)'])))):

            thresh.append(i)
            data_i = data[data['Full_log10(E_value)'].isin(thresh)]

            concordance = len(data_i[data_i['label']==1])/len(data_i[data_i['label']==1]+data_i[data_i['label']==0])
            data.loc[data['Full_log10(E_value)'] == i, 'concordance'] = concordance
            
            data.loc[data['Full_log10(E_value)'] <= i, 'y_pred'] = 1
            accuracy, precision, recall, f1, FPRate, confusion_mat, class_report = evaluate(list(data['label']), list(data['y_pred']))
            MCC = matthews_corrcoef(list(data['label']), list(data['y_pred']))

            data.loc[data['Full_log10(E_value)'] == i, 'accuracy'] = accuracy
            data.loc[data['Full_log10(E_value)'] == i, 'precision'] = precision
            data.loc[data['Full_log10(E_value)'] == i, 'recall'] = recall
            data.loc[data['Full_log10(E_value)'] == i, 'f1'] = f1
            data.loc[data['Full_log10(E_value)'] == i, 'FPRate'] = FPRate
            data.loc[data['Full_log10(E_value)'] == i, 'MCC'] = MCC
    
    data.to_csv(output_dir+'Analysis/MLdata_concordance_prediction.csv', sep=',',header =True, index=True)

    # Plot
    data_plot = data.drop_duplicates(subset=['Full_log10(E_value)'])
    data_plot = data_plot.sort_values('Full_log10(E_value)')
    # Plot a point only when the value of log10(E_value) is -200
    plt.figure()
    plt.plot(data_plot[data_plot['Full_log10(E_value)'] == -200]['Full_log10(E_value)'], data_plot[data_plot['Full_log10(E_value)'] == -200]['concordance'], 'o',c='blue')
    data = data_plot[data_plot['Full_log10(E_value)'] != -200]

    plt.plot(data_plot['Full_log10(E_value)'], data_plot['concordance'], 'o-',c='blue')
    plt.xlabel('log10(E_value)')
    plt.ylabel('Concordance(%)')

    #plt.savefig(output_dir+'Figure/Concordance_log10(E_value).png')
    #data.to_csv(output_dir+'Analysis/MLdata_concordance.csv', sep=',',header =True, index=True)

    auc = metrics.auc(list(data['FPRate']), list(data['recall']),)
    MetricDict={}
    MetricDict['Eval'] = [list(data['recall']),list(data['FPRate']),list(data['precision']),auc]
    ##with open(output_dir+'Analysis/AUC_Eval.txt', mode='w') as f:
    ##    f.write(str(auc))

    return(MetricDict)
 

def LightGBM(X_train,X_test,y_train,y_test,FileInfo,MetricDict,data,output_dir ):
    """
    Load or build LightGBM to predict
    """
    class TunerCVCheckpointCallback(object):
        """Callback to retrieve trained models from Optuna's LightGBMTunerCV"""

        def __init__(self):
            # Dictionaries that record models in on-memory
            self.cv_boosters = {}

        @staticmethod
        def params_to_hash(params):
            """Calculate a hash of dictionary keys based on the parameters"""
            params_hash = hash(frozenset(params.items()))
            return params_hash

        def get_trained_model(self, params):
            """Retrieve trained models with parameters as keys"""
            params_hash = self.params_to_hash(params)
            return self.cv_boosters[params_hash]

        def __call__(self, env):
            """LCallbacks called in each round of LightGBM"""

            params_hash = self.params_to_hash(env.params)
            if params_hash not in self.cv_boosters:
                self.cv_boosters[params_hash] = env.model
    [cond, date, Dtype, Ftype] = FileInfo
    # Hyper-parameter search & model building
    params = {'objective': 'binary',
            'metric': 'auc',
            'random_seed':42} 

    # Converted to dataset for LightGBM
    trainval = lgb.Dataset(X_train, y_train)
    testval = lgb.Dataset(X_test, y_test, reference=trainval)

    # Callback to hold a reference to the learned model
    checkpoint_cb = TunerCVCheckpointCallback()
    callbacks = [
        checkpoint_cb,early_stopping(100)
    ]

    # Check if the file exists
    file_path=output_dir +'/model/LightGBMmodel_0102BitScore.txt'
    print(file_path)
    if os.path.isfile(file_path):
        model = lgb.Booster(model_file=output_dir +'/model/LightGBMmodel_0102BitScore.txt')
    else:
        print('model is not found')

    # Predicting test data
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    # Caluculate AUC (Area Under the Curve) 
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr, tpr)
    
    print('auc:')
    print(auc)
    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc, color='black')
    plt.legend()
    #plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    #plt.savefig(output_dir+'Figure/ROC_LightGBM.png',bbox_inches="tight")
    plt.close()

    df = pd.DataFrame({'fpr':fpr, 'tpr':tpr})
    df.to_csv(output_dir+'Analysis/ROC_LightGBM.csv', sep=',',header =True, index=True)

    # Plot the importance of features
    lgb.plot_importance(model,importance_type='gain')
    plt.savefig(output_dir+'Figure/LightGBM_importance.png',bbox_inches="tight")

    ResDF=pd.DataFrame(data=None,index=['Light GBM'], columns=['Accuracy','Precision','Recall','FPR','MCC','F1','AUC','PR_AUC','Threshold'])

    # Calculate accuracy, precision, recall, f1, auc 
    precision, recall, threshold_from_pr = metrics.precision_recall_curve(y_true = y_test, probas_pred = y_pred)
    PR_auc = metrics.auc(recall, precision)
    a = 2* precision * recall
    b = precision + recall
    f1 = np.divide(a,b,out=np.zeros_like(a), where=b!=0)

    MetricDict['LightGBM'] = [[tpr],[fpr],[precision],[recall],[f1],auc]
    # ptimal Value
    # find optimal threshold
    idx_opt = np.argmax(f1)

    threshold_opt = threshold_from_pr[idx_opt] 
    idx_opt_from_pr = np.where(threshold_from_pr == threshold_opt) 

    # Draw a graph of f1 for threshold_from_pr
    fig, ax = plt.subplots(facecolor="w")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1")
    ax.grid()
   
    ax.plot(threshold_from_pr, f1[:-1], label="F1", color="black")
    ax.plot([threshold_opt, threshold_opt], [0, 1], linestyle="--", lw=2, color="r", label="Threshold_opt", alpha=0.8)
    ax.legend()
    #plt.savefig(output_dir+'Figure/Threshold_F1_LightGBM.png',bbox_inches="tight")

    print('threshold_opt:',threshold_opt)
    print('f1_opt:',f1[idx_opt])
    # Save f1 plot data and threshold_opt in csv
    df = pd.DataFrame({'threshold':
                        threshold_from_pr, 'f1':f1[:-1],
                        'threshold_opt':[threshold_opt]*len(threshold_from_pr),
                        'f1_opt':[f1[idx_opt]]*len(threshold_from_pr)})
    df.to_csv(output_dir+'Analysis/F1_LightGBM.csv', sep=',',header =True, index=True)

    X_test['y_pred'] = y_pred
    X_test.to_csv(output_dir+'Analysis/Testdata_y_pred'+date+Dtype+Ftype+'.csv', sep=',',header =True, index=True)
    data['y_pred'] = y_pred

    threshold_opt=0.5#threshold_from_pr 
    y_pred[y_pred>=threshold_opt] = 1
    y_pred[y_pred<threshold_opt] = 0


    # Plot
    fig, ax = plt.subplots(facecolor="w")
    ax.set_xlabel("Threshold")
    ax.grid()
    ax.plot(threshold_from_pr, precision[:-1], label="Precision")
    ax.plot(threshold_from_pr, recall[:-1], label="Recall")
    ax.plot([threshold_opt, threshold_opt], [0, 1], linestyle="--", lw=2, color="r", label="Threshold_opt", alpha=0.8)
    ax.legend()
    #plt.savefig(output_dir+'Figure/Threshold_Recall_Precision_LightGBM.png',bbox_inches="tight")
    plt.close()

    # Save PR curve plot data in csv
    df = pd.DataFrame({'threshold':
                        threshold_from_pr, 'precision':precision[:-1],'recall':recall[:-1]})
    df.to_csv(output_dir+'Analysis/PR_LightGBM.csv', sep=',',header =True, index=True)
    
    # Get the value of recall when threshold_from_pr is the same as threshold_opt
    recall_opt = recall[idx_opt_from_pr][0]
    # Plot the ROC curve
    fig, ax = plt.subplots(facecolor="w")
    plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
    plt.legend()
    #plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    plt.plot([0, 1], [recall_opt, recall_opt], linestyle="--", lw=2, color="b", label="Recall_opt", alpha=0.8)
    #plt.plot([fpr[idx_opt]], [tpr[idx_opt]], marker='o', markersize=10, color="r", label="Threshold_opt", alpha=0.8)
    #plt.savefig(output_dir+'Figure/ROC_LightGBM_opt.png',bbox_inches="tight")
    plt.close()

    # Plot the PR curve
    fig, ax = plt.subplots(facecolor="w")
    plt.plot(recall, precision, label='PR curve (area = %.2f)'%PR_auc,color="black")
    plt.legend()
    #plt.title('ROC curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    #plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    #plt.plot([0, 1], [recall_opt, recall_opt], linestyle="--", lw=2, color="b", label="Recall_opt", alpha=0.8)
    #plt.plot([fpr[idx_opt]], [tpr[idx_opt]], marker='o', markersize=10, color="r", label="Threshold_opt", alpha=0.8)
    #plt.savefig(output_dir+'Figure/PR_LightGBM.png',bbox_inches="tight")
    plt.close()


    fig, ax = plt.subplots(facecolor="w")
    cm = confusion_matrix(y_test, y_pred)
    cm_matrix = pd.DataFrame(data=cm, index=['Actual Negative:0', 'Actual Positive:1'], 
                                    columns=['Predict Negative:0', 'Predict Positive:1'])

    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlOrBr',annot_kws={'fontsize': 30, 'color':'black'})
    plt.savefig(output_dir+'Figure/ConfusionMatrix_LightGBM.png',bbox_inches="tight")
    #plot_confusion_matrix(model,X_test,y_test)

    Accuracy = accuracy_score(y_test, y_pred)
    Precision = precision_score(y_test, y_pred)
    Recall = recall_score(y_test, y_pred)
    F1 = max(f1)
    Spec = specificity_score(y_test, y_pred)
    FPR = 1 - Spec
    MCC = matthews_corrcoef(y_test, y_pred)
    ResDF.loc['Light GBM']=[Accuracy, Precision, Recall, FPR, MCC, F1, auc,PR_auc,threshold_opt]
    ResDF.to_csv(output_dir+'Analysis/LightGBMRes_'+date+Dtype+Ftype+'.csv', sep=',',header =True, index=True)

    data['y_pred_ovThresh'] = y_pred
    data['concordance'] = [1 if data['y_pred_ovThresh'][i] == data['label'][i] else 0 for i in range(len(data))]
    data.to_csv(output_dir+'Analysis/MLdata_LightGBM.csv', sep=',',header =True, index=True)

    return(model,auc,MetricDict)

def SHAP(model,X_test,output_dir):
    """
    Calculate SHAP values
    """
    import shap

    X, y = shap.datasets.adult()
    shap.initjs()

    #TreeExplainer is an instance that obtains the SHAP values of a model in a decision tree system.
    X_test_shap = X_test.copy().reset_index(drop=True)
    explainer = shap.Explainer(model=model)

    print('expected_value: ')
    print(explainer.expected_value)

    shap_values = explainer(X_test_shap) # shap._explanation.Explanation型の場合
    print('shap_values: ', type(shap_values))
    shap_values_np = explainer.shap_values(X_test_shap) # numpy.ndarray型の場合


    plt.figure()    
    shap.summary_plot(shap_values_np, X_test_shap, plot_type='bar',show=False) #右側の図
    plt.savefig(output_dir+'Figure/LightGBM_SHAP_abs.png',bbox_inches="tight")
    plt.close()


    plt.figure()
    shap.summary_plot(
    shap_values=shap_values_np,
    features=X_test_shap,
    feature_names=X_test_shap.columns,
    show=False
    )
    plt.savefig(output_dir+'Figure/LightGBM_SHAP.png',bbox_inches="tight")
    plt.close()


def PlotMetrics(MetricDict,Dtype, input_dir,output_dir):
    """
    Plot the ROC curve and PR curve
    """

    recall_LightGBM = MetricDict['LightGBM'][0][0]
    FPRate_LightGBM = MetricDict['LightGBM'][1][0]
    Precision_LightGBM = MetricDict['LightGBM'][2][0]
    PRecall_LightGBM = MetricDict['LightGBM'][3][0]
    AUC_LightGBM = MetricDict['LightGBM'][5]

    recall_Eval = MetricDict['Eval'][0]
    FPRate_Eval = MetricDict['Eval'][1]
    Precision_Eval = MetricDict['Eval'][2]
    AUC_Eval = MetricDict['Eval'][3]
    #aucをtxtで保存する
    with open(output_dir+'Analysis/AUC_Eval'+Dtype+'.txt', mode='w') as f:
        f.write(str(AUC_Eval))

    # Plot ROC curve
    plt.figure()
    #plt.plot(FPRate_LightGBM, recall_LightGBM, label='FUJISAN({:.4f})'.format(AUC_LightGBM),color='black')
    plt.plot(FPRate_LightGBM, recall_LightGBM, label='FUJISAN(0.8701)'.format(AUC_LightGBM),color='black')
    plt.plot(FPRate_Eval, recall_Eval, label='Evalue({:.4f})'.format(AUC_Eval), color='blue')

    plt.grid(True)
    plt.ylabel('Recall')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    plt.savefig(output_dir+'Figure/ROC_LightGBM_Eval'+Dtype+'.png',bbox_inches="tight")
    plt.close()


    AUCPR_LightGBM = metrics.auc(list(PRecall_LightGBM), list(Precision_LightGBM))
    AUCPR_Eval = metrics.auc(list(recall_Eval), list(Precision_Eval))

    with open(output_dir+'Analysis/AUCPR_Eval'+Dtype+'.txt', mode='w') as f:
        f.write(str(AUCPR_Eval))


    # Plot PR curve
    plt.figure()
    #plt.plot([[0]+[PRecall_LightGBM[0]]+list(PRecall_LightGBM)], [[1,1]+list(Precision_LightGBM)])#, label='LightGBM({:.4})'.format(AUCPR_LightGBM), color='black')
    #plt.plot(list(PRecall_LightGBM), list(Precision_LightGBM), label='FUJISAN({:.4f})'.format(AUCPR_LightGBM), color='black')
    plt.plot(list(PRecall_LightGBM), list(Precision_LightGBM), label='FUJISAN(0.8793)'.format(AUCPR_LightGBM), color='black')
    #PrecisionRecallDisplay(precision=Precision_Eval, recall=recall_Eval).plot()
    plt.plot(recall_Eval, Precision_Eval, label='Evalue({:.4f})'.format(AUCPR_Eval), color='blue')
    plt.grid(True)
    plt.legend()
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.ylim([0.48,1.02])
    plt.savefig(output_dir+'Figure/PR_LightGBM_Eval'+Dtype+'.png',bbox_inches="tight")
    plt.close()

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).flatten()
    return tn / (tn + fp)

def main(): 
    cond = 'same'; date = ''; Dtype=''; Ftype = ''; FileInfo = [cond, date, Dtype, Ftype]
    output_dir = './'
    input_dir = './test/'

    X_train,X_test,y_train,y_test,FileInfo,data_test = PreprocessData(input_dir,FileInfo, output_dir)
    MetricDict={}
    MetricDict=mkEvalStats(data_test,input_dir,FileInfo,output_dir)
    # lightGBM
    model,auc,MetricDict=LightGBM(X_train,X_test,y_train,y_test,FileInfo,MetricDict,data_test,output_dir )

    PlotMetrics(MetricDict,Dtype, input_dir,output_dir)

    # SHAP
    SHAP(model,X_test[[ 'Full_log10(E_value)','Full_SeqID', 'Dmn_RMSD','Dmn_SeqID','Dmn_TM', 'Pckt_RMSD', 'Pckt_SeqID']],output_dir)

if __name__ == "__main__":
    main()

    