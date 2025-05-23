# -*- coding: utf-8 -*-
"""
Code to implement a random forest classifier model to predict gully presence 
using pan-European spatial covariates. A random forest classifier is implemented
however the code is also provided to implement catboost.
Author: Francis Matthews fmatthews1381@gmail.com
"""

import geopandas as gpd
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
import shap
from PyALE import ale 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import sys
from functions import tidy_df, plot_correlation_matrix
from catboost import CatBoostClassifier, Pool


version = "v5"
gee_data_exists = False
export_models = False
spatially_balanced = False
#signifies the sampling method describes in the manuscript
downsample_method = "MIXED"
model_type = "RF"
tune_hps = False
cross_validation = True
plot_ale = False
plot_shap = False
plot_learning_curve = False

data_dir = "YOUR_PATH/INPUT_FEATURES"
r_dir = "YOUR_PATH/MODEL"
type_ = downsample_method
dir_gee = os.path.join(data_dir, 'GEE_SPATIALLY_UNBALANCED"')
all_data = pd.read_csv(os.path.join(data_dir, f'ALL_FEATURES_SPATIALLY_UNBALANCED_{version}.csv')).drop(['Unnamed: 0', 'geometry'], axis = 1)
points = gpd.read_file(os.path.join(data_dir, f'GULLY_POINTS_SPATIALLY_UNBALANCED_{version}.shp'))
figure_dir = r_dir

    
sns.set_style("whitegrid")

all_data = all_data.sort_values("POINT_ID")
all_data["POINT_ID"] = all_data["POINT_ID"].astype('int64')
print(len(all_data))

#CHECK COLUMNS ARE ADDED
df_cols_len = len(all_data.columns)

if gee_data_exists == True: 
    for file in os.listdir(dir_gee):
        p = os.path.join(dir_gee, file)
        df = pd.read_csv(p).rename(columns = {"id": "FID"}).drop(['system:index', '.geo', 'Gully'], axis = 1)
        print(p)
        all_data = all_data.merge(df, how = 'left', on = 'FID', validate = 'm:1')
        if len(all_data.columns) == df_cols_len:
            print("MERGING ERROR")
        df_cols_len = len(all_data.columns)
        #print(c.sum())

print(len(all_data))
all_data = all_data.set_index("POINT_ID")

"""
Create a classifier to distinguish between gullies vs no gullies and analyse
the prediction
"""
to_remove = ['POINT_NUTS',
 'POINT_LAT',
 'POINT_LONG',
 'SURVEY_OBS',
 'Gully',
 'index_right',
 'EnS',
 'EnS_name',
 'EnZ',
 'EnZ_name',
 'Area_km2',
 'FID',
 'CC_median_100m.tif']

features = list(all_data.columns)
for x in to_remove:
    features.remove(x)
    
#all of these features need to be downloaded and harmonised for EU prediction
features_processed = ['EU_LS_Mosaic_100m.tif',
 'WATEM_Code_Mean_t_ha_yr_100m.tif',
 'EU_CFactor_final_V7.tif',
 'Kst_EU28.tif',
 'Clay.tif',
 'R-factor September.tif',
 'Coarse_fragments.tif',
 'R-factor August.tif',
 'R-factor July.tif',
 'R-factor January.tif',
 'EU_CropMap_22_v1_stratum_EU27-HR_100m.tif',
 'Silt1.tif',
 'AWC.tif',
 'Sand1.tif',
 'CC_Median.tif',
 'elev-stdev',
 'vrm',
 'tri',
 'slope',
 'tcurv',
 'roughness']
#these features are extracted from Google Earth Engine
features_gee = ['Percent_Tree_Cover',
 'Percent_NonTree_Vegetation',
 'Percent_NonVegetated',
 'SOIL_0',
 'SOIL_1',
 'SOIL_2',
 'TOPO2_0',
 'TOPO2_1',
 'TOPO2_2',
 'TOPO2_3',
 'TOPO2_4',
 'TOPO3_0',
 'TOPO3_1',
 'TOPO3_2',
 'TOPO3_3',
 'TOPO3_4',
 'TOPO3_5',
 'TOPO3_6',
 'TOPO3_7',
 'TOPO3_8',
 'TOPO_0',
 'TOPO_1',
 'TOPO_2',
 'TOPO_3',
 'TOPO_4',
 'TOPO_5',
 'TOPO_6',
 'TOPO_7',
 'TOPO_8',
 'TOPO_9']



replace_vals = {"CC_median_100m.tif": 0}

#set out of bounds areas to 0
#the mask variable is used to filter the LUCAS points - this matters for the no-gully spatial domain
#the mask variable should correspond with the final mask used    
all_data = tidy_df(all_data, mask_variable = 'EU_CFactor_final_V7_100m.tif', 
                   downsample_method = downsample_method, remove_nan_rows = False,
                   replace_vals = None, oversample_ratio = 1)

#plot_correlation_matrix(all_data, features_processed)
plot_correlation_matrix(all_data, features)
class_w = compute_class_weight(class_weight="balanced", classes=np.unique(all_data["Gully"]), y= all_data["Gully"])
class_w = dict(zip(np.unique(all_data["Gully"]), class_w))

model_runs = {"FULL_FEATURES": features}


runs = list(model_runs.keys())

results = {}


for key in runs:
    
    models_dir = os.path.join(r_dir, f'{model_type}_Models')
                 
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    x = all_data[model_runs[key]]
    y = all_data['Gully']

    
    #make copies for the cross validation. index needs to be reset
    x_roc = x.copy().reset_index(drop = True).to_numpy()
    y_roc = y.copy().reset_index(drop = True).to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 1)
        
    X_ale = X_test.fillna(0)
    

    if tune_hps == True:
        clf = RandomForestClassifier()
        #perform a simple grid search hyperparametetr tuning using a typical set of hyperparameters
        #n estimators is set high to optimise the probability estimate
        n_estimators = [int(x) for x in np.linspace(start = 2000, stop = 10000, num = 10)]
        max_features = ['log2', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
        rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, 
                                       n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        
        rf_random.fit(X_train,y_train)
        
        with open(os.path.join(models_dir, f'RF_{key}_{type_}_{version}_hyperparameters.txt'), 'w') as f:
            f.write(json.dumps(rf_random.best_params_))
        
    else:
        if model_type == "RF":
            try:
                with open(os.path.join(models_dir, f'RF_{key}_{type_}_{version}_hyperparameters.txt'),'r') as json_file:
                    rf_params = json.load(json_file)
        
                clf = RandomForestClassifier(n_estimators = rf_params['n_estimators'], min_samples_split =  rf_params['min_samples_split'],
                                             min_samples_leaf = rf_params['min_samples_leaf'], max_features = rf_params['max_features'],
                                             max_depth = rf_params['max_depth'], bootstrap = rf_params['bootstrap'])
            except:
                print("NO FITTED HYPERPARAMETERS FOUND: USING STANDARD MODEL")
                clf = RandomForestClassifier(n_estimators = 10000)

        elif model_type == "CATBOOST":
            
            clf = CatBoostClassifier(loss_function='Logloss',
                           verbose=True)
            
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    auc_score = metrics.roc_auc_score(y_test, y_prob[:, 1])
    clf_report = metrics.classification_report(y_test, y_pred)
    
    
        
    sns.set_theme(font_scale=2)
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    
    if plot_learning_curve == True:
        params = {
        "X": x,
        "y": y,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
        "score_type": "both",
        "n_jobs": 4,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "score_name": "roc_auc",
        }
        
        LearningCurveDisplay.from_estimator(clf, **params)

    
    if cross_validation == True:
        n_splits = 5
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
        
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        fig, ax = plt.subplots(figsize=(16, 13))
        for fold, (train, test) in enumerate(cv.split(x_roc, y_roc)):
            clf_roc = RandomForestClassifier()
            clf_roc.fit(x_roc[train], y_roc[train])
            viz = RocCurveDisplay.from_estimator(
                clf_roc,
                x_roc[test],
                y_roc[test],
                name=f"ROC fold {fold}",
                alpha=0.3,
                lw=1,
                ax=ax,
                plot_chance_level=(fold == n_splits - 1),
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="firebrick",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )
        
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        
        ax.set(
            xlabel="False Positive Gully Classification Rate",
            ylabel="True Positive Gully Classification Rate",
            title="GE-LUCAS: Mean ROC curve from 5-fold cross validation",
        )
        ax.legend(loc="lower right")
        plt.savefig(os.path.join(figure_dir, "cross_validation.svg"), dpi = 300)
        plt.show()
        
        
    
    """
    #testing a logistic regression model
    logisticRegr = LogisticRegression(max_iter=1000)
    logisticRegr.fit(X_train,y_train)
    lr_pred = logisticRegr.predict(X_test)
    print("Accuracy Logistic Regression:",metrics.accuracy_score(y_test, lr_pred))
    """

    feature_imp = pd.DataFrame(clf.feature_importances_, index = x.columns).sort_values(by = [0], ascending=False)
    feature_imp['File_name'] = feature_imp.index
    feature_imp['Feature'] = pd.DataFrame(feature_imp.index.to_list())[0].str.replace('_', ' ').values
    feature_imp.columns = ['Feature importance', 'File_name', 'Feature']
    feature_imp['Name'] = feature_imp['Feature'].str.replace(' 100m.tif', '')
    feature_imp['Name'] = feature_imp['Name'].str.upper()
    feature_imp['Name'] = feature_imp['Name'].str.replace('1', '')
    feature_imp['Name'] = feature_imp['Name'].str.replace(' 22 V STRATUM EU27-HR', '')
    feature_imp['Name'] = feature_imp['Name'].str.replace(' EU28', '')
    
    f, ax = plt.subplots(figsize=(7, 20))
    sns.barplot(data = feature_imp, x = 'Feature importance', y = 'Name', 
                linewidth=2.5, edgecolor=".5", facecolor=(0, 0, 0, 0), ax = ax)
    ax.set_ylabel('Feature name', fontsize=35)
    ax.set_xlabel('Feature importance', fontsize=35)
    plt.savefig(os.path.join(figure_dir, "feature_importance.svg"), dpi = 300)
    
    
    if plot_ale == True:
        for feature in feature_imp.index.values[:10]:
            ale_eff = ale(
            X=X_ale, model=clf, feature=[feature], grid_size=50, include_CI=True)
    
    if plot_shap == True:
        samples = X_train.rename(columns = dict(zip(feature_imp['File_name'].values, feature_imp['Name'].values)))
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(samples, check_additivity=False)
        shap.summary_plot(shap_values[1], samples, cmap=plt.get_cmap("viridis"), 
                          color_bar_label='Predictive feature value (relative)', show=False)
        plt.savefig(os.path.join(figure_dir, "shapley_plot.svg"), dpi = 300)
        plt.show()
    
    #take a sample of 100 
    pred_summary = pd.DataFrame()
    pred_summary['gully_obs'] = y_test 
    pred_summary['gully_pred'] = y_pred
    pred_summary['False_positive'] = np.where(np.logical_and(pred_summary['gully_obs'] == 0, pred_summary['gully_pred'] == 1), 1, 0)
    pred_summary['False_negative'] = np.where(np.logical_and(pred_summary['gully_obs'] == 1, pred_summary['gully_pred'] == 0), 1, 0)
    pred_summary['Pred_overview'] = 'Correct prediction'
    pred_summary['Pred_overview'] = np.where(pred_summary['False_positive'] == 1, 'False positive', pred_summary['Pred_overview'])
    pred_summary['Pred_overview'] = np.where(pred_summary['False_negative'] == 1, 'False negative', pred_summary['Pred_overview'])

    
    model_runs[f'Results_{key}'] = pred_summary
    model_runs[f'AUC_{key}'] = auc_score
    model_runs[f'FEATURE_IMP_{key}'] = feature_imp
    
    f_list = feature_imp.index.to_list()[:10]
    f_list.append('Gully')
    plot_data = all_data[f_list].melt(id_vars = "Gully", var_name = 'feature')
    sns.set(font_scale = 1.4)
    sns.set_style("white")

    f, ax = plt.subplots(figsize=(25, 10))
    sns.boxplot(x = 'feature', y = 'value', hue = 'Gully', data = plot_data,
                ax = ax)
    ax.set_yscale('log')
    
    
    if export_models == True:
        if model_type == 'RF':
            with open(os.path.join(models_dir, f'RF_{key}_{type_}_{version}'),'wb') as f:
                pickle.dump(clf,f)
                
            classified_points = pred_summary.merge(points, how = 'left', on = 'POINT_ID', validate = 'm:1')
            classified_points_shp = gpd.GeoDataFrame(classified_points, geometry = classified_points.geometry,
                                                 crs = points.crs)
            classified_points_shp.to_file(os.path.join(r_dir, f'POINTS_RF_{key}_{type_}_{version}.shp'))
        elif model_type == "CATBOOST":
            clf.save_model(os.path.join(models_dir, f'CATBOOST_{key}_{type_}_{version}'))

if export_models == True:
    with open(os.path.join(models_dir, f'Results_{model_type}_{type_}_{version}.pkl'),'wb') as f:
        pickle.dump(model_runs,f)


