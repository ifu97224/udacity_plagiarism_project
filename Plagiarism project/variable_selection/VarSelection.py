import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from .SBS import SBS
from .timing import timing

class variable_selection():
    """ The variable selection class contains a set of methods to support variable selection in classification modeling """    
    def __init__(self, df, target_var, k_features = 10, random_state = 1):
        
        """ Method for initializing a VarSelection object
        
        Args:
        df (Pandas Dataframe)
        target_var (string)
        k_features (int)
        test_size (float)
        random_state (int)
                
        Attributes:
            df (Dataframe):  Pandas dataframe containing the data to run the tests on
            target_var (String):  The name of the target variable on which to run the tests (should be binary for classification)
            k_features (int):  the number of features to return e.g. the top X correlated variables
            random_state (int):  random number seed
        
        """
        self.df = df
        self.target_var = target_var
        self.k_features = k_features
        self.random_state = random_state


    def prep_cat_vars(self, df):
    
        """ Method for preparing a dataframe that has categorical variables (creates dummies)

        Args:
        df (Dataframe)

        Attributes:
        df:  Pandas dataframe containing the target and feature variables

        Returns:
        Pandas Dataframe with categorical variables one hot encoded
        """
    
        cat_features = df.loc[:, df.dtypes == object]

        if not cat_features.empty:
            cat_features_transform = pd.get_dummies(cat_features)

            # Append back the numeric variables to get an encoded dataframe
            numeric_features = df._get_numeric_data()

            df_encoded = pd.concat([cat_features_transform, numeric_features], axis = 1)
    
            return df_encoded
    
        else:
            return None
        
    @timing
    def squared_corr(self):
        """ Method for calculating the top X variables with the highest squared correlation with the target variable
            
        Args:
        df (Dataframe)
        target_var (String)
        k_features (Integer)
    
        Attributes:
        df:  Pandas dataframe containing the target and feature variables
        target_var:  The target variable
        k_features:  The number of features to return
        
        Returns:
        Pandas Dataframe with the top X variables related to the target by squared correlation
        """
        # Get all numeric data into a dataframe
        corr_df_input = self.df._get_numeric_data()

        # Calculate the squared correlation and remove the target
        squared_corr = pd.DataFrame(corr_df_input.corr()[self.target_var][:]).reset_index()
        squared_corr.columns = ('Variable', 'Squared_Correlation')
        squared_corr.loc[:,'Squared_Correlation'] = squared_corr[['Squared_Correlation']]**2
        squared_corr = squared_corr[squared_corr.Variable != self.target_var]

        # Order and select the top X
        squared_corr.sort_values(by = 'Squared_Correlation', ascending = False, inplace = True)
        squared_corr = squared_corr.iloc[:self.k_features]
    
        return squared_corr
    
    @timing
    def rf_imp_rank(self, RandomForestClassifier):
  
        """ Method for calculating the top X variables with the highest rf importance with the target variable
        
        Args:
        df (Dataframe)
        target_var (String)
        k_features (Integer)
        RandomForestClassifier (Class)
    
        Attributes:
        df:  Pandas dataframe containing the target and feature variables
        target_var:  The target variable
        k_features:  The number of features to return
        RandomForestClassifier:  A Random Forest classifier using sci-kit learn RandomForestClassifier
        
        Returns:
        Pandas Dataframe with the top X variables by RF Importance 
        """
        df = self.df

        cat_features = df.loc[:, df.dtypes == object]

        if not cat_features.empty:
            df = self.prep_cat_vars(df)

        X = df.drop([self.target_var], axis=1)
        y = df[self.target_var]
        feat_labels = pd.DataFrame(X.columns)

        # Run the random forest model
        forest = RandomForestClassifier.fit(X, y)

        # Get the rf importance and append the feature variable labels
        importance = pd.DataFrame(forest.feature_importances_)
        rf_importance = feat_labels.merge(importance,left_index = True, right_index = True)
        rf_importance.columns = ['features','rf_importance']
        rf_importance.sort_values('rf_importance', ascending = False, inplace = True) 
        rf_importance['rf_rank'] = range(1, len(rf_importance) + 1)

        rf_importance = rf_importance[rf_importance.rf_rank <= self.k_features]

        return rf_importance
    
    @timing
    def abs_reg_coeffs(self, linear_model):

        """ Method for calculating the top X variables by the absolute coefficient
        
        Args:
        df (Dataframe)
        target_var (String)
        k_features (Integer)
        linear_model (Class)
    
        Attributes:
        df:  Pandas dataframe containing the target and feature variables
        target_var:  The target variable
        k_features:  The number of features to return
        linear_model:  A Linear Model using sci-kit learn linear_model
        
        Returns:
        Pandas Dataframe with the top X variables by absolute coefficient size
        """
    
        df = self.df
        
        cat_features = df.loc[:, df.dtypes == object]

        if not cat_features.empty:
            df = self.prep_cat_vars(df)

        X = df.drop([self.target_var], axis=1)
        y = df[self.target_var]
        feat_labels = pd.DataFrame(X.columns)

        # fit the model
        lm = linear_model.fit(X, y)

        # get the coefficients and features into a data frame and create the rank
        lm_coeff = pd.DataFrame(lm.coef_).T
        feat_labels = pd.DataFrame(X.columns[1:])
        lm_reg_coeff = feat_labels.merge(lm_coeff,left_index = True, right_index = True)
        lm_reg_coeff.columns = ['features','coeff']
        lm_reg_coeff['coeff_abs'] = lm_reg_coeff['coeff'].abs()
        lm_reg_coeff.sort_values('coeff_abs', ascending = False, inplace = True) 
        lm_reg_coeff['coeff_rank'] = range(1, len(lm_reg_coeff) + 1)

        lm_reg_coeff = lm_reg_coeff[lm_reg_coeff.coeff_rank <= self.k_features]
    
        return lm_reg_coeff
    
    @timing
    def rfe(self, clf, cv_folds, scoring, var_list = []):

        """ Method for running reccursive feature selection using RFECV from sklearn feature_selection

        Args:
        df (Dataframe)
        target_var (String)
        clf (Class)
        cv_folds (Integer)
        scoring (String)
        var_list (List)

        Attributes:
        df:  Pandas dataframe containing the target and feature variables
        target_var:  The target variable
        clf:  An sklearn classifier
        cv_folds:  The number of CV folds to use
        scoring:  The method of scoring e.g. 'roc_auc'
        var_list:  A list of variables on which to run RFECV

        Returns:
        Matplot lib chart showing the CV score by number of variables and list containing the variables selected
        """

        df = self.df

        if len(var_list) != 0:
            var_list.append(self.target_var)
            df = df[var_list]

        cat_features = df.loc[:, df.dtypes == object]

        if not cat_features.empty:
            df = self.prep_cat_vars(df)

        X = df.drop([self.target_var], axis=1)
        y = df[self.target_var]

        clf = clf

        rfecv = RFECV(estimator=clf, 
                      step=1, 
                      cv=StratifiedKFold(cv_folds),
                      scoring=scoring)

        rfecv.fit(X, y)

        print("Optimal number of features : %d" % rfecv.n_features_)

        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (AUC)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()

        selected_vars = X[X.columns[rfecv.support_]].columns

        return selected_vars
    
    @timing
    def feat_agglom(self, n_clusters, standardize = True):
    
        """ Method for running feature agglomeration

        Args:
        df (Dataframe)
        target_var (String)
        n_clusters (Integer)
        Standardize (Boolean)

        Attributes:
        df:  Pandas dataframe containing the target and feature variables
        target_var:  The target variable
        n_clusters:  Number of clusters to return

        Returns:
        Pandas dataframe containing the feature name and the cluster number

        """

        df = self.df
        
        cat_features = df.loc[:, df.dtypes == object]

        if not cat_features.empty:
            df = self.prep_cat_vars(df)

        X = df.drop([self.target_var], axis=1) 
        X_df = X
        X = X_df.values

        if standardize == True:
            scaler = StandardScaler() 
            X = scaler.fit_transform(X)

        agglo = FeatureAgglomeration(n_clusters=n_clusters)
        clusters = agglo.fit_transform(X)

        cluster_numbers = pd.DataFrame(agglo.labels_)
        feat_labels = pd.DataFrame(X_df.columns)
        var_clust = feat_labels.merge(cluster_numbers, left_index = True, right_index = True)
        var_clust.columns = ['Feature_Label','Cluster_Number']
        var_clust.sort_values('Cluster_Number', inplace = True)

        return var_clust
    
    @timing
    def best_subsets(self, clf, subsets, var_list = []):

        """ Method for running best subsets

        Args:
        df (Dataframe)
        target_var (String)
        clf (Class)
        subsets (Integer)
        var_list (List)

        Attributes:
        df:  Pandas dataframe containing the target and feature variables
        target_var:  The target variable
        clf:  An sklearn classifier
        subsets:  The number of subsets to run
        var_list:  A list of variables on which to run best subsets

        Returns:
        Matplot lib chart showing the validation score by number of subsets and list containing the variables for
        the best subset
        """
        df = self.df
        
        if len(var_list) != 0:
            var_list.append(self.target_var)
            df = df[var_list]

        cat_features = df.loc[:, df.dtypes == object]

        if not cat_features.empty:
            df = self.prep_cat_vars(df)

        X_df = df.drop([self.target_var], axis=1) 
        X = X_df.values
        y = df[self.target_var].values

        sbs = SBS(clf, k_features = subsets, random_state = self.random_state)

        sbs.fit(X, y)

        scores = sbs.scores_
        max_score = scores.index(max(scores))
        subset_cols = list(sbs.subsets_[max_score])
        best_features = X_df.columns[0:][subset_cols]

        # Show the plot of the score by all subsets
        k_feat = [len(k) for k in sbs.subsets_]
        plt.plot(k_feat, sbs.scores_, marker = 'o')
        plt.ylabel('Score')
        plt.xlabel('Number of Features')
        plt.grid()
        plt.show()

        return best_features