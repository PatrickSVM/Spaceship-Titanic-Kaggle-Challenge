import pandas as pd
import numpy as np


def annotate_bar_perc(plot, n_rows, text_size=14, text_pos=(0,8), prec=2):
    """
    Function that annotates a stacked matplotlib barplot with percentage labels.
    
    """
    
    # Annotate the Bar-plots with category-percentage
    conts = plot.containers # Get containers, for each class of Transported one (True/False)
    
    for i in range(len(conts[0])): 
        height = sum([cont[i].get_height() for cont in conts]) # Calculate height of bar
        text = f"{height/n_rows*100:.{prec}f}%" # Create Annotation

        # Add text for every big bar
        plot.annotate(text,
                    (conts[0][i].get_x() + conts[0][i].get_width() / 2, height),  # Position xy
                    ha='center', # Centering
                    va='center',  
                    size=text_size, 
                    xytext=text_pos, # Coordinates of Text, Coord-System is defined by textcoords
                    textcoords='offset points') # Coord-System (So from the annotated position + xytext)
    return plot





def preprocess_PassengerId(data):
    """
    Preprocess PassengerID. Returns three columns:
        1. GroupID   - Unique ID of the group the passenger is in
        2. GroupPos  - Position in the group, that is assigned to passenger
        3. GroupSize - New feature, that assigns each passenger the size of the group he is part of
    """
    
    new_ID = data.PassengerId.str.split("_", expand=True)
    new_ID.columns = ["GroupID", "GroupPos"]
    new_ID.GroupPos = new_ID.GroupPos.str.replace("0","").astype(int)
    
    # Get dictionary of ID to GroupSize (extract max GroupPos from positions in unique ID)
    group_size_dict = new_ID.groupby("GroupID").max().to_dict()["GroupPos"]
    
    # Assign group size to each row from dict
    new_ID["GroupSize"] = new_ID.apply(lambda row: group_size_dict[row["GroupID"]], axis=1)
    
    # Delete preceeding 0s from GroupID
    new_ID.GroupID = new_ID.GroupID.str.replace(pat=r"\b0+(?=\d)", repl="", regex=True).astype(int)
    
    return new_ID




def preprocess_Cabin(data):
    """
    Preprocess Cabin. Returns three columns:
        1. Deck    - Letter referring to the Deck the cabin is located on 
        2. Number  - Cabin Number
        3. Side    - Side of the ship, either P for Port or S for Starboard
    """
    
    new_cols = data.Cabin.str.split("/", expand=True)
    new_cols.columns = ["Deck", "CabinNum", "Side"]
    return new_cols






def preprosess_spaceship_titanic(df, log_transform_exp=False):
    """
    Function that preprocesses the dataframe of the Spaceship Titanic dataset according to the
    process described in the notebook. 
    
    Returns the preprocessed dataset that can be fed to the imputation function.


    Parameters:
    -----------
    
    df (pandas.DataFrame)   : Raw pandas dataframe of the Spaceship Titanic Challenge
                                        
    log_transform_exp (bool): Boolean controlling whether expenses are log-transformed


    Returns:
    --------
    preprocessed (pandas.DataFrame): Preprocessed pandas dataframe
    
    """
    
    # Copy dataframe
    preprocessed = df.copy()
    
    # Convert CryoSleep and VIP to bool
    preprocessed[["CryoSleep", "VIP"]] = preprocessed[["CryoSleep", "VIP"]].astype("bool")
    
    # Add TotalExp column
    preprocessed["TotalExp"] = preprocessed[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(1)
    
    # LogTransform expenses (add one to avoid -Inf) if specified
    if log_transform_exp:
        preprocessed[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalExp']] = np.log(preprocessed[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalExp']]+1)
    
    # Preprocess PassengerId 
    preprocessed = pd.concat([preprocessed, preprocess_PassengerId(preprocessed)], axis=1)
    
    # Preprocess Cabin 
    preprocessed = pd.concat([preprocessed, preprocess_Cabin(preprocessed)], axis=1)
    preprocessed = preprocessed.drop("Cabin", axis=1)
    
    return preprocessed





def impute_group_category(group_categories, mode):
    """
    Function that takes series of categorical feature from a group 
    and finds most probable value to impute. 
    
    The mode (str) is the value in the overall dataset occuring the most, 
    that is used in cases where there are no information from groups. 
    """
    # Check if group size > 1
    if len(group_categories)>1:
        
        # Get unique planets without nan
        homes_count = group_categories.value_counts()
        
        if len(homes_count)==1: # Only one planet in group
            return homes_count.index[0]
        
        elif len(homes_count)==0: # Only NaNs, impute mode Earth
            return mode
        
        else: # More than one planet in group
            
            if (homes_count==max(homes_count)).sum()==1: # If clear decision (so max only occurs once) take it
                return homes_count.idxmax()
            else: # There are two planets with same max count - sample one of them to reduce bias 
                np.random.seed(123) # Set random seed for numpy (important for repoducability)
                return np.random.choice(homes_count[homes_count==max(homes_count)].index) # Sample one of them
    
    else:
        return mode
    
    
    
    
    
    
    
def impute_expenses(data, strategy):
    """
    Function that takes dataframe and imputes all NaN-expenses 
    according to the strategy described in the previous EDA.
    
    Returns the dataframe with imputed expenses.
    
    Parameters:
    -----------
    
    data (pandas.DataFrame): Dataframe with NaN-expenses
    
    strategy (str)         : String specifying strategy to impute expenses.
                                        
                             Possible values are: 
                             - 'mean' for mean imputation
                             - 'median' for median imputation
                             - 'group_mean' for imputation based on groups
                             - 'group_median' for imputation based on groups

    
    Returns:
    --------
    
    expense_df (pandas.DataFrame): Imputed pandas dataframe
                                  
    """
    
    # Copy df
    df = data.copy()
    
    
    if strategy=='median':
            # Each index filled with value in deck median series
            df = df.fillna({"RoomService":df.groupby("Deck")["RoomService"].transform("median"), 
                                   "FoodCourt":df.groupby("Deck")["FoodCourt"].transform("median"), 
                                   "Spa":df.groupby("Deck")["Spa"].transform("median"),
                                   "VRDeck":df.groupby("Deck")["VRDeck"].transform("median"), 
                                   "ShoppingMall":df.groupby("Deck")["ShoppingMall"].transform("median")})
                
        
        
    elif strategy=='mean':
        # Each index filled with value in deck mean series
        df = df.fillna({"RoomService":df.groupby("Deck")["RoomService"].transform("mean"), 
                               "FoodCourt":df.groupby("Deck")["FoodCourt"].transform("mean"), 
                               "Spa":df.groupby("Deck")["Spa"].transform("mean"),
                               "VRDeck":df.groupby("Deck")["VRDeck"].transform("mean"), 
                               "ShoppingMall":df.groupby("Deck")["ShoppingMall"].transform("mean")})



    elif strategy=='group_mean':
        # Each index filled with mean value of the group if group bigger 1 (otherwise deck median)
        df[df.GroupSize>1] = df[df.GroupSize>1].fillna({"RoomService":df.groupby("GroupID")["RoomService"].transform("mean"), 
                                                   "FoodCourt":df.groupby("GroupID")["FoodCourt"].transform("mean"), 
                                                   "Spa":df.groupby("GroupID")["Spa"].transform("mean"),
                                                   "VRDeck":df.groupby("GroupID")["VRDeck"].transform("mean"), 
                                                   "ShoppingMall":df.groupby("GroupID")["ShoppingMall"].transform("mean")})

        df[df.GroupSize==1] = df[df.GroupSize==1].fillna({"RoomService":df.groupby("Deck")["RoomService"].transform("median"), 
                                                   "FoodCourt":df.groupby("Deck")["FoodCourt"].transform("median"), 
                                                   "Spa":df.groupby("Deck")["Spa"].transform("median"),
                                                   "VRDeck":df.groupby("Deck")["VRDeck"].transform("median"), 
                                                   "ShoppingMall":df.groupby("Deck")["ShoppingMall"].transform("median")})



    elif strategy=='group_median':
        # Each index filled with mean value of the group if group bigger 1 (otherwise deck median)
        df[df.GroupSize>1] = df[df.GroupSize>1].fillna({"RoomService":df.groupby("GroupID")["RoomService"].transform("median"), 
                                                   "FoodCourt":df.groupby("GroupID")["FoodCourt"].transform("median"), 
                                                   "Spa":df.groupby("GroupID")["Spa"].transform("median"),
                                                   "VRDeck":df.groupby("GroupID")["VRDeck"].transform("median"), 
                                                   "ShoppingMall":df.groupby("GroupID")["ShoppingMall"].transform("median")})

        df[df.GroupSize==1] = df[df.GroupSize==1].fillna({"RoomService":df.groupby("Deck")["RoomService"].transform("median"), 
                                                   "FoodCourt":df.groupby("Deck")["FoodCourt"].transform("median"), 
                                                   "Spa":df.groupby("Deck")["Spa"].transform("median"),
                                                   "VRDeck":df.groupby("Deck")["VRDeck"].transform("median"), 
                                                   "ShoppingMall":df.groupby("Deck")["ShoppingMall"].transform("median")})

    else:
        raise ValueError("Wrong parameter value given.")

    return df









def impute_spaceship_titanic(preprocessed_df, proba_imp=True, expense_strat='group_mean', age_strat='group_mean', drop_outliers=False):
    """
    Function that imputes values for the Spaceship Titanic dataset based on the previous EDA. 
    
    Returns the dataset with imputed values.


    Parameters:
    -----------
    
    preprocessed_df (pandas.DataFrame): Preprocessed dataframe according to previous 
                                        EDA
                                        
    proba_imp (bool)                  : Boolean controlling whether only safe 
                                        imputations should be done or whether 
                                        probabilistic imputations (based on the 
                                        relatonships found in EDA should be included 
                                        too
                                        
    expense_strat (str)               : String specifying strategy to impute 
                                        different expense categories. 
                                        
                                        Possible values are: 
                                        - 'mean' for mean imputation
                                        - 'median' for median imputation
                                        - 'group_mean' for imputation based on groups
                                        - 'group_median' for imputation based on groups
    
    age_strat (str)                   : String specifying strategy to impute age.
                                        
                                        Possible values are: 
                                        - 'mean' for mean imputation
                                        - 'median' for median imputation
                                        - 'group_mean' for imputation based on groups
                                        
    drop_outliers (bool)              : Boolean specifying whether to drop outliers                    

    
    Returns:
    --------
    
    impute (pandas.DataFrame): Imputed pandas dataframe
    
    """

    # Copy DF
    impute = preprocessed_df.copy()

    
    # Delete outliers/implausible datapoints in training case:
    if drop_outliers:
        
        # Delete passengers under 12 years that are travelling alone
        impute = impute[~((impute.Age<12) & (impute.GroupSize==1))]

        # Throw away deck T
        impute = impute[impute.Deck!="T"]


    
    # Start with imputations that were found during EDA and that are 'safe'.

    # CryoSleep=True -> All expenses are 0
    exp_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    mask1 = impute.CryoSleep & impute.loc[:,exp_cols].isna().any(1)
    impute.loc[mask1, exp_cols] = impute.loc[mask1, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)

    # TotalExp > 0 -> CryoSleep=False
    impute[impute.TotalExp>0] = impute[impute.TotalExp>0].fillna({"CryoSleep":0}) 

    # CryoSleep=False but age <= 12 -> All expenses are 0
    mask2 = ~impute.CryoSleep & impute.loc[:,exp_cols].isna().any(1)
    impute.loc[mask1, exp_cols] = impute.loc[mask1, exp_cols].fillna(0)

    # Deck=G -> VIP=False and HomePlanet=Earth
    impute[impute.Deck=="G"] = impute[impute.Deck=="G"].fillna({"VIP":False, "HomePlanet":"Earth"}) 

    # Deck A, B or C -> HomePlanet=Europa
    impute[impute.Deck.isin(["A", "B", "C"])] = impute[impute.Deck.isin(["A", "B", "C"])].fillna({"HomePlanet":"Europa"}) 

    

    
    # If specified, the probabilistic imputations are also executed in the following code.
    
    if proba_imp:

        # Set missing VIPs to False (in total only 2.3% VIPs so very likely)
        impute = impute.fillna({"VIP":False})

        # CryoSleep=NaN but Age<=12 -> CryoSleep=False (from analysis above)
        impute[impute.CryoSleep.isna() & (impute.Age<=12)] = impute[impute.CryoSleep.isna() & (impute.Age<=12)].fillna({"CryoSleep":False})

        # CryoSleep=NaN but Age>12 -> CryoSleep=True
        impute[impute.CryoSleep.isna() & (impute.Age>12)] = impute[impute.CryoSleep.isna() & (impute.Age>12)].fillna({"CryoSleep":True})
        
        # Impute Destination with Destination from group, otherwise with most likely one: TRAPPIST-1e
        impute["Destination"] = impute["Destination"].fillna(impute.groupby("GroupID")["Destination"].transform(impute_group_category, mode="TRAPPIST-1e"))
        
        # Impute HomePlanet with HomePlanet from group, otherwise with most likely one: Earth
        impute["HomePlanet"] = impute["HomePlanet"].fillna(impute.groupby("GroupID")["HomePlanet"].transform(impute_group_category, mode="Earth"))
        
        # Impute Deck with Deck from group, otherwise with most likely one: F
        impute["Deck"] = impute["Deck"].fillna(impute.groupby("GroupID")["Deck"].transform(impute_group_category, mode="F"))
        
        # Impute Side with Side from group, otherwise with "Missing":
        impute["Side"] = impute["Side"].fillna(impute.groupby("GroupID")["Side"].transform(impute_group_category, mode="Missing"))
        
        # Impute CabinNum with CabinNum from group, otherwise with "Missing":
        impute["CabinNum"] = impute["CabinNum"].fillna(impute.groupby("GroupID")["CabinNum"].transform(impute_group_category, mode="Missing"))
        
        
        # All expenses = 0 besides NaN values -> other expenses also 0 with high likelihood (showed in last part of analysis)
        mask3 = ~impute.CryoSleep & ((impute.iloc[:,6:11].isna().sum(1) + (impute.iloc[:,6:11]==0).sum(1))==5) & (impute.iloc[:,6:11].isna().sum(1)>0)
        impute.loc[mask3, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = impute.loc[mask3, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)

        
        # Expense imputation
        impute = impute_expenses(impute, strategy=expense_strat)
        
        
        
        # Age imputation
        if age_strat=='mean':
            impute = impute.fillna({"Age":impute.Age.mean().round()})
            
        elif age_strat=='group_mean':
            # Impute group mean if possible
            impute[impute.GroupSize>1] = impute[impute.GroupSize>1].fillna({"Age":impute.groupby("GroupID")["Age"].transform("mean").round()})
                                                                        
            # Ohterwise take mean age of all passengers travelling alone
            impute[impute.GroupSize==1] = impute[impute.GroupSize==1].fillna({"Age": impute[impute.GroupSize==1].Age.mean().round()})
        
        
        else:
            raise ValueError("Wrong parameter value given.")
                                                                            
                                                                        
        
        # Columns that will not be used anyways are just filled with string "Missing" (we can still keep those rows):
        # Name/CabinNum/Side
        impute = impute.fillna({"Name":"Missing"})
    
    return impute







def post_imputation_process_spaceship_titanic(imputed_df, was_log_transf):
    """
    Function that is applied after imputation, that recalculates TotalExp 
    and adds boolean-columns for travelling alone and spending nothing. 
    
    Returns the final dataframe.


    Parameters:
    -----------
    
    imputed_df (pandas.DataFrame): Df after preprocessing and imputation
                                        
    was_log_transformed (bool)   : Boolean controlling whether expenses were 
                                   log-transformed (important for TotalExp 
                                   update)


    Returns:
    --------
    
    final (pandas.DataFrame): Final pandas dataframe
    
    """
    
    # Copy df
    final = imputed_df.copy()
    
    # Impute small amount of remaining NaNs
    final = final.fillna({"CryoSleep": False, "Age":final.Age.mean().round()})
    
    # Transform age to int
    final.Age = final.Age.astype(int)
    
    # Update TotalExp
    if was_log_transf: # If they were log-transformed, backtransform, recalculate total, transform
        final[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = np.exp(final[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])-1
        final["TotalExp"] = final[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(1)
        
        # Transform again
        all_exp_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalExp']
        final[all_exp_cols] = np.log(final[all_exp_cols]+1)
        
        # Add prefix to mark as log transformed
        final.columns = [f"Log{col}" if col in all_exp_cols else col for col in final.columns]
    
    else: # If expenses were not log-transformed
        final["TotalExp"] = final[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(1)
        
    # Add marker for 0 expenses
    if was_log_transf:
        final["NoExpenses"] = (final.LogTotalExp == 0)
    else:
        final["NoExpenses"] = (final.TotalExp == 0)
    
    # Add marker for traveling alone
    final["Alone"] = (final.GroupSize == 1)
    
    return final







def preprocess_impute_spaceship_titanic(df, **kwargs):
    
    """
    Function that put togehter the whole preprocessing pipeline.
    
    Returns the final dataframe.


    Parameters:
    -----------
    
    df (pandas.DataFrame): Raw spaceship titanic pandas dataframe
                                        
    kwargs               : All other keyword arguments for other
                           functions:
                           - log_transform_exp
                           - proba_imp
                           - expense_strat
                           - age_strat
    
    Further information regarding kwargs:
    
    proba_imp (bool)                  : Boolean controlling whether only safe 
                                        imputations should be done or whether 
                                        probabilistic imputations (based on the 
                                        relatonships found in EDA should be included 
                                        too
                                        
    expense_strat (str)               : String specifying strategy to impute 
                                        different expense categories. 
                                        
                                        Possible values are: 
                                        - 'mean' for mean imputation
                                        - 'median' for median imputation
                                        - 'group_mean' for imputation based on groups
                                        - 'group_median' for imputation based on 
                                           groups
    
    age_strat (str)                   : String specifying strategy to impute age.
                                        
                                        Possible values are: 
                                        - 'mean' for mean imputation
                                        - 'median' for median imputation
                                        - 'group_mean' for imputation based on groups
                                        

    Returns:
    --------
    
    final_df (pandas.DataFrame): Final pandas dataframe after preprocessing pipeline
    
    """
    
    # Set default values for kwargs
    kwargs.setdefault('log_transform_exp', False)
    kwargs.setdefault('proba_imp', True)
    kwargs.setdefault('expense_strat', 'group_mean')
    kwargs.setdefault('age_strat', 'group_mean')
    kwargs.setdefault('drop_outliers', False)
    
    # Preprocess df
    preprocessed_df = preprosess_spaceship_titanic(df, log_transform_exp=kwargs['log_transform_exp'])
    
    # Impute missing values
    imputed_df = impute_spaceship_titanic(preprocessed_df, 
                                         proba_imp=kwargs['proba_imp'], 
                                         expense_strat=kwargs['expense_strat'],
                                         age_strat=kwargs['age_strat'],
                                         drop_outliers=kwargs['drop_outliers'])
    
    # Finalize preprocessing
    final_df = post_imputation_process_spaceship_titanic(imputed_df, was_log_transf=kwargs['log_transform_exp'])
    
    return final_df