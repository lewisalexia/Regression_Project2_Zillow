
# Imports
import env
import os

# Numbers
import pandas as pd 
import numpy as np
from scipy import stats

# Vizzes
import matplotlib.pyplot as plt
import seaborn as sns

# Splits
from sklearn.model_selection import train_test_split

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------

# Visualizations

def hist_zillow(df):
    """This function display histograms for all columns"""
    plt.figure(figsize=(12,6))
   
    for i, col in enumerate(df.columns):
        plt.tight_layout(pad=3.0)
        plot_number = i + 1
        plt.subplot(2, 4, plot_number)
        plt.hist(df[col])
        plt.title(f"{col}")   

    plt.subplots_adjust(left=0.1,
                bottom=0.1, 
                right=0.9, 
                top=0.9, 
                wspace=0.4, 
                hspace=0.4)
    
    plt.show() 

def visual_explore_univariate(df):
    """This function takes in a DF and explores each variable visually as well as the value_counts to identify outliers. Works for numerical."""
    for i, col in enumerate(df.columns[:-3]):
        plt.subplot(3,1,i+1)
        sns.boxplot(data=df, x=col, palette='Spectral')
        plt.title(f"{col.capitalize()} Boxplot")
        plt.show()


    
def plot_variable_pairs(df):
    """This function takes in a df and returns a pairplot with regression line."""
    sns.pairplot(data=df,\
             kind='reg',corner=True, plot_kws={'line_kws':{'color':'red'}}\
             , palette="Accent")
    plt.show()

def plot_categorical_vars(df):
    """This function takes in a df and a hardcoded target variable to explore.
    ---
    This function assigns the df columns to categorical and numerical lists. 
    ---
    Object types indicate "buckets" which indicates a categorical variable (col_cat).
    ---
    The function will then print for each col in col_cat:
        * Value Counts
        * Proportional size of data
        * Hypotheses (null + alternate)
        * Analysis and summary using CHI^2 test function (chi2_test from stats_conclude.py)
        * A conclusion statement
        * A graph representing findings
    """
    col_cat = [] #this is for my categorical varibles
    col_num = [] #this is for my numerical varibles (not using this at the moment)
    target = 'assessed_worth' # assigning target variable
    
    # assign
    for col in df.columns:
        if col in df.select_dtypes(include=['number']):
            col_num.append(col)
        else:
            col_cat.append(col)
            
    # iterate through categorical
    for col in col_cat:
        print(f"{col.capitalize()} Count")
        print(df[col].value_counts())
        print('---')
        print('Proportion of Dataset')
        print(round(df[col].value_counts(normalize=True)*100),2)
        print('---')
        print(f'Hypothesize')
        print(f"H_0: {col.capitalize().replace('_',' ')} does not affect {target.replace('_',' ')}")
        print(f"H_a: {col.capitalize().replace('_',' ')} affects {target.replace('_',' ')}")
        print('---')
        print('Analyze')
        observed = pd.crosstab(df[col], df[target])
        α = 0.05
        chi2, pval, degf, expected = stats.chi2_contingency(observed)
        print(f'p-value = {pval} < {α}')
        print('----')
        if pval < α:
            print (f"We reject the null hypothesis, County does affect {target.replace('_',' ')}.")
        else:
            print (f"We fail to reject the null hypothesis, County does not affect {target.replace('_',' ')}.")
    
        # visualize
        sns.boxenplot(data=df, x=col, y=target, palette='Accent')
        plt.title(f"boxenplot of {col.capitalize().replace('_',' ')} vs {target.replace('_',' ')}")
        plt.axhline(df[target].mean(), color='blue', linestyle='--')
        plt.show()

def plot_heatmap(df):
    """This functions returns a heatmap of a df"""
    worth_corr = df.corr(method='pearson')
    sns.heatmap(worth_corr, cmap='PRGn', annot=True, mask=np.triu(worth_corr))
    plt.title(f"Assessed Worth Correlation Heatmap")
    plt.show()

def plot_bed_bath(train, bed, bath):
    """This function returns the violinplot graphs for the bed and bath columns
    in relation to the target variable of assessed worth."""
    sns.violinplot(data=train, x=bed, y='assessed_worth', palette='mako', alpha=.5)
    plt.yticks(ticks = np.arange(0, 1_000_000, step=150_000))
    plt.title(f"Number of Beds and Home Worth")
    plt.axhline(train.assessed_worth.mean(), label=(f"Average Worth"), color='blue',linestyle='--')
    plt.legend()
    plt.show()
    print()

    sns.violinplot(data=train, x=bath, y='assessed_worth', palette='mako', alpha=.5)
    plt.yticks(ticks = np.arange(0, 1_000_000, step=150_000))
    plt.title(f"Number of Baths and Home Worth")
    plt.axhline(train.assessed_worth.mean(), label=(f"Average Worth"), color='blue',linestyle='--')
    plt.legend()
    plt.show()
    print()   



def plot_variables(df, bed, bath):
    """This function plots the distributions of two sets of data.
    The first set is the number of beds and baths against house worth.
    The second set is the distribution of sqft with number of beds and bath."""

    #plot our number of beds and baths
    plt.figure(figsize=(12,8))
    sns.barplot(data=df, x=bath, y='assessed_worth',hue=bed, ci=None, dodge=True, palette='colorblind')
    plt.xlabel('Bath')
    plt.ylabel('Worth')
    plt.axhline(df.assessed_worth.mean(), ls=':', color='black', label='Average Worth')
    plt.title(f"Bed and Bath vs House Worth")
    plt.show()

    #plot our sqft and number of beds and bath
    sns.barplot(data=df, x=bath, y='sqft',hue=bed, ci=None, dodge=True, palette='colorblind')
    plt.xlabel('Bath')
    plt.ylabel('Sqft')
    plt.axhline(df.sqft.mean(), ls=':', color='black', label='Average Worth')
    plt.title(f"Distribution of Sqft and Number of Beds and Baths")
    plt.show()

def plot_sqft(df):
    """This function graphs the distribution of sqft and worth with mean lines"""
    #plot our sqft and worth
    plt.figure(figsize=(12,10))
    sns.scatterplot(data=df, x='sqft', y='assessed_worth', alpha=.5,size=5, ci=None, legend=False)
    plt.xlabel('Sqft')
    plt.ylabel('Worth')
    plt.axhline(df.assessed_worth.mean(), color='purple', label='Average Worth', linewidth=3)
    plt.axvline(df.sqft.mean(), color='purple', label='Average Sqft', linewidth=3)
    plt.title(f"Distribution of Sqft and Worth")
    plt.show()


# -------------------------------------------------------------------------

# # KEILA'S BATHROOM FULL AND HALF CODE
# def categorize_bathrooms(bath):
#     if bath.is_integer():
#         return 'full'
#     else:
#         return 'half'
    
# create a new column with the categorized bathrooms
# train['bathroom_type'] = train['bathroom'].apply(categorize_bathrooms)

# -------------------------------------------------------------------------

# ENCODE

def encode_zillow(df):
    """This function encodes a df
    ---
    Format: df_encoded = function()
    """
    # HARDCODED drop
    df = df.drop(columns=['date'])

    # encode / get dummies
    dummy_df = pd.get_dummies(df[['county']], dummy_na=False, drop_first=True)
    print(f"Encoded County column and renamed encoded columns for readability")

    # clean up and return final product
    df = pd.concat([df, dummy_df], axis=1)
    df_encoded = df.rename(columns={'county_Orange':'orange', 'county_Ventura':'ventura'})
    df_encoded = df_encoded.drop(columns=['county'])

    print(f"DataFrame is encoded and ready for modeling. :)")

    return df_encoded


