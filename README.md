# Zillow
# Project Description
 
Improvement of current model used to predict property value for single family residential properties.
 
# Project Goal
 
* Find the key drivers of property value for single family properties.
* Construct a ML Regression Model that predicts assessed worth of Single Family Properties using attributes of the properties from the 2017 dataset.
* Assessed Worth is 'taxvaluedollarcnt' on the original data.
 
# Initial Thoughts
 
My initial hypothesis is that property value is affected by the number of bathrooms.
 
# The Plan
 
* Acquire data from MySQL Server (using Codeup credentials in env.py file).
 
* Prepare data
   * Look at the data frame's info and note:
		* nulls
		* corresponding value counts
		* object types
		* numerical types
		* names of columns
 
* Explore data in search of drivers of worth
   * Answer the following initial questions:
       1. What is more important for worth: bathrooms or bedrooms?
       2. Does sqft affect worth?
       3. Does having A/C affect worth?
           * Unable to answer this question as during exploration the AC column had over 38,000 nulls
       4. Does county affect worth?
       
* Develop a Model to predict worth:
    * Use drivers identified in the explore phase to build predictive models of different types.
    * Evaluate models on the train and validate datasets.
    * Select the best model based on the best fit.
    * Evaluate the best model on the test data.

* Draw conclusions
	* Identify drivers from zillow data. 
	* Make recommendations for improvements.

# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|Bed| The number of bedrooms|
|Bath| The number of bathrooms|
|Sqft| The square feet of the property|
|Assessed Worth| **target** This is the assessed worth of the home|
|County| This is the location of the property, (LA, Orange, Venutra)|
|Date| This is the transaction date|

# Steps to Reproduce
1) Clone this repo.
2) Acquire the data from MySQL servers using your own Codeup credentials stored in an env.py file.
3) Put the data in the file containing the cloned repo.
4) Run notebook.
 
# Takeaways and Conclusions
* There is a feature(s) not in this dataset that contributes greater to a house's worth based on location.
* In this first iteration, we will proceed to modeling with the features we have confirmed to be relevant.
* Given more time I would ask for the data to be reengineered to potentially find and replace nulls with relevant information.
* If that isn't possible I would try to engineer some features to increase accuracy.
    * Such as, combining pool, garage, air conditioning into one feature to see if that adds any value.
    * Location features

# Recommendations
* For the data engineers: Engineer Location Features to improve predicatability of the model
* For the data scientsists: Check for multicollinearity among the predictor variables and remove any highly correlated variables, replacing with new features
* For the business: Improvements in features will increase performance and lead to better results
