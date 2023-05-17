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
	* Identify takeaways from zillow data.	 
	* Make recommendations for improvements to the dataset.
    * Merge findings back onto the zillow database to predict worth of the home.

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
* There is a feature that is not in this dataset that likely contributes greater to a house's worth. It may have to be engineered from existing features that are correlated and non-linear.
* In this first iteration, we will proceed to modeling with the features we have confirmed to be relevant from the original dataset.
* Given more time I would ask for the data to be reengineered to potentially find and replace nulls with relevant information.
* If that isn't possible I would try to engineer some features to increase accuracy.
    * Such as, combining pool, garage, air conditioning into one feature to see if that adds any value.

# Recommendations
* Fill in the dataset by collecting more values to replace all the nulls
* Engineer Features
* It almost feels like real-life with the cheaper the house, the more important bedrooms, bathrooms, and sqft are. While the homes climb in price, these features become less important. Something else makes the worth and the sale...