# Online Retail Data Set
The following is an Exploratory Data Analysis (EDA) on the open data base provided by UCI Machine Learning Repository. Details on the data set can be found at: http://archive.ics.uci.edu/ml/datasets/Online+Retail. The objective of this notebook was to observe patterns and trends for classification and clustering, which eventually led to regression. Details on classification, clustering, and regression will be provided within the **[Section 0: Summary of Notebook](#section0)**

# Table of Contents

## [Section 0: Summary of Notebook](#section0)
## [Section 1: Libraries, Functions, and Predefined Models](#section1)
$\dots$**[Section 1.1: Libaries](#section1.1)**

$\dots$**[Section 1.2: Functions](#section1.2)**

$\dots\dots$**[Section 1.2.1: Functions for Section 3](#section1.2.1)**

$\dots\dots$**[Section 1.2.2: Functions for Section 4](#section1.2.2)**

$\dots\dots$**[Section 1.2.3: Functions for Section 5](#section1.2.3)**

$\dots\dots$**[Section 1.2.4: Functions for Section 6](#section1.2.4)**

$\dots$**[Section 1.3: Processed Data and Predefined Models](#section1.3)**

$\dots\dots$**[Section 1.3.1: Processed Data and Models from Section 4](#section1.3.1)**

$\dots\dots$**[Section 1.3.2: Processed Data and Models from Section 5](#section1.3.2)**

$\dots\dots$**[Section 1.3.3: Processed Data and Models from Section 6](#section1.3.3)**
## [Section 2: Preprocessing the 2010 E-Commerce Datset](#section2)
$\dots$**[Section 2.1: Missing Descriptions](#section2.1)**

$\dots$**[Section 2.2: Missing CustomerID](#section2.2)**

$\dots$**[Section 2.3: Duplicate Data](#section2.3)**
## [Section 3: Exploring the Variables](#section3)
$\dots$**[Section 3.1: InvoiceNo](#section3.1)**

$\dots\dots$**[Section 3.1.1: InvoiceNo Outside of Canceled Orders](#section3.1.1)**

$\dots\dots$**[Section 3.1.2: Canceled Orders](#section3.1.2)**

$\dots$**[Section 3.2: Unique Stock Code](#section3.2)**

$\dots$**[Section 3.3: Quantity, Unit Price, and GrossIncome](#section3.3)**
## [Section 4: Classification of Customers](#section4)
$\dots$**[Section 4.1: Base Classifiers](#section4.1)**

$\dots\dots$**[Section 4.1.1: Base Decision Regions](#section4.1.1)**

$\dots\dots$**[Section 4.1.2: Base Confusion Matrices](#section4.1.2)**

$\dots\dots$**[Section 4.1.3: Base ROC Curves and AUC Scores](#section4.1.3)**

$\dots\dots$**[Section 4.1.4: Base Accuracy](#section4.1.4)**

$\dots\dots$**[Section 4.1.5: Base Classifier Conclusion](#section4.1.5)**

$\dots$**[Section 4.2: Ensemble Methods](#section4.2)**

$\dots\dots$**[Section 4.2.1: Soft Majority Voting](#section4.2.1)**

$\dots\dots\dots$**[Section 4.2.1.1: Soft Majority Decision Regions](#section4.2.1.1)**

$\dots\dots\dots$**[Section 4.2.1.2: Soft Majority Confusion Matrix](#section4.2.1.2)**

$\dots\dots\dots$**[Section 4.2.1.3: Soft Majority ROC Curves and AUC Scores](#section4.2.1.3)**

$\dots\dots\dots$**[Section 4.2.1.4: Soft Majority Accuracy](#section4.2.1.4)**

$\dots\dots\dots$**[Section 4.2.1.5: Soft Majority Conclusion](#4.2.1.5)**

$\dots\dots$**[Section 4.2.2: Random Forest](#section4.2.2)**

$\dots\dots\dots$**[Section 4.2.1.1: Random Forest Decision Regions](#section4.2.2.1)**

$\dots\dots\dots$**[Section 4.2.2.2:  Random Forest Confusion Matrices](#section4.2.2.2)**

$\dots\dots\dots$**[Section 4.2.2.3: Random Forest ROC Curves and AUC Scores](#section4.2.2.3)**

$\dots\dots\dots$**[Section 4.2.2.4: Random Forest Accuracy](#section4.2.2.4)**

$\dots\dots\dots$**[Section 4.2.2.5: Random Forest Conclusion](#section4.2.2.5)**

$\dots$**[Section 4.3: Hyperparamterization With GridSearch](#section4.3)**

$\dots\dots$**[Section 4.3.1: The Grid Search](#section4.3.1)**

$\dots\dots$**[Section 4.3.2: Hyperparamterizated Classifiers Decision Regions](#section4.3.2)**

$\dots\dots$**[Section 4.3.3: Hyperparamterizated Classifiers Confusion Matrices](#section4.3.3)**

$\dots\dots$**[Section 4.3.4: Hyperparamterizated Classifiers ROC Curves and AUC Scores](#section4.3.4)**

$\dots\dots$**[Section 4.3.5: Hyperparamterizated Classifiers Accuracy](#section4.3.5)**

$\dots\dots$**[Section 4.3.6: Hyperparamterizated Classifiers Conclusion](#section4.3.6)**
## [Section 5: Clustering to Determine Topics of Description](#section5)
$\dots$**[Section 5.1: Description Preprocessing](#section5.1)**

$\dots$**[Section 5.2: Latent Dirichlet (LDA) Allocation of Description](#section5.2)**

$\dots$**[Section 5.3: Hyperparamterization of LDA by GridSearch](#section5.3)**

$\dots$**[Section 5.4: Best LDA Results](#section5.4)**

$\dots$**[Section 5.5: Visualizing LDA](#section5.5)**

$\dots$**[Section 5.6: LDA Conclusion](#section5.6)**
## [Section 6: Multiple Linear Regression of Daily Income From Sales](#section6)
$\dots$**[Section 6.1: Aggregating the Sales by Date](#section6.1)**

$\dots$**[Section 6.2: Scatter Plots of Income Over Various Sales](#section6.2)**

$\dots$**[Section 6.3: Multiple Linear Regression](#section6.3)**

$\dots\dots$**[Section 6.3.1: Predictors and Outcome](#section6.3.1)**

$\dots\dots$**[Section 6.3.2: Ordinary Multiple Linear Regression Model and Step Wise Selected Model](#section6.3.2)**

$\dots\dots$**[Section 6.3.3: Comparing Original vs Step Wise](#section6.3.3)**

$\dots$**[Section 6.4: ElasticNet](#section6.4)**

$\dots\dots$**[Section 6.4.1: Hyperparameterizaing ElasticNet](#section6.4.1)**

$\dots\dots$**[Section 6.4.2: Comparison to Previous Regressions](#section6.4.2)**

$\dots$**[Section 6.5: Regression Conclusion](#section6.5)**
## [Section 7: Notebook Conclusion](#section7)

## Section 0: Summary of Notebook<a name="section0"></a>
An EDA where we went through the following steps:
* **[Section 1](#section1)**
    - Contains all our custom functions used throughout the notebook.
    - Loads in all hyperparamterized models from local disk.
    - And loads in processed data from disk.
    
* **[Section 2](#section2)**
    - Processed missing data entries:
        - Leading to the removal of rows with missing Descriptions
        - And the introduction of 2 classes for customers to distinguish customers with a CustomerID and those without.
    - Constructed new variables:
        - GrossIncome=Quantity\*UnitPrice
        - Refining the datetime of InvoiceDate to only a date variable
        - Refining the datetime of InvoiceDate to a month variable
    - Removed outliers from: Quantity, UnitPrice, and GrossIncome.
* **[Section 3](#section3)**
    - Explored the columns of: InvoiceNo, StockCode, Quantity, UnitPrice, and GrossIncome
    - Observed variation of Quantity, UnitPrice, and GrossIncome between Return and Guest Customers.
* **[Section 4](#section4)**
    - Distinguished types of Customer by Machine Learning Classification Algorithms
    - Techniques used include: Principal Component Analysis (PCA), Logistic Regression (LR), K-Nearest Neighbors(KNN), Decision Trees(Trees), Soft Majority Voting (MV), Random Forrest (RF).
    - Hyperparamterized our Machine Learning Algorithms by Grid Search.
* **[Section 5](#section5)**
    - Created artificial topics of products sold based on word association of the descriptions.
    - Technique used was Latent Dirichlet Allocation (LDA).
        - Required some additional processing of description.
* **[Section 6](#section6)**
    - Processed data to sum up total sales done with CustomerType and item Topic.
    - Performed Multiple Linear Regression, Step-Wise Selection of Multiple Linear Regression, and ElasticNet to determine Daily Income based off the sales with the previously mentioned line.
    
* **[Conclusion](#section7)**
    - Here we reflect on what we observed and the potential use of the machines we made.
    - We also go into detail of potential projects that we see remain in the dataset