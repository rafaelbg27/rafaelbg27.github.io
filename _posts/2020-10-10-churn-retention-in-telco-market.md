---
layout: post
title: "Churn Retention in Teleco Market"
author: "Rafael Bernardes Gonçalves"
categories: ""
tags: [machine learning, telecom]
image: telecom-market.jpg
---

# Context

– Most telecom companies suffer from voluntary churn of customers
– Telecommunications companies spend millions of dollars to acquire new customers each year
– If these customers leave, the company not only loses their future benefit, but also the resources spent to acquire it
– Marketing and customer relations represent a large category of cost for this type of company. Therefore, it is critical to perform at a high level to secure a competitive advantage
– The Advanced Analytics team is well positioned to develop fact based retention models, as they have the technical skills needed to extract, combine and analyze large volumes of customers data across multi divisional companies

# Initial Data Analysis

First, let's import the libraries and the data. Notice that the column names are not in Python's default snake case. Let's rename them just to make it easier to work with the data. Also, we need to change the floating point from "comma" (brazilian default) to "dot".

I'm going to use the Data Science Project pipeline presented on "Hands–On Machine Learning with Scikit–Learn and TensorFlow" (2e), quite simple and useful. Check appendix for further info on this.

```python
# Reading data

missing_values = [" ", "na", "--"]
db_raw = pd.read_csv("customers.csv", sep=";", na_values = missing_values)

# Switching floating point from comma to dot

db_raw["MonthlyCharges"] = db_raw["MonthlyCharges"].apply(lambda x: str(x).replace(',','.'))
db_raw["TotalCharges"] = db_raw["TotalCharges"].apply(lambda x: str(x).replace(',','.'))

db_raw["MonthlyCharges"] = db_raw["MonthlyCharges"].astype(float)
db_raw["TotalCharges"] = db_raw["TotalCharges"].astype(float)

# Renaming the columns to snake_case

db_raw.rename(columns={"customerID": "customer_id", 
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
"Churn": "churn"}, inplace=True)

# Visualize the changes

pd.set_option('display.max_columns', None)
db_raw.replace('No internet service', 'No', inplace=True)
db_raw.replace('No phone service', 'No', inplace=True)
db_raw.head(5)
```

Now, we need to clean our data, a small inspection of the sheet describing all the variables already tell us that only 3 attributes are numerical and not boolean: "tenure", "monthly_charges" and "total_charges".

Due to that, we need to turn our categorical data into numerical, using an encoder (or manually, if you want to follow a certain order of labelling).

Afterwards, we have a dataset with numbers only, as the method "describe" shows us.

A closer look at the description of the target variable "churn", gives us a precious information about the balance of our dataset. As expected, most telecom clients DON'T voluntary churn (approximately 75% on this data). This has HUGE impact, specially on the models accuracy, because if we simply tell the company that no clients will ever churn, we will have 75% accuracy, which, in several applications of Machine Learning model, is a great achievement. This is going to be further discussed on this notebook.

```python
# Copying the data is important when you are exploring the features, so you don't get many errors when runing a cell

X = db_raw

# Dealing with the categorical data

cat = ['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'internet_service', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'contract', 'paperless_billing', 'payment_method', 'churn']

for label in cat:
    X[label] = LabelEncoder().fit_transform(X[label])

# To find duplicates, we can't forget that the "customer_id" column is suposed to be unique, so removing it before looking for duplicates is a great idea

key = X.columns[1:]
X = X.drop_duplicates(subset=key)
X = X.dropna()

# Summarizing the dataframe to check how many duplicates we had

X.head()
```