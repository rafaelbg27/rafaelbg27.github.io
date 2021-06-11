---
layout: post
title: "Student performances in exams: plotting with seaborn and discussing meritocracy"
author: "Rafael Bernardes Gonçalves"
categories: "insight"
tags: [data science, machine learning, students, meritocracy]
image: /students-performance/thumbnail.png
---

Hi! Today we are going to explore plotting bar charts and practice how to gain insights of a dataset with the most beautiful plotting library in python: seaborn!

## 0) Problem description and looking at the big picture

One of the biggest discussions nowadays is about meritocracy and how your own effort leads you to success. On the other hand, this raises a deeper and harder discussion on where we start on the journey of life. It's really hard to think that someone born in a favela will have the same opportunities as the son of a successful salesman that doesn't have to worry with getting a job and helping his family paying the bills.

## 1) Imports and general definitions

As usual, we start by importing the libraries and changing the default path folder to the root

```python
# Standand libraries for DS
import os

import matplotlib.pyplot as plt
import pandas as pd
import requests

# Plotting
import seaborn as sns

# Data prep
from sklearn.preprocessing import LabelEncoder
```

```python
# Changing working directory
os.chdir("../")
```

```python
# Folders path
paths = {"data": "0-data\\", "pipeline": "2-pipeline\\", "output": "3-output\\"}

# Random state
seed = 42
```

## 2) Data description

Bear in mind that this is a small and fictional dataset, our goal is only to do some data exploration and feature engineering. My original idea was to use a classification algorithm and train it to predict the student characteristics based on the grades, to show have a wider discussion on machine learning "hidden biases", but as I said, the data is fictional and, hence, too "clean". So, let's learn how to explore our data with plots with _seaborn_!

This dataset consists of test results and background information about thousands of students, including:

- Gender;
- Race/ethnicity;
- Parental level of education;
- Lunch: if the student chose free/reduced lunch or the regular to take during the test;
- Test preparation;

It was acquired from [kaggle](https://www.kaggle.com/spscientist/students-performance-in-exams) and you can download the full code at my [github repository](https://github.com/rafaelbg27/kaggle-student-performance-in-exams)

```python
# Reading data
db_raw = pd.read_csv(paths["data"] + "students-performance.csv")
db_raw.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>race/ethnicity</th>
      <th>parental level of education</th>
      <th>lunch</th>
      <th>test preparation course</th>
      <th>math score</th>
      <th>reading score</th>
      <th>writing score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>group B</td>
      <td>bachelor's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>72</td>
      <td>72</td>
      <td>74</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>69</td>
      <td>90</td>
      <td>88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>group B</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>90</td>
      <td>95</td>
      <td>93</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>group A</td>
      <td>associate's degree</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>47</td>
      <td>57</td>
      <td>44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>none</td>
      <td>76</td>
      <td>78</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Changing columns name to snake_case
db_raw.rename(
    columns={
        "race/ethnicity": "ethnicity",
        "parental level of education": "parental_education",
        "test preparation course": "test_preparation",
        "math score": "math_score",
        "reading score": "reading_score",
        "writing score": "writing_score",
    },
    inplace=True,
)

db_raw = db_raw.loc[
    (db_raw["math_score"] >= 20)
    & (db_raw["reading_score"] >= 20)
    & (db_raw["writing_score"] >= 20)
]
db_raw = db_raw.dropna()
db_raw.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 994 entries, 0 to 999
    Data columns (total 8 columns):
     #   Column              Non-Null Count  Dtype
    ---  ------              --------------  -----
     0   gender              994 non-null    object
     1   ethnicity           994 non-null    object
     2   parental_education  994 non-null    object
     3   lunch               994 non-null    object
     4   test_preparation    994 non-null    object
     5   math_score          994 non-null    int64
     6   reading_score       994 non-null    int64
     7   writing_score       994 non-null    int64
    dtypes: int64(3), object(5)
    memory usage: 69.9+ KB

## 3) Data exploring and insights

Now that we’ve imported the data, we can round the grades to have less bars on the bar plots and make the visualization easier.

```python
db = db_raw.copy()

for column in ["math_score", "reading_score", "writing_score"]:

    category = pd.cut(
        db[column],
        bins=[20, 30, 40, 50, 60, 70, 80, 90, 100],
        labels=[30, 40, 50, 60, 70, 80, 90, 100],
    )

    db.drop(column, axis=1, inplace=True)
    db.insert(len(db.columns), column, category)

db.head(10)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>ethnicity</th>
      <th>parental_education</th>
      <th>lunch</th>
      <th>test_preparation</th>
      <th>math_score</th>
      <th>reading_score</th>
      <th>writing_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>group B</td>
      <td>bachelor's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>80</td>
      <td>80</td>
      <td>80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>70</td>
      <td>90</td>
      <td>90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>group B</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>90</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>group A</td>
      <td>associate's degree</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>50</td>
      <td>60</td>
      <td>50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>none</td>
      <td>80</td>
      <td>80</td>
      <td>80</td>
    </tr>
    <tr>
      <th>5</th>
      <td>female</td>
      <td>group B</td>
      <td>associate's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>80</td>
      <td>90</td>
      <td>80</td>
    </tr>
    <tr>
      <th>6</th>
      <td>female</td>
      <td>group B</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>90</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>7</th>
      <td>male</td>
      <td>group B</td>
      <td>some college</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>40</td>
      <td>50</td>
      <td>40</td>
    </tr>
    <tr>
      <th>8</th>
      <td>male</td>
      <td>group D</td>
      <td>high school</td>
      <td>free/reduced</td>
      <td>completed</td>
      <td>70</td>
      <td>70</td>
      <td>70</td>
    </tr>
    <tr>
      <th>9</th>
      <td>female</td>
      <td>group B</td>
      <td>high school</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>40</td>
      <td>60</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>

We will make a bar chart for each variable, as a very basic data exploring, but we can take several conclusions from each one of them.

As all plots follow the same pattern of code, let's create a code to make things easier to read and understand.

```python
def plot_barchart(feature, xlabel):
    fig = plt.figure(figsize=(14, 10))
    plt.title(f"Score by {xlabel}", fontsize=17)

    # math
    ax1 = plt.subplot(311)
    ax1 = sns.countplot(
        data=db,
        x=feature,
        hue="math_score",
        palette="husl",
        order=db[feature].value_counts(ascending=False).index,
    )
    plt.legend([])
    ax1.set_ylabel("Math Score", fontsize=12)
    ax1.set_xlabel("", fontsize=12)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # reading
    ax2 = plt.subplot(312, sharex=ax1)
    ax2 = sns.countplot(
        data=db,
        x=feature,
        hue="reading_score",
        palette="husl",
        order=db[feature].value_counts(ascending=False).index,
    )
    plt.legend([])
    ax2.set_ylabel("Reading Score", fontsize=12)
    ax2.set_xlabel("", fontsize=12)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # writing
    ax3 = plt.subplot(313, sharex=ax1)
    ax3 = sns.countplot(
        data=db,
        x=feature,
        hue="writing_score",
        palette="husl",
        order=db[feature].value_counts(ascending=False).index,
    )
    plt.legend([])
    ax3.set_ylabel("Writing Score", fontsize=12)
    ax3.set_xlabel(xlabel, fontsize=14)
    plt.show()
```

### 3.1) Gender

Just to clear things out: for each bar plot, the further to the right, the higher the students grade.

So, concerning the gender we have some small variations, showing that boys do a little better in math, but far worse in reading and writing. Take a while understanding this graph, because it’s simpler than the followings. The important thing to observe is the size of the middle bars in comparison with the 90-100 score ones.

In conclusion, nothing very surprising or conclusive, this may be because of the small size of the dataset - around 1000 students.

```python
feature = "gender"
xlabel = "Gender"
plot_barchart(feature, xlabel)
```

![png](/assets/img/students-performance/output_14_0.png)

### 3.2) Ethnicity

I tried looking for what definition of race/ethnicity is being used on the dataset, but couldn’t find it. So it was very tempting to simply drop this column, but instead I plotted the same bar chart just to see how your race can interfere in your results, but it’s a little too random and confusing.

```python
feature = "ethnicity"
xlabel = "Ethnicity"
plot_barchart(feature, xlabel)
```

![png](/assets/img/students-performance/output_16_0.png)

### 3.3) Parental education

This the most important feature for insights and discussion. The parental level of education is divided into:

- Some high school
- High school
- Some college
- Associate’s degree
- Bachelor’s degree
- Master’s degree

This is the plot that shows the disparity of grades distribution very well, note how the master’s degree have scores way above average, along with the bachelor’s degree group.

```python
feature = "parental_education"
xlabel = "Parental Education"
plot_barchart(feature, xlabel)
```

![png](/assets/img/students-performance/output_18_0.png)

### 3.4) Lunch

The lunch is probably directly correlated with the financial situation of the student. Here, we can see a huge performance difference between the two groups, but another possible interpretation is just that eating well during the test may improve your score.

```python
feature = "lunch"
xlabel = "Lunch"
plot_barchart(feature, xlabel)
```

![png](/assets/img/students-performance/output_20_0.png)

### 3.5) Test preparation

The test preparation feature is the more intuitive one, so our only direct conclusion is that students that have completed the test preparation have higher grades. Though we can clearly recognize that this effect is stronger in writing and reading than in math.

```python
feature = "test_preparation"
xlabel = "Test Preparation"
plot_barchart(feature, xlabel)
```

![png](/assets/img/students-performance/output_22_0.png)

## 4) Feature relevance and correlation

Just out of curiosity, let's to group all the different student profiles, take out the mean of the scores and sort the resulting dataframe.

```python
db = db_raw.copy()

profiles = []

for ethnicity in db["ethnicity"].unique():
    for parental_education in db["parental_education"].unique():
        for lunch in db["lunch"].unique():
            for test_preparation in db["test_preparation"].unique():
                profiles.append(
                    {
                        "ethnicity": ethnicity,
                        "parental_education": parental_education,
                        "lunch": lunch,
                        "test_preparation": test_preparation,
                    }
                )

for i in range(len(profiles)):
    profile = db.loc[
        (db["ethnicity"] == profiles[i]["ethnicity"])
        & (db["parental_education"] == profiles[i]["parental_education"])
        & (db["lunch"] == profiles[i]["lunch"])
        & (db["test_preparation"] == profiles[i]["test_preparation"])
    ]

    profiles[i]["math_score"] = profile["math_score"].mean()
    profiles[i]["reading_score"] = profile["reading_score"].mean()
    profiles[i]["writing_score"] = profile["writing_score"].mean()
    profiles[i]["mean_score"] = (
        profile["math_score"].mean()
        + profile["reading_score"].mean()
        + profile["writing_score"].mean()
    ) / 3
```

```python
df = pd.DataFrame(data=profiles)
df = df.sort_values(by=["mean_score"])
df = df.reset_index().drop("index", axis=1)
df = df.dropna()
df.sort_values("mean_score", ascending=False).head(20)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ethnicity</th>
      <th>parental_education</th>
      <th>lunch</th>
      <th>test_preparation</th>
      <th>math_score</th>
      <th>reading_score</th>
      <th>writing_score</th>
      <th>mean_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>115</th>
      <td>group E</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>completed</td>
      <td>91.000000</td>
      <td>99.000000</td>
      <td>97.500000</td>
      <td>95.833333</td>
    </tr>
    <tr>
      <th>114</th>
      <td>group B</td>
      <td>bachelor's degree</td>
      <td>free/reduced</td>
      <td>completed</td>
      <td>87.000000</td>
      <td>90.000000</td>
      <td>88.000000</td>
      <td>88.333333</td>
    </tr>
    <tr>
      <th>113</th>
      <td>group B</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>83.500000</td>
      <td>92.500000</td>
      <td>88.500000</td>
      <td>88.166667</td>
    </tr>
    <tr>
      <th>112</th>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>77.285714</td>
      <td>83.571429</td>
      <td>85.071429</td>
      <td>81.976190</td>
    </tr>
    <tr>
      <th>111</th>
      <td>group C</td>
      <td>bachelor's degree</td>
      <td>standard</td>
      <td>completed</td>
      <td>77.615385</td>
      <td>82.384615</td>
      <td>82.769231</td>
      <td>80.923077</td>
    </tr>
    <tr>
      <th>110</th>
      <td>group E</td>
      <td>bachelor's degree</td>
      <td>free/reduced</td>
      <td>completed</td>
      <td>80.333333</td>
      <td>80.666667</td>
      <td>81.333333</td>
      <td>80.777778</td>
    </tr>
    <tr>
      <th>109</th>
      <td>group E</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>77.666667</td>
      <td>81.666667</td>
      <td>81.000000</td>
      <td>80.111111</td>
    </tr>
    <tr>
      <th>108</th>
      <td>group E</td>
      <td>bachelor's degree</td>
      <td>standard</td>
      <td>completed</td>
      <td>82.857143</td>
      <td>77.571429</td>
      <td>79.857143</td>
      <td>80.095238</td>
    </tr>
    <tr>
      <th>107</th>
      <td>group C</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>completed</td>
      <td>73.750000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>78.583333</td>
    </tr>
    <tr>
      <th>106</th>
      <td>group B</td>
      <td>associate's degree</td>
      <td>standard</td>
      <td>completed</td>
      <td>75.000000</td>
      <td>79.888889</td>
      <td>80.444444</td>
      <td>78.444444</td>
    </tr>
    <tr>
      <th>105</th>
      <td>group E</td>
      <td>associate's degree</td>
      <td>standard</td>
      <td>completed</td>
      <td>78.916667</td>
      <td>78.250000</td>
      <td>78.000000</td>
      <td>78.388889</td>
    </tr>
    <tr>
      <th>104</th>
      <td>group B</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>75.750000</td>
      <td>80.250000</td>
      <td>78.250000</td>
      <td>78.083333</td>
    </tr>
    <tr>
      <th>103</th>
      <td>group E</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>79.625000</td>
      <td>78.500000</td>
      <td>76.000000</td>
      <td>78.041667</td>
    </tr>
    <tr>
      <th>102</th>
      <td>group D</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>75.250000</td>
      <td>78.125000</td>
      <td>80.000000</td>
      <td>77.791667</td>
    </tr>
    <tr>
      <th>101</th>
      <td>group D</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>75.562500</td>
      <td>77.062500</td>
      <td>80.125000</td>
      <td>77.583333</td>
    </tr>
    <tr>
      <th>100</th>
      <td>group D</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>completed</td>
      <td>73.500000</td>
      <td>76.500000</td>
      <td>82.500000</td>
      <td>77.500000</td>
    </tr>
    <tr>
      <th>99</th>
      <td>group E</td>
      <td>high school</td>
      <td>standard</td>
      <td>completed</td>
      <td>75.250000</td>
      <td>78.250000</td>
      <td>78.000000</td>
      <td>77.166667</td>
    </tr>
    <tr>
      <th>98</th>
      <td>group D</td>
      <td>master's degree</td>
      <td>free/reduced</td>
      <td>completed</td>
      <td>69.250000</td>
      <td>78.250000</td>
      <td>83.750000</td>
      <td>77.083333</td>
    </tr>
    <tr>
      <th>97</th>
      <td>group A</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>77.800000</td>
      <td>77.000000</td>
      <td>76.200000</td>
      <td>77.000000</td>
    </tr>
    <tr>
      <th>96</th>
      <td>group A</td>
      <td>bachelor's degree</td>
      <td>standard</td>
      <td>completed</td>
      <td>80.666667</td>
      <td>73.333333</td>
      <td>76.666667</td>
      <td>76.888889</td>
    </tr>
  </tbody>
</table>
</div>

As we can see, looking at the top of the clusters we have only students with highly educated parents.

Another way of seeing all the features together and how they relate to each other is with the correlation matrix. Luckily, seaborn also provide a correlation matrix function. In the matrix, the closer the absolute value of the correlation is to 1, than more the features relate to each other.

You can find out more about these [here](https://en.wikipedia.org/wiki/Correlation_and_dependence).

```python
# Feature correlations
db = db_raw.copy()
cat = ["gender", "ethnicity", "parental_education", "lunch", "test_preparation"]

for label in cat:
    db[label] = LabelEncoder().fit_transform(db[label])

corrMatrix = db.corr()
fig = plt.figure(figsize=(14, 10))
sns.heatmap(corrMatrix, annot=True)
plt.show()
```

![png](/assets/img/students-performance/output_27_0.png)

## 5) Final thoughts

In conclusion, it’s clear that the features related to parental education and money are deeply related to the student result on the exams. This is why meritocracy is so debatable when there is no isonomy between all people. A great video exemplifying this (in portuguese) can be watched in this link.

If you liked this “insight project” please let me know in the comments bellow!

Thanks for your time :D
