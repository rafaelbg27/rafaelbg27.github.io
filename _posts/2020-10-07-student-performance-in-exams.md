---
layout: post
title: "Student performances in exams: a discussion on the meritocracy trap"
author: "Rafael Bernardes Gonçalves"
categories: ""
tags: [data science, machine learning, students, meritocracy]
image: meritocracy.jpg
---

One of the biggest discussions nowadays is about meritocracy and how your own effort leads you to sucess. On the other hand, this raises a deeper and harder discussion on where we start on the journey of life. It's really hard to think that someone born in a favela will have the same opportunities as the son of a great salesman that doesn't have to worry with getting a job and helping his family paying the bills.

# Dataset

This data set consists of test results and background information about thousands of students, including:

* Gender;
* Race/ethnicity;
* Parental level of education;
* Lunch: if the student chose free/reduced luch or the regular to take during the test;
* Test preparation;

It was acquired from [kaggle](https://www.kaggle.com/spscientist/students-performance-in-exams)

# Data exploring and insights

```python
# Reading data

db_raw = pd.read_csv("data/students-performance.csv")

# Changing columns name to snake_case

db_raw.rename(columns={"race/ethnicity": "ethnicity",
    "parental level of education": "parental_education",
    "test preparation course": "test_preparation",
    "math score": "math_score",
    "reading score": "reading_score",
    "writing score": "writing_score"}, inplace=True)

# Removing outliers

db_raw = db_raw.loc[(db_raw["math_score"] >= 20)
    & (db_raw["reading_score"] >= 20)
    & (db_raw["writing_score"] >= 20)]

display(db_raw.head())
```

Now that we've imported the data, we can round the grades to have less bars on the bar plots and make the visualization easier.

```python
# Copying the dataframe so we still have the original data

db = db_raw.copy()

# Rounding the scores to the ceil value (multiple of 10)

for column in ["math_score", "reading_score", "writing_score"]:

    category = pd.cut(db[column],bins=[20, 30, 40, 50, 60, 70, 80, 90, 100],
    labels=[30, 40, 50, 60, 70, 80, 90, 100])

    db.drop(column, axis=1, inplace=True)
    db.insert(len(db.columns), column, category)

# Visulize if worked

display(db.head())
```

Now, we will make a bar chart for each variable, as a very basid data exploring, but we can take several conclusions from each one of them.

```python
# Creating the figure
fig = plt.figure(figsize=(14,10))
plt.title('Score by Gender', fontsize=17)

# math
ax1 = plt.subplot(311)
ax1 = sns.countplot(data=db, x='gender', hue='math_score', 
    palette='husl', order = db['gender'].value_counts(ascending=False).index)
plt.legend([])
ax1.set_ylabel('Math Score', fontsize=12)
ax1.set_xlabel('', fontsize=12)
plt.setp(ax1.get_xticklabels(), visible=False)

# reading
ax2 = plt.subplot(312, sharex=ax1)
ax2 = sns.countplot(data=db, x='gender', hue='reading_score', 
    palette='husl', order = db['gender'].value_counts(ascending=False).index)
plt.legend([])
ax2.set_ylabel('Reading Score', fontsize=12)
ax2.set_xlabel('', fontsize=12)
plt.setp(ax2.get_xticklabels(), visible=False)

# writing
ax3 = plt.subplot(313, sharex=ax1)
ax3 = sns.countplot(data=db, x='gender', hue='writing_score', 
    palette='husl', order = db['gender'].value_counts(ascending=False).index)
plt.legend([])
ax3.set_ylabel('Writing Score', fontsize=12)
ax3.set_xlabel('Gender', fontsize=14)
```

<img src="https://github.com/rafaelbg27/rafaelbg27.github.io/blob/gh-pages/assets/img/students_performance/gender.png" alt="drawing" width="150"/>

All charts will follow the same pattern, this is why I've decided to post just one snippet of code, so all you need to do in ordem to replicate it is changing the name. Just to clear things out: for each bar plot, the further to the right, the higher the students grade.

So, concerning the gender we have some small variations, showing that boys do a little better in math, but far worse in reading and writing. Take a while understading this graph, because it's simpler than the followings. The important thing to observe is the size of the middle bars in comparison with the 90-100 score ones.

In conclusion, nothing vary surprising or conclusive, this may be because of the small size of the dataset - around 1000 students.

<img src="https://github.com/rafaelbg27/rafaelbg27.github.io/blob/gh-pages/assets/img/students_performance/test_preparation.png" alt="drawing" width="150"/>



<img src="https://github.com/rafaelbg27/rafaelbg27.github.io/blob/gh-pages/assets/img/students_performance/lunch.png" alt="drawing" width="150"/>



<img src="https://github.com/rafaelbg27/rafaelbg27.github.io/blob/gh-pages/assets/img/students_performance/ethnicity.png" alt="drawing" width="150"/>



<img src="https://github.com/rafaelbg27/rafaelbg27.github.io/blob/gh-pages/assets/img/students_performance/parental_education.png" alt="drawing" width="150"/>

# Meritocracy Trap