---
layout: post
title: "Student performances in exams: a discussion on the meritocracy trap"
author: "Rafael Bernardes Gonçalves"
categories: "insight"
tags: [data science, machine learning, students, meritocracy]
image: meritocracy.jpg
---

One of the biggest discussions nowadays is about meritocracy and how your own effort leads you to success. On the other hand, this raises a deeper and harder discussion on where we start on the journey of life. It's really hard to think that someone born in a favela will have the same opportunities as the son of a successful salesman that doesn't have to worry with getting a job and helping his family paying the bills.

# Dataset

Bear in mind that this is a small dataset, so we will only do some data exploration and feature engineering. My original idea was to use a classification algorithm and train it to predict the student characteristics based on the grades, to show have a wider discussion on machine learning "hidden biases".

This data set consists of test results and background information about thousands of students, including:

* Gender;
* Race/ethnicity;
* Parental level of education;
* Lunch: if the student chose free/reduced lunch or the regular to take during the test;
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

W
e will make a bar chart for each variable, as a very basic data exploring, but we can take several conclusions from each one of them.

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

All charts will follow the same pattern, this is why I've decided to post just one snippet of code, so all you need to do in order to replicate it is changing the name. Just to clear things out: for each bar plot, the further to the right, the higher the students grade.

So, concerning the **gender** we have some small variations, showing that boys do a little better in math, but far worse in reading and writing. Take a while understanding this graph, because it's simpler than the followings. The important thing to observe is the size of the middle bars in comparison with the 90-100 score ones.

In conclusion, nothing very surprising or conclusive, this may be because of the small size of the dataset - around 1000 students.

![profile]({{ "assets/img/students-performance/gender.PNG" | absolute_url }})

The **test preparation** feature is the more intuitive one, so our only direct conclusion is that students that have completed the test preparation have higher grades.

![profile]({{ "assets/img/students-performance/test_preparation.PNG" | absolute_url }})

The **lunch** is probably directly correlated with the financial situation of the student. Here, we can see a huge performance difference between the two groups, but another possible interpretation is just that eating well during the test may improve your score.

![profile]({{ "assets/img/students-performance/lunch.PNG" | absolute_url }})

I tried looking for what definition of race/ethnicity is being used on the dataset, but couldn't find it. So it was very tempting to simply drop this column, but instead I plotted the same bar chart just to see how your race can interfere in your results, but it's a little too random and confusing.

![profile]({{ "assets/img/students-performance/ethnicity.PNG" | absolute_url }})

This the most important feature for insights and discussion. The **parental level of education** is divided into:

- Some high school
- High school
- Some college
- Associate's degree
- Bachelor's degree
- Master's degree

This is the plot that shows the disparity of grades distribution very well, note how the *master's degree* have scores way above average, along with the *bachelor's degree* group.

![profile]({{ "assets/img/students-performance/parental_education.PNG" | absolute_url }})

# Feature relevance and correlation

Just out of curiosity, I'm going to group all the different student profiles, take out the mean of the scores and sort the dataframe.

```python
# Copying the dataframe
db = db_raw.copy()

profiles = []

for ethnicity in db["ethnicity"].unique():
    for parental_education in db["parental_education"].unique():
        for lunch in db["lunch"].unique():
            for test_preparation in db["test_preparation"].unique():
                profiles.append({"ethnicity": ethnicity,
                    "parental_education": parental_education,
                    "lunch": lunch,
                    "test_preparation": test_preparation})

# For each profile, take the mean scores
for i in range(len(profiles)):
    profile = db.loc[(db['ethnicity'] == profiles[i]['ethnicity'])
        & (db['parental_education'] == profiles[i]['parental_education'])
        & (db['lunch'] == profiles[i]['lunch'])
        & (db['test_preparation'] == profiles[i]['test_preparation'])]

    profiles[i]["math_score"] = profile["math_score"].mean()
    profiles[i]["reading_score"] = profile["reading_score"].mean()
    profiles[i]["writing_score"] = profile["writing_score"].mean()
    profiles[i]["mean_score"] = (profile["math_score"].mean()
        + profile["reading_score"].mean()
        + profile["writing_score"].mean())/3

# Turn the dictionary into dataframe
df = pd.DataFrame(data=profiles)
df = df.sort_values(by=['mean_score'])
df = df.reset_index().drop("index", axis=1)
df = df.dropna()
display(df)
```
![profile]({{ "assets/img/students-performance/list.PNG" | absolute_url }})

As we can see, looking at the top of the clusters we have only students with highly educated parents.

Another way of seeing all the features together and how they relate to each other is with the correlation matrix:

```python
# Feature correlations

db = db_raw.copy()

cat = ['gender', 'ethnicity', 'parental_education', 'lunch', 'test_preparation']

for label in cat:
    db[label] = LabelEncoder().fit_transform(db[label])


corrMatrix = db.corr()

fig = plt.figure(figsize=(14,10))
sns.heatmap(corrMatrix, annot=True)
plt.show()
```
![profile]({{ "assets/img/students-performance/corr_matrix.PNG" | absolute_url }})

In conclusion, it's clear that the features related to parental education and money are deeply related to the student result on the exams. This is why meritocracy is so debatable when there is no isonomy between all people. A great video exemplifying this (in portuguese) can be watched in [this link](https://www.youtube.com/watch?v=YINTTVjBrY4&list=PLU6WL9E2gOnjwbt3oZCqdMdltsZTaKEXL&index=37).

If you liked this "insight project" please let me know!

Thanks for your time, bye!