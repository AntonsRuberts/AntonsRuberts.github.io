---
title: CausalML - Analysing AB Test Results
date: 2020-05-10
tags: [testing, machine learning, causality]
header: 
    image: "/assets/images/causalml_test/header.png"
excerpt: "Establish treatment effect of a test in the presence of confounders using CausalML."
mathjax: "true"
---

#Experiments in Marketing
Experiments are a staple of causal reasoning and are widely used in many disciplines. In digital marketing, A/B test refers to the experiment design that splits the user base into two groups - control and treatment. Treatment group gets exposed to a new feature/design/offer/etc. while the control group's experience stays the same. The behaviours of two groups are monitored and compared with regards to a specific metric (e.g. conversion rate). If the treatment group has a statistically better performance, then the experiment is considered as a success and the feature gets rolled out to the entire user base. Sounds almost trivial, right? Well, unfortunately then the real-life happens and our experiment designs go to hell. 

Let's imagine a scenario: we want to know whether serving a website with localised translations is better than our current version of one-size-fits-all approach. We decide to run an A/B test and track the conversions of two groups. You can download the data [here](https://github.com/AntonsRuberts/datascience_marketing/tree/master/data) and get full code in my [github repo](https://github.com/AntonsRuberts/datascience_marketing/blob/master/CausalML_Analysing_AB_Test.ipynb).

## Data Preprocessing
This dataset is split into two tables, which I'll need to join. Also, most of the variables need to be transformed to serve as inputs into the ML model of choice (here it's LightGBM). To get the data ready, we'll follow the following pre-processing steps:

1. Get seasonal variables from date
2. Join two dataset
3. Transform categorical into numerical

```python
main = pd.read_csv('./data/test_table.csv')
users = pd.read_csv('./data/user_table.csv')

#PREPROCESSING
#Adding seasonal variables
main['date'] = pd.to_datetime(main.date, format = '%Y-%m-%d')
main['month'] = main['date'].apply(lambda x: x.month)
main['day_month'] = main['date'].apply(lambda x: x.day)

#Joining user data
main = main.merge(users, how='inner')

#Drop Spain country
main = main.loc[main['country'] != 'Spain', :]

#Transforming to numerical
main['ads_channel'] = main['ads_channel'].fillna('direct')
categorical = ['source', 'device', 'browser_language', 'ads_channel', 'browser', 'sex', 'country']
for c in categorical:
    main[c] = LabelEncoder().fit_transform(main[c])
    main[c] = main[c].astype('category')
```
Here's how the resulting data should look like:
<img src="{{ site.url }}{{ site.baseirl }}/assets/images/data_head.png", alt="Data Head after Preprocessing">

```python
#dropping date column
main = main.drop('date', axis=1)

sns.set_style("whitegrid")
sns.barplot(x = ['Control', 'Treatment'], y = main.groupby('test')['conversion'].mean().values)
plt.title('Conversion Rate')
plt.show()
print(f'Difference between Control and Treatment {np.round(main.groupby("test")["conversion"].mean()[1] - main.groupby("test")["conversion"].mean()[0], 5)}')
```

$$z = y+z$$