Propensity Score Methods for Benchmarking
================

## Setting Up the Data

The example data comes from the state of New York’s
[**SPARCS**](https://www.health.ny.gov/statistics/sparcs/access/)
dataset, which contains medical billing data from all hospitals in the
state. Before running the model the data needs to be imported, and some
of the features need to be recoded.

For the purposes of this example we utilize Suffolk county as the region
of interest (including all hospitals in the county as our pool of peer
hospitals). In addition we limit analysis to the top 30 most common DRG
codes billed by these hospitals.

``` python
import pandas as pd
from provider_scorecard.benchmark import Benchmark

# load data
# set county of interest to Suffolk
df = pd.read_csv("C:/Users/gioc4/Documents/blog/data/sparcs2.zip")
df = df[df['Hospital County'] == 'Suffolk']

# some recoding
df['Length of Stay'] = [120 if x == '120 +' else x for x in df['Length of Stay']]
df['Length of Stay'] = df['Length of Stay'].astype(int)
df['APR DRG Code'] = df['APR DRG Code'].astype(str)

# get only the top 30 drg codes
drg = df.groupby('APR DRG Code')['APR DRG Code'].count().sort_values(ascending=False).head(30)
drg = drg.astype(str)

# final dataframe
df = df[df['APR DRG Code'].isin(drg.index)]
```

Before getting started, we can examine the number of claims billed for
each hospital in Suffolk county.

``` python
df.groupby('Facility Name')['Facility Name'].count().sort_values(ascending=False)
```

    Facility Name
    Stony Brook University Hospital                                   17831
    Good Samaritan Hospital Medical Center                            13667
    South Shore University Hospital                                   12709
    Huntington Hospital                                               10601
    St. Charles Hospital                                               7790
    Peconic Bay Medical Center                                         5468
    St Catherine of Siena Hospital                                     5314
    Long Island Community Hospital                                     4574
    John T Mather Memorial Hospital of Port Jefferson New York Inc     4178
    Stony Brook Southampton Hospital                                   3057
    Stony Brook Eastern Long Island Hospital                            847
    Name: Facility Name, dtype: int64

## Setting Model Parameters

In order to fit a benchmarking model, we need to identify three things.
**(1)** the features that will be used to create the propensity score.
These are features that we are not interested in evaluating, but we are
interested in using to create a balanced comparison unit. **2** the
features that will be benchmarked against. These are the features that
we are interested in comparing to a weighted benchmark unit. **3** the
focal hospital(s) that should be used.

Below we use patient characteristics (age, gender, race), along with the
length of stay, admission type, DRG code, and severity of illness to
create the propensity score. The features we will be benchmarking are
the total charges, total costs, and the type of payment accepted by the
hospital. In this example we use the 3 largest hospitals in Suffolk
county.

``` python
# input features
xfeat = [
    "Age Group",
    "Gender",
    "Race",
    "Length of Stay",
    "Type of Admission",
    "APR DRG Code",
    "APR Severity of Illness Description",
    "APR Risk of Mortality",
]

# evaluation features
efeat = ["Total Charges", "Total Costs", "Payment Typology 1"]

# focal hospitals
hosp = [
    'Stony Brook University Hospital',
    'Good Samaritan Hospital Medical Center',
    'South Shore University Hospital',
]
```

## Fitting the Model

Fitting a benchmarking model is simple. The only things that are
required are a pandas dataframe containing the listed features, a
boolean indicator identifying the focal provider and peer providers, and
two lists of strings identifying the predictor and evaluation features.
Fitting and obtaining results are accomplished using `.fit()` and
`.evaluate()`.

Below, we create a for loop to iterate through the three focal
hospitals, and then extract the results and store the models in separate
lists.

``` python
res_list = []
mod_list = []

# loop through hospitals, store dataframes in list
for h in hosp:
    tr = df["Facility Name"] == h

    bch = Benchmark(
        data=df, focal_indicator=tr, predictor_features=xfeat, evaluation_features=efeat
    )
    bch.fit()
    bch.evaluate()

    # store results
    mod_list.append(bch)
    res_list.append(
        pd.DataFrame(
            {"Hospital": h, "Outcome": bch.Xeval.columns, "Value": bch.outcomes}
        )
    )
```

## Evaluating Results

We can verify that balance is achieved across our features of interest
by using the `calc_balance()` function. We can see that even using
default parameters with no tuning we are able to achieve quite goodd
balance on our first hospital of interest.

``` python
mod_list[0].calc_balance()
```

    Length of Stay:(4.54, 4.45)
    Age Group_0 to 17:(0.21, 0.21)
    Age Group_18 to 29:(0.09, 0.09)
    Age Group_30 to 49:(0.21, 0.21)
    Age Group_50 to 69:(0.19, 0.19)
    Age Group_70 or Older:(0.3, 0.3)
    Gender_F:(0.58, 0.58)
    Gender_M:(0.42, 0.42)
    Race_Black/African American:(0.06, 0.06)
    Race_Multi-racial:(0.01, 0.01)
    Race_Other Race:(0.16, 0.17)
    Race_White:(0.76, 0.76)
    Type of Admission_Elective:(0.11, 0.11)
    Type of Admission_Emergency:(0.57, 0.57)
    Type of Admission_Newborn:(0.18, 0.18)
    Type of Admission_Not Available:(0.0, 0.0)
    Type of Admission_Trauma:(0.0, 0.0)
    Type of Admission_Urgent:(0.14, 0.13)
    APR DRG Code_137:(0.09, 0.09)
    APR DRG Code_139:(0.02, 0.02)
    APR DRG Code_140:(0.01, 0.01)
    APR DRG Code_174:(0.02, 0.02)
    APR DRG Code_175:(0.02, 0.02)
    APR DRG Code_192:(0.01, 0.01)

    APR DRG Code_194:(0.04, 0.04)
    APR DRG Code_201:(0.03, 0.03)
    APR DRG Code_244:(0.01, 0.01)
    APR DRG Code_249:(0.01, 0.01)
    APR DRG Code_254:(0.02, 0.02)
    APR DRG Code_263:(0.01, 0.01)
    APR DRG Code_308:(0.01, 0.01)
    APR DRG Code_324:(0.01, 0.01)
    APR DRG Code_326:(0.02, 0.02)
    APR DRG Code_347:(0.01, 0.01)
    APR DRG Code_383:(0.02, 0.02)
    APR DRG Code_403:(0.01, 0.01)
    APR DRG Code_420:(0.02, 0.01)
    APR DRG Code_45:(0.03, 0.03)
    APR DRG Code_463:(0.03, 0.03)
    APR DRG Code_469:(0.02, 0.02)
    APR DRG Code_53:(0.02, 0.02)
    APR DRG Code_540:(0.07, 0.07)
    APR DRG Code_560:(0.13, 0.13)
    APR DRG Code_640:(0.19, 0.19)
    APR DRG Code_720:(0.11, 0.11)
    APR DRG Code_770:(0.0, 0.0)
    APR DRG Code_772:(0.0, 0.0)

    APR DRG Code_775:(0.01, 0.01)
    APR Severity of Illness Description_Extreme:(0.11, 0.11)
    APR Severity of Illness Description_Major:(0.26, 0.26)
    APR Severity of Illness Description_Minor:(0.3, 0.3)
    APR Severity of Illness Description_Moderate:(0.33, 0.33)
    APR Risk of Mortality_Extreme:(0.12, 0.12)
    APR Risk of Mortality_Major:(0.19, 0.19)
    APR Risk of Mortality_Minor:(0.53, 0.53)
    APR Risk of Mortality_Moderate:(0.16, 0.16)

We can then also extract our results as well. The values below describe
the predicted difference between our focal hospital and the weighted
peer comparison hospital.
``` python
res_list[0]
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
      <th>Hospital</th>
      <th>Outcome</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Stony Brook University Hospital</td>
      <td>Total Charges</td>
      <td>5920.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Stony Brook University Hospital</td>
      <td>Total Costs</td>
      <td>5336.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Stony Brook University Hospital</td>
      <td>Payment Typology 1_Blue Cross/Blue Shield</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Stony Brook University Hospital</td>
      <td>Payment Typology 1_Department of Corrections</td>
      <td>-0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Stony Brook University Hospital</td>
      <td>Payment Typology 1_Federal/State/Local/VA</td>
      <td>-0.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Stony Brook University Hospital</td>
      <td>Payment Typology 1_Managed Care, Unspecified</td>
      <td>-0.07</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Stony Brook University Hospital</td>
      <td>Payment Typology 1_Medicaid</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Stony Brook University Hospital</td>
      <td>Payment Typology 1_Medicare</td>
      <td>-0.03</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Stony Brook University Hospital</td>
      <td>Payment Typology 1_Miscellaneous/Other</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Stony Brook University Hospital</td>
      <td>Payment Typology 1_Private Health Insurance</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Stony Brook University Hospital</td>
      <td>Payment Typology 1_Self-Pay</td>
      <td>0.05</td>
    </tr>
  </tbody>
</table>
</div>
