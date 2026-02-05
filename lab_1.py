# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load data
college = pd.read_csv("cc_institution_details.csv")
placement = pd.read_csv("placement.csv")

# %% [markdown]
## College completion data set:
# - Columns like "hbcu", "flagship", and others have a majority as null values. It may be best practice 
# to replace them with corresponding values (ex. boolean should be False).
# - FTE value: most colleges have a very low value, but a few colleges have extreme high values, ranging 
# from 33 to 127k. Same with the cohort_size column.
# - "Similar" column: it is a boolean, but there are both null values and string values present.
# - "Site" column: just like many others, for machine learning, we will probably not be examining these 
# columns since they are not numerical features.
# - Question: How can we predict the graduation rate with the information given?
#
## Job Placement data set
# - For each column in the data set, there is no clear data type. Following this, the column "workex" has 
# entries as "yes" or "no", but they would be transformed into a True or False boolean.
# - The "status" column could be transformed into a boolean column. 
# - The "salary" column has multiple missing values (null values).
# - Question: how can we predict salary based on the information given?
# %%
# --------------------------------------------------
# College completion dataset
# IBM: graduation rate
# --------------------------------------------------

# %%
# Convert some columns into categorical type
college_category_cols = ["level", "control"]
college[college_category_cols] = college[college_category_cols].astype("category")

# %%
# Correct columns to boolean
college["hbcu"] = college["hbcu"].apply(
    lambda x: True if ((x == "X") or (x is True)) else False
)
college["flagship"] = college["flagship"].apply(
    lambda x: True if ((x == "X") or (x is True)) else False
)

# %%
# Fill null values with zero for specific columns
college_fill_na = [
    "aid_value",
    "aid_percentile",
    "endow_value",
    "endow_percentile",
    "retain_value",
    "retain_percentile",
    "ft_fac_value",
    "ft_fac_percentile",
    "pell_value",
    "pell_percentile",
]
college[college_fill_na] = college[college_fill_na].fillna(0)

# %%
# Drop VSA-related columns
vsa_cols = [name for name in college.columns if "vsa" in name]
college = college.drop(columns=vsa_cols)

# %%
# Drop unnecessary columns
college = college.drop(
    columns=[
        "long_x",
        "lat_y",
        "unitid",
        "similar",
        "nicknames",
        "ft_fac_value",
        "ft_fac_percentile",
        "endow_value",
        "endow_percentile",
        "state_sector_ct",
        "carnegie_ct",
    ]
)

# %%
# Drop rows with missing graduation rates
college = college.dropna(
    subset=[
        "grad_100_value",
        "grad_100_percentile",
        "grad_150_value",
        "grad_150_percentile",
    ]
)

# %%
# Normalize numerical columns
college_number_cols = list(college.select_dtypes("number"))
college[college_number_cols] = MinMaxScaler().fit_transform(
    college[college_number_cols]
)

# %%
# Create dummy variables for categorical columns
college_category_list = list(college.select_dtypes("category"))
college_1h = pd.get_dummies(college, columns=college_category_list)

# %%
# Create binary graduation rate column
college_1h["grad_100_f"] = pd.cut(
    college_1h["grad_100_value"],
    bins=[-1, 0.5, 1],
    labels=[0, 1],
)

# %%
# Calculate prevalence
college_prevalence = (
    college_1h["grad_100_f"].value_counts()[1]
    / len(college_1h["grad_100_f"])
)

# %%
# Drop additional unnecessary columns
college_dt = college_1h.drop(
    ["index", "chronname", "city", "state", "counted_pct"],
    axis=1,
)

# %%
# Train / test / tune splits
train, test = train_test_split(
    college_dt,
    train_size=2425,
    stratify=college_dt["grad_100_f"],
)

tune, test = train_test_split(
    test,
    train_size=0.5,
    stratify=test["grad_100_f"],
)

# %% [markdown]
# The cleaned data set should be able to help answer the question. It contains many parameters to assist answering the 
# question from, like the national awards data, SAT scores, etc. However, one concern I have is that some of the colleges 
# have only reported on a certain percent of their students (counted_pct column), which may lead to inaccurate results. 

# %%
# --------------------------------------------------
# Job placement dataset
# IBM: salary
# --------------------------------------------------

# %%
# Convert columns to categorical type
placement_category_cols = [
    "gender",
    "ssc_b",
    "hsc_b",
    "hsc_s",
    "degree_t",
    "specialisation",
]
placement[placement_category_cols] = placement[placement_category_cols].astype(
    "category"
)

# %%
# Correct columns to boolean
placement["workex"] = placement["workex"].apply(
    lambda x: True if ((x == "Yes") or (x is True)) else False
)
placement["status"] = placement["status"].apply(
    lambda x: True if ((x == "Placed") or (x is True)) else False
)

# %%
# Normalize numerical columns
placement_number_cols = list(placement.select_dtypes("number"))
placement[placement_number_cols] = MinMaxScaler().fit_transform(
    placement[placement_number_cols]
)

# %%
# Create dummy variables for categorical columns
placement_category_list = list(placement.select_dtypes("category"))
placement_1h = pd.get_dummies(placement, columns=placement_category_list)

# %%
# Create binary salary column
placement_1h["salary_f"] = pd.cut(
    placement_1h["salary"],
    bins=[-1, 0.1, 1],
    labels=[0, 1],
)

# %%
# Calculate prevalence
placement_prevalence = (
    placement_1h["salary_f"].value_counts()[1]
    / len(placement_1h["salary_f"])
)

# %%
# Drop unnecessary columns and rows with missing salary
placement_dt = placement_1h.drop(["sl_no", "salary"], axis=1)
placement_dt = placement_dt.dropna(subset=["salary_f"])

# %%
# Train / test / tune splits
train, test = train_test_split(
    placement_dt,
    train_size=111,
    stratify=placement_dt["salary_f"],
)

tune, test = train_test_split(
    placement_dt,
    train_size=0.75,
    stratify=placement_dt["salary_f"],
)

# %% [markdown]
# The data should be able to help us answer the question. There are many categorical variables split into the dummy columns.
# One issue that may be faced with the cleaned/prepared data set is the fact that some rows were deleted since they had null
# values for their salary (no job). Since the question is to predict salary, then those columns would not be useful.