import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preprocess_college_data(url):
    # Read in the data set
    college=pd.read_csv(url)
    
    # Convert some columns into categorical type
    college_category_cols = ["level", "control"]
    college[college_category_cols] = college[college_category_cols].astype("category")

    # Correct columns to boolean
    college["hbcu"] = college["hbcu"].apply(
        lambda x: True if ((x == "X") or (x is True)) else False
    )
    college["flagship"] = college["flagship"].apply(
    lambda x: True if ((x == "X") or (x is True)) else False
    )

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

    # Drop VSA-related columns
    vsa_cols = [name for name in college.columns if "vsa" in name]
    college = college.drop(columns=vsa_cols)

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

    # Drop rows with missing graduation rates
    college = college.dropna(
        subset=[
            "grad_100_value",
            "grad_100_percentile",
            "grad_150_value",
            "grad_150_percentile",
        ]
    )

    # Normalize numerical columns
    college_number_cols = list(college.select_dtypes("number"))
    college[college_number_cols] = MinMaxScaler().fit_transform(
        college[college_number_cols]
    )

    # Create dummy variables for categorical columns
    college_category_list = list(college.select_dtypes("category"))
    college_1h = pd.get_dummies(college, columns=college_category_list)

    # Create binary graduation rate column
    college_1h["grad_100_f"] = pd.cut(
        college_1h["grad_100_value"],
        bins=[-1, 0.5, 1],
        labels=[0, 1],
    )

    # Calculate prevalence
    college_prevalence = (
        college_1h["grad_100_f"].value_counts()[1]
        / len(college_1h["grad_100_f"])
    )

    # Drop additional unnecessary columns
    college_dt = college_1h.drop(
        ["index", "chronname", "city", "state", "counted_pct"],
        axis=1,
    )

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

    return train, tune, test, college_prevalence

def preprocess_placement_data(url):
    # Load in the data set
    placement=pd.read_csv(url)

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

    # Correct columns to boolean
    placement["workex"] = placement["workex"].apply(
        lambda x: True if ((x == "Yes") or (x is True)) else False
    )
    placement["status"] = placement["status"].apply(
        lambda x: True if ((x == "Placed") or (x is True)) else False
    )

    # Normalize numerical columns
    placement_number_cols = list(placement.select_dtypes("number"))
    placement[placement_number_cols] = MinMaxScaler().fit_transform(
        placement[placement_number_cols]
    )

    # Create dummy variables for categorical columns
    placement_category_list = list(placement.select_dtypes("category"))
    placement_1h = pd.get_dummies(placement, columns=placement_category_list)

    # Create binary salary column
    placement_1h["salary_f"] = pd.cut(
        placement_1h["salary"],
        bins=[-1, 0.1, 1],
        labels=[0, 1],
    )

    # Calculate prevalence
    placement_prevalence = (
        placement_1h["salary_f"].value_counts()[1]
        / len(placement_1h["salary_f"])
    )

    # Drop unnecessary columns and rows with missing salary
    placement_dt = placement_1h.drop(["sl_no", "salary"], axis=1)
    placement_dt = placement_dt.dropna(subset=["salary_f"])

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
    return train, test, tune, placement_prevalence