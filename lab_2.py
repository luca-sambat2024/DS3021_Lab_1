# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier


# %%
def preprocess_college_data(
    url="cc_institution_details.csv",
    category="grad_100_value",
):
    """Preprocess the college dataset."""

    # Read dataset
    college = pd.read_csv(url)

    # Convert selected columns to categorical
    college_category_cols = ["level", "control"]
    college[college_category_cols] = college[
        college_category_cols
    ].astype("category")

    # Convert indicator columns to boolean
    college["hbcu"] = college["hbcu"].apply(
        lambda x: True if (x == "X" or x is True) else False
    )
    college["flagship"] = college["flagship"].apply(
        lambda x: True if (x == "X" or x is True) else False
    )

    # Fill null values with zero
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
    vsa_cols = [col for col in college.columns if "vsa" in col]
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

    # Drop rows with missing graduation rate values
    college = college.dropna(
        subset=[
            "grad_100_value",
            "grad_100_percentile",
            "grad_150_value",
            "grad_150_percentile",
        ]
    )

    # Normalize numeric columns
    college_number_cols = list(college.select_dtypes("number"))
    college[college_number_cols] = MinMaxScaler().fit_transform(
        college[college_number_cols]
    )

    # One-hot encode categorical columns
    college_category_list = list(college.select_dtypes("category"))
    college_1h = pd.get_dummies(college, columns=college_category_list)

    # Create binary graduation rate column
    college_1h[f"{category}_f"] = pd.cut(
        college_1h[category],
        bins=np.arange(0, 1.01, 0.2),
        labels=[1, 2, 3, 4, 5],
        include_lowest=True,
    )

    # Calculate prevalence
    college_prevalence = (
        college_1h[f"{category}_f"].value_counts()[1]
        / len(college_1h[f"{category}_f"])
    )

    # Drop additional columns
    college_dt = college_1h.drop(
        ["index", "chronname", "city", "state", "counted_pct"],
        axis=1,
    )

    # Train / test / tune splits
    train, test = train_test_split(
        college_dt,
        train_size=2425,
        stratify=college_dt[f"{category}_f"],
    )

    tune, test = train_test_split(
        test,
        train_size=0.5,
        stratify=test[f"{category}_f"],
    )

    return train, tune, test, college_prevalence, college_dt


def maxmin(x):
    """Min-max normalize a pandas Series."""
    return (x - min(x)) / (max(x) - min(x))


train, tune, test, prevalence, df = preprocess_college_data(
    url="cc_institution_details.csv",
    category="grad_100_value",
)

# %%
# Questions 1 and 2: Creating the KNN model
y = df["grad_100_value_f"]
X = df.loc[:, ["aid_value", "cohort_size"]]
u = X.apply(maxmin)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(u, y)

y_hat = model.predict(u)
y_hat_prob = np.max(model.predict_proba(u), axis=1)
accuracy_raw = model.score(u, y)

print(f"Accuracy with k=3: {accuracy_raw}")

# %%
# Confusion matrix and results dataframe
confusion = pd.crosstab(y, y_hat)
print(confusion)

q3 = pd.DataFrame( # Question 3
    {
        "y": np.array(y),
        "y_hat": np.array(y_hat),
        "prob": np.array(y_hat_prob),
    }
)
print(q3)

# %% [markdown]
# Question 4: If the parameter k is changed, the threshold function will become more fine. Since there will be (presumeably) 
# more neighbors when k increases, the probability values will get more precise. The confusion matrix should not look the 
# same, because the model will be different, and therefore the values may have different noise.

# %%
# Compare k=3 vs k=5
model = KNeighborsClassifier(n_neighbors=5)
model.fit(u, y)

y_hat = model.predict(u)
confusion = pd.crosstab(y, y_hat)

print(confusion)

# %% [markdown]
# Question 5: The confusion matrix shows that the model was a little bit less accurate (for k=5, 2 greater than before). One
# possible concern I see here is that the model got less accurate, meaning that the model should stay at k=3. This means that,
# with a higher k-value, the model will be less accurate in being able to predict the graduation rate.

# %%
k_grid = [(2 * k + 1) for k in range(50)]


def k_test(column): # The two functions were combined into one since they used the same parameter (question 6)
    """Evaluate KNN across multiple k values.""" # The first part of the function creates the train/test split.
    df_local = preprocess_college_data(category=column)[4] 

    y = df_local[f"{column}_f"]
    X = df_local.loc[:, ["aid_value", "cohort_size"]]
    u = X.apply(maxmin)

    u_train, u_test, y_train, y_test = train_test_split(
        u, y, test_size=0.3
    )

    accuracy = []

    comp_model = KNeighborsClassifier(n_neighbors=3)
    comp_model.fit(u_test, y_test)
    print(f"Normal model with k=3: {comp_model.score(u, y)}")

    for k in k_grid: # The second part of the function inside the loop evaluates the accuracy at different k-values.
        test_model = KNeighborsClassifier(n_neighbors=k)
        test_model.fit(u_train, y_train)
        accuracy.append(test_model.score(u_test, y_test))
        plt.plot(accuracy) # Creates a plot showing the k-values and their accuracies.

    ind = np.argmax(accuracy)
    k_optimal = k_grid[ind]

    opt_model = KNeighborsClassifier(n_neighbors=k_optimal)
    opt_model.fit(u_train, y_train)

    print(
        f"Optimized model with best k: "
        f"{opt_model.score(u_test, y_test)}"
    )

    return k_optimal


opt_k = k_test("grad_100_value")
print(opt_k)

# %% [markdown]
# Question 7: Interestingly enough, the models seems to have nearly the same accuracy. Sometimes when run, k=3 yields 
# a better result, and other times, the optimized k-value is better. This could be because they are being trained on 
# the same dataset, so only randomness will determine the better k-value (by a small margin).

# %%
train1, tune1, test1, prevalence1, df1 = preprocess_college_data( 
    url="cc_institution_details.csv", # This does the exact same as before, however we are using a different column now.
    category="grad_150_value", # Question 8
)

opt_k_150 = k_test("grad_150_value")