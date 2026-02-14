# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# %%
def preprocess_college_data(url="cc_institution_details.csv", category="grad_100_value"):
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
    college_1h[category+"_f"] = pd.cut(
        college_1h[category],
        bins=np.arange(0,1.01,0.2),
        labels=[1, 2, 3, 4, 5],
        include_lowest=True,
    )

    # Calculate prevalence
    college_prevalence = (
        college_1h[category+"_f"].value_counts()[1]
        / len(college_1h[category+"_f"])
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
        stratify=college_dt[category+"_f"],
    )

    tune, test = train_test_split(
        test,
        train_size=0.5,
        stratify=test[category+"_f"],
    )

    return train, tune, test, college_prevalence, college_dt

def maxmin(x):
    u=(x-min(x))/(max(x)-min(x))
    return u

train, tune, test, prevalence, df=preprocess_college_data(url="cc_institution_details.csv", category="grad_100_value")

# %%
# Questions 1 and 2: creating the KNN model.
y=df["grad_100_value_f"]
X=df.loc[:,["aid_value","cohort_size"]]
u=X.apply(maxmin)

model=KNeighborsClassifier(n_neighbors=3)
model.fit(u,y)
y_hat=model.predict(u)
y_hat_prob=np.max(model.predict_proba(u), axis=1)
accuracy_raw=model.score(u,y)
print("Accuracy with k=3: "+str(accuracy_raw))

# %%
# Create the confusion matrix. Then, create the data frame with the test target values, test predicted values, and the probability.
confusion=pd.crosstab(y,y_hat)
print(confusion)

q3=pd.DataFrame({"y":np.array(y),"y_hat":np.array(y_hat),"prob":np.array(y_hat_prob)}) # Question 3
print(q3)
# %% [markdown]
# Question 4: If the parameter k is changed, the threshold function will become more fine. Since there will be (presumeably) more neighbors
# when k increases, the probability values will get more precise. The confusion matrix should not look the same, because the
# model will be different, and therefore the values may have different noise.
# %%
# Here we are making another model, effectively comparing the confusion matrices from k=3 to k=5.
model=KNeighborsClassifier(n_neighbors=5)
model.fit(u,y)
y_hat=model.predict(u)
confusion=pd.crosstab(y,y_hat)
print(confusion)
# %% [markdown]
# Question 5: The confusion matrix shows that the model was a little bit less accurate (for k=5, 2 greater than before). One
# possible concern I see here is that the model got less accurate, meaning that the model should stay at k=3. This means that,
# with a higher k-value, the model will be less accurate in being able to predict the graduation rate.
# %%
k_grid=[(2*k+1) for k in range(50)]

def k_test(column): # I combined the two functions into one, since they use the same parameter (question 6).
    df=preprocess_college_data(category=column)[4] # The first part is outside a loop, creating the train/test split.
    y=df[column+"_f"]
    X=df.loc[:,["aid_value","cohort_size"]]
    u=X.apply(maxmin)
    u_train, u_test, y_train, y_test=train_test_split(u,y,test_size=0.3)
    accuracy=[]
    comp_model=KNeighborsClassifier(n_neighbors=3)
    comp_model.fit(u_test,y_test)
    print("Normal model with k=3: "+str(comp_model.score(u,y)))
    comp_y_hat=comp_model.predict(u)
    for k in k_grid: # The second part of the loop is going through and evaluating the k-values.
        test_model=KNeighborsClassifier(n_neighbors=k)
        test_model.fit(u_train,y_train)
        accuracy.append(test_model.score(u_test,y_test))
        plt.plot(accuracy) # Create a graph to see which k-values are better.
    ind=np.argmax(accuracy)
    k_optimal=k_grid[ind]
    opt_model=KNeighborsClassifier(n_neighbors=k_optimal)
    opt_model.fit(u_train, y_train)
    opt_y_hat=opt_model.predict(u_test)
    print("Optimized model with best k: "+str(opt_model.score(u_test,y_test)))
    return k_optimal

opt_k=k_test("grad_100_value")
print(opt_k)

# %% [markdown]
# Interestingly enough, the models seems to have nearly the same accuracy. Sometimes when run, k=3 yields a better result,
# and other times, the optimized k-value is better. This could be because they are being trained on the same dataset, so 
# only randomness will determine the better k-value (by a small margin).
# %%
train1, tune1, test1, prevalence1, df1=preprocess_college_data(url="cc_institution_details.csv", category="grad_150_value")
opt_k_150=k_test("grad_150_value") # Here we are effectively doing the same thing, but using a different column.
# %%
