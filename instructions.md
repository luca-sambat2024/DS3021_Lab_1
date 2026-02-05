# Instructons

Goal: Build two data prep pipelines using different datasets to get practice with data preparation and question building. In doing so create a new github repo for your work. Think of this as a stand alone project that requires the creation of a workspace and repository. In the repo it is likely best practices to create three files. One for the actual assignment details (this file), a second (python file to) answer questions 1 - 3 and third (python file) for question four.

Step one: Review these two datasets and brainstorm problems that could be addressed with the dataset. Identify a question for each dataset.

[College Completion Data Dictionary](https://www.kaggle.com/datasets/thedevastator/boost-student-success-with-college-completion-da/data)

- [Dataset](https://github.com/UVADS/DS-3021/blob/main/data/cc_institution_details.csv)


[Job_Placement](https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv)

- [Data Dictionary (kinda) for Job Placement](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement/discussion/280612) - You’ll need to infer from the column names but also the comments on the site.

Step two: Work through the steps outlined in the examples to include the following elements:

- Write a generic question that this dataset could address.

- What is a independent Business Metric for your problem? Think about the case study examples we have discussed in class.

- Data preparation:
    - correct variable type/class as needed
    - collapse factor levels as needed
    - one-hot encoding factor variables
    - normalize the continuous variables
    - drop unneeded variables
    - create target variable if needed
    - Calculate the prevalence of the target variable
    - Create the necessary data partitions (Train,Tune,Test)

Step three: What do your instincts tell you about the data. Can it address your problem, what areas/items are you worried about?

Step four: Create functions for your two pipelines that produces the train and test datasets. The end result should be a series of functions that can be called to produce the train and test datasets for each of your two problems that includes all the data prep steps you took. This is essentially creating a DAG for your data prep steps. Imagine you will need to do this for multiple problems in the future so creating functions that can be reused is important. You don’t need to create one full pipeline function that does everything but rather a series of smaller functions that can be called in sequence to produce the final datasets. Use your judgement on how to break up the functions.