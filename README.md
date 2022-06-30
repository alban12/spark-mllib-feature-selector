# Spark MLLib Feature Selector
An implementation of the Nested Monte Carlo algorithm to select feature based on Spark MLLib

The `nrpa.py` file provide a `nrpa_feature_selector` function that take as arguments : 
* `level : The level for the nested search 
* `iterations` : The number of iteration to do at the root 
* `train_set_dataframe` : The dataframe that will be test the features on 
* `validation_set_dataframe` : The dataframe that will be validate the features on 
* `feature_space_size` : The feature space size (i.e. the number of features)
* `learning_algorithm` : The algorithm that 
* `policy` : The policy to choose the move (should be uniform on each moves at the initialization)
* `task` : The machine learning task (`classification` or `regression`)
* `metric` (optional): The metric used to quantify the goodness of a feature
