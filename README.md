# Pima Indians diabetes prediction

by Derek Huang

_last updated on: 01-26-2020_  
_file created on: 01-24-2020_

This repository is a quick analysis of the Kaggle dataset of diabetes occurrence in Pima Indian females, which may be found [here](https://www.kaggle.com/uciml/pima-indians-diabetes-database). By no means is this a comprehensive exploration; for example, the model hypothesis set can be expanded, PCA features experimented with, and some more involved feature engineering performed (both manual and with packages such as `featuretools`).

Contributors: Derek Huang

## Data files

All data can be found in the `./data` directory in .csv file format. Training and test data splits involve the same two sets of examples, although transformations applied to each pair of feature matrices may vary.

* **./data/diabetes.csv**

The original data, containing all feature columns and the `Outcome` response columns, with shape (768, 9).

* **./data/X_train.csv**

Training data split containing the 8 feature columns, with shape (614, 8).

* **./data/X_test.csv**

Test data split with shape (154, 8).

* **./data/Xs_train.csv**

Training examples of `X_train.csv` standardized to have zero sample mean and unit sample variance.

* **./data/Xs_test.csv**

Test examples of `X_test.csv` standardized to have zero sample mean and unit sample variance.

* **./data/Xe_train.csv**

Training excamples of `X_train.csv` augmented with 4 additional binary categorical variables, with shape (614, 12).

* **./data/Xe_test.csv**

Test examples of `X_test.csv` augmented with 4 additional binary categorical variables, with shape (154, 12).

* **./data/Xr_train.csv**

Standardized training examples of `Xs_train.csv`, but with PCA features. Shape (614, 8).

* **./data/Xr_test.csv**

Standardized test examples of `Xs_test.csv`, but with PCA features. Shape (154, 8).

* **./data/y_train.csv**

Response vector for training examples, shape (614, 1).

* **./data/y_test.csv**

Response vector for test examples, shape (154, 1).

## Models

Except for `gbcr_srcv.pickle`, all files are pickled `sklearn.model_selection.GridSearchCV` instances, to preserve cross-validation results alongside the best cross-validated estimator. `gbcr_srcv.pickle` is a pickled `shizuka.base.shizukaBaseCV` class instance. Cross-validation used was 5-fold cross validation, with accuracy as a scoring metric.

* **./models/lgr_gscv.pickle**

`GridSearchCV` results from training a `sklearn` regularized logistic regression model on `X_train` and `y_train`.

* **./models/gbc_gscv.pickle**

`GridSearchCV` results from training a `sklearn` gradient boosting model with tree stumps as base learners and with shrinkage on `X_train` and `y_train`.

* **./models/gbce_gscv.pickle**

`GridSearchCV` results from training a `sklearn` gradient boosting model with tree stumps as base learners and with shrinkage on `Xe_train` and `y_train`.

* **./models/gbcr_srcv.pickle**

`shizukaBaseCV` results from training a `sklearn` gradient boosting model with tree stumps as base learners and with shrinkage on `X_train` and `y_train`, but for each validation iteration, resampling the training folds using the `imblearn` implementation of SMOTE. Unlike `GridSearchCV` objects, the parameters were already fixed.

## Figures

Contains figures produced by `shizuka.plotting.multiclass_stats` and `shizuka.plotting.coef_plot` for each model used, with the file name corresponding to the model pickled in `./models`. Also includes `diabetes_classes.png`, a bar plot of the imbalance between the positive and negaative classes, `pca_plot.png`, which plots the first two principal components against each other, with points colored by their class membership, and `pca_variances.png`, a bar plot of the percentage of variation explained by each principal component.
