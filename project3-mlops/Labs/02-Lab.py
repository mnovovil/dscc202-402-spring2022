# Databricks notebook source
# MAGIC %md
# MAGIC # Lab: Grid Search with MLflow
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you:<br>
# MAGIC  - Import the housing data
# MAGIC  - Perform grid search using scikit-learn
# MAGIC  - Log the best model on MLflow
# MAGIC  - Load the saved model
# MAGIC  
# MAGIC ## Prerequisites
# MAGIC - Web browser: Chrome
# MAGIC - A cluster configured with **8 cores** and **DBR 7.0 ML**

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Setup
# MAGIC 
# MAGIC For each lesson to execute correctly, please make sure to run the **`Classroom-Setup`** cell at the<br/>
# MAGIC start of each lesson (see the next cell) and the **`Classroom-Cleanup`** cell at the end of each lesson.

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Import
# MAGIC 
# MAGIC Load in same Airbnb data and create train/test split.

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv")
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform Grid Search using scikit-learn
# MAGIC 
# MAGIC We want to know which combination of hyperparameter values is the most effective. Fill in the code below to perform <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV" target="_blank"> grid search using `sklearn`</a> over the 2 hyperparameters we looked at in the 02 notebook, `n_estimators` and `max_depth`.

# COMMAND ----------

# TODO
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
 
# dictionary containing hyperparameter names and list of values we want to try
parameters = {'n_estimators': np.arange(1,10,1), 
              'max_depth': np.arange(1,10,1) }
 
rf = RandomForestRegressor()
grid_rf_model = GridSearchCV(rf, parameters, cv=3)
grid_rf_model.fit(X_train, y_train)
 
best_rf = grid_rf_model.best_estimator_
for p in parameters:
  print("Best '{}': {}".format(p, best_rf.get_params()[p]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Best Model on MLflow
# MAGIC 
# MAGIC Log the best model as `grid-random-forest-model`, its parameters, and its MSE metric under a run with name `RF-Grid-Search` in our new MLflow experiment.

# COMMAND ----------

# TODO
from sklearn.metrics import mean_squared_error
 
with mlflow.start_run(run_name= "RF-Grid-Search") as run:
  # log params
    parameters = {'n_estimators': np.arange(1,10,1), 
                  'max_depth': np.arange(1,10,1) }
    
  # Create predictions of X_test using best model
    rf = RandomForestRegressor()
    grid_rf_model = GridSearchCV(rf, parameters, cv=3)
    grid_rf_model.fit(X_train, y_train)
    predictions = grid_rf_model.predict(X_test)
    
  # Log model with name
    mlflow.sklearn.log_model(grid_rf_model, "grid-random-forest-model")
  
  # Create and log MSE metrics using predictions of X_test and its actual value y_test
    mse = mean_squared_error(y_test, predictions)
    print(f"mse: {mse}")
    print(f"best-params: {grid_rf_model.best_params_}")
    
        #log metrics and best params
    mlflow.log_metric("mse", mse)
    mlflow.log_params(grid_rf_model.best_params_)
    
    #record runID and experimentdID
    runID = run.info.run_uuid
    experimentID = run.info.experiment_id
    artifactURI = mlflow.get_artifact_uri()
    
    print("Inside MLflow Run with id {}, and experimentID {}, and artifactURI: {}".format(runID, experimentID, artifactURI))

# COMMAND ----------

# MAGIC %md
# MAGIC Check on the MLflow UI that the run `RF-Grid-Search` is logged has the best parameter values found by grid search.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Load the Saved Model
# MAGIC 
# MAGIC Load the trained and tuned model we just saved. Check that the hyperparameters of this model matches that of the best model we found earlier.
# MAGIC 
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Use the `artifactURI` variable declared above.

# COMMAND ----------

# TODO
artifactURI = "dbfs:/databricks/mlflow-tracking/185354766206883/241f40588e504b19b71d316ab752b706/artifacts"
mlflow.pyfunc.load_model(artifactURI+"/grid-random-forest-model")

# COMMAND ----------

# MAGIC %md
# MAGIC Time permitting, continue to grid search over a wider number of parameters and automatically save the best performing parameters back to `mlflow`.

# COMMAND ----------

# TODO
# TODO
from sklearn.metrics import mean_squared_error
 
with mlflow.start_run(run_name= "RF-Grid-Search") as run:
  # log params
    parameters = {'n_estimators': np.arange(1,1000,1), 
                  'max_depth': np.arange(1,20,5) }
    
  # Create predictions of X_test using best model
    rf = RandomForestRegressor()
    grid_rf_model = GridSearchCV(rf, parameters, cv=3)
    grid_rf_model.fit(X_train, y_train)
    predictions = grid_rf_model.predict(X_test)
    
  # Log model with name
    mlflow.sklearn.log_model(grid_rf_model, "grid-random-forest-model")
  
  # Create and log MSE metrics using predictions of X_test and its actual value y_test
    mse = mean_squared_error(y_test, predictions)
    print(f"mse: {mse}")
    print(f"best-params: {grid_rf_model.best_params_}")
    
        #log metrics and best params
    mlflow.log_metric("mse", mse)
    mlflow.log_params(grid_rf_model.best_params_)
    
    #record runID and experimentdID
    runID = run.info.run_uuid
    experimentID2 = run.info.experiment_id
    artifactURI2 = mlflow.get_artifact_uri()
    
    print("Inside MLflow Run with id {}, and experimentID {}, and artifactURI: {}".format(runID, experimentID2, artifactURI2))

# COMMAND ----------

# MAGIC %md
# MAGIC Time permitting, use the `MlflowClient` to interact programatically with your run.

# COMMAND ----------

# TODO
# TODO
artifactURI2 = "" #supposed to plug in the artifact ID that I get from the previous one
mlflow.pyfunc.load_model(artifactURI2+"/grid-random-forest-model")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> See the solutions folder for an example solution to this lab.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Cleanup<br>
# MAGIC 
# MAGIC Run the **`Classroom-Cleanup`** cell below to remove any artifacts created by this lesson.

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Cleanup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Next Steps
# MAGIC 
# MAGIC Start the next lesson, [Packaging ML Projects]($../03-Packaging-ML-Projects ).

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
