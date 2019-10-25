import mlflow.sklearn
from pyspark.sql import SparkSession
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from example import GenericPythonFunctionEstimator


# let's prepare split the data with a training and test set
def prepare_data(products):
    le = LabelEncoder()
    le.fit(products["category"])
    X_train, X_test, y_train, y_test = train_test_split(products["description"], le.transform(products["category"]),
                                                        test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test, le


# let's build our Scikit Learn pipeline
def get_scikit_learn_pipeline(use_idf=True, norm='l1'):
    return Pipeline([
        ('vect', CountVectorizer(stop_words='english', analyzer='word', strip_accents="ascii")),
        ('tfidf', TfidfTransformer(use_idf=use_idf, norm=norm)),
        ('randomForest', GenericPythonFunctionEstimator())
    ])


if __name__ == "__main__":
    spark = SparkSession.Builder().getOrCreate()
    products = spark.sql("select * from quentin.products limit 2000").toPandas()
    delta_version = spark.sql("SELECT MAX(version) AS VERSION FROM (DESCRIBE HISTORY quentin.products)").head()[0]

    X_train, X_test, y_train, y_test, le = prepare_data(products)
    with mlflow.start_run():
        # ###################################### #
        # let's train our Scikit Learn Pipeline: #
        # ###################################### #
        pipeline = get_scikit_learn_pipeline(use_idf=True, norm='l1')
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        f1 = metrics.f1_score(y_test, predictions, average='micro')

        # ####################################### #
        # Logging parameters and model to MLFlow  #
        # ####################################### #
        mlflow.log_param("num_trees", 50)
        mlflow.log_param("delta_version", delta_version)
        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(pipeline, "model")
        mlflow.log_artifact("/dbfs/quentin/demo/mlflow/confusion_matrix.png", "model")