import click
from joblib import dump
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

from step_counter.datasets import load_data_as_dataframe


@click.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("model_save_path", type=click.Path())
@click.option("--seed", "-s", "seed", type=int, default=42, help="Random seed")
@click.option(
    "--kfolds",
    "-kf",
    "kfolds",
    type=int,
    default=5,
    help="Number of folds for CV training",
)
def main(data_path: str, model_save_path: str, seed: int, kfolds: int):
    """
    Train and save logistic regression model

    DATA_PATH: path to input data
    MODEL_SAVE_PATH: path to save model
    """
    # load data
    data = load_data_as_dataframe(data_path)

    X = data[["x", "y", "z"]].values
    y = data["button_state"].values
    model = make_pipeline(
        preprocessing.StandardScaler(),
        LogisticRegression(
            random_state=seed,
            class_weight="balanced",
        ),
    )
    print("Training model with cross validation...")
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)
    for i, (train_index, test_index) in enumerate(kf.split(X), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        # TODO quickly evaluate model
        # TODO save debugging metrics and plots to file
        y_pred = model.predict(X_test)
        print(f"Accuracy score {i}/{kfolds}: {model.score(X_test, y_test):.2f}")
        print(f"Classification report {i}/{kfolds}:")
        print(classification_report(y_test, y_pred))
    # re-train the model on the entire training set
    print("Training model on entire training set...")
    model.fit(X, y)
    # TODO quickly evaluate model
    # TODO save debugging metrics and plots to file
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))

    # save model
    print(f"Saving model to {model_save_path}")
    dump(model, model_save_path)


if __name__ == "__main__":
    main()
