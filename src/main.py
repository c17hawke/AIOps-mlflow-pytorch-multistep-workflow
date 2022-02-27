import mlflow


if __name__ == '__main__':
    with mlflow.start_run(run_name="main") as runs:
        mlflow.run(".", "get_data", use_conda=False)
        mlflow.run(".", "train", use_conda=False)
