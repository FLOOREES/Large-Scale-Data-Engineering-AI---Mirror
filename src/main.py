from src.spark_session import get_spark_session

from src.pipeline import Pipeline

if __name__ == "__main__":
	pipeline = Pipeline(spark=get_spark_session())
	pipeline.run()
	print("Pipeline completed successfully.")