from src.spark_session import get_spark_session

from src.pipeline import Pipeline

if __name__ == "__main__":
	spark = get_spark_session()

	pipeline = Pipeline(spark=spark, max_stage=3)
	pipeline.run()
	print("Pipeline completed successfully.")