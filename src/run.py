from src.spark_session import get_spark_session
from src.setup import make_folder_structure

from src.pipeline import Pipeline

if __name__ == "__main__":
	make_folder_structure()
	spark = get_spark_session()

	pipeline = Pipeline(spark=spark, max_stage=3)
	pipeline.run()
	print("Pipeline completed successfully.")