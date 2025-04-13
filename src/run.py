from spark_session import get_spark_session
from setup import make_folder_structure

from pipeline import Pipeline

if __name__ == "__main__":
	make_folder_structure()
	spark = get_spark_session()

	pipeline = Pipeline(spark=spark, max_stage=5, analysis="map")
	pipeline.run()
	print("Pipeline completed successfully.")