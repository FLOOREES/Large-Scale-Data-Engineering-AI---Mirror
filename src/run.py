"""Main file to run the entire pipeline of the project."""

from spark_session import get_spark_session
from setup import make_folder_structure
import geo_relations

from pipeline import Pipeline

if __name__ == "__main__":
	make_folder_structure()
	spark = get_spark_session()

	# Generate geographic relations
	geo_relations.main()

	# Stages:
	# 1. Landing Zone
	# 2. Formatted Zone
	# 3. Trusted Zone
	# 4. Exploitation Zone
	# 5. Analysis (model, visualizer, or both)

	pipeline = Pipeline(spark=spark, start_stage=4, max_stage=4, analysis="both")
	pipeline.run()
