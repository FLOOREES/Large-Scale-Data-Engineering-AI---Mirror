from pyspark.sql import SparkSession
from spark_session import get_spark_session

from landing.landing import LandingZone
from formatted.formatted import FormattedZone
from trusted.trusted import TrustedZone
from exploitation.exploitation import ExploitationZone
from analysis.kg_embeddings import KGEmbeddings
from analysis.kg_query import KGQueryPipeline

class Pipeline:
	"""
	Main class to orchestrate the data pipeline.
	"""
	def __init__(self, start_stage: int = 1, max_stage: int = 5, analysis: str = "both", kg_embeddings_config: dict = None):
		assert start_stage in [1, 2, 3, 4, 5], "start_stage must be between 1 and 5"
		assert max_stage in [1, 2, 3, 4, 5], "max_stage must be between 1 and 5"
		assert start_stage <= max_stage, "start_stage cannot be greater than max_stage"
		assert analysis in ["query", "embeddings", "both"], "analysis must be either 'query', 'embeddings', or 'both'"

		if start_stage < 5:
			self.spark = get_spark_session()
			self.landing = LandingZone() # No spark is used in landing zone
			self.formatted = FormattedZone(spark=self.spark)
			self.trusted = TrustedZone(spark=self.spark)
			self.exploitation = ExploitationZone(spark=self.spark)
		else:
			self.spark = None

		self.kg_embeddings_config = kg_embeddings_config
		self.analysis = analysis
		self.start_stage = start_stage
		self.max_stage = max_stage

		if self.analysis == "both":
			self.analysis_embeddings = KGEmbeddings(**kg_embeddings_config)
			self.analysis_query = None
		elif self.analysis == "embeddings":
			self.analysis_embeddings = KGEmbeddings(**kg_embeddings_config)
			self.analysis_query = None
		elif self.analysis == "query":
			self.analysis_embeddings = None
			self.analysis_query = None

	def run(self):
		if self.start_stage <= 1 <= self.max_stage:
			print("-"*100)
			print("\n\nPIPELINE: STARTING LANDING ZONE (STAGE 1)\n\n")
			print("-"*100)
			self.landing.run()
		
		if self.start_stage <= 2 <= self.max_stage:
			print("-"*100)
			print("\n\nPIPELINE: STARTING FORMATTED ZONE (STAGE 2)\n\n")
			print("-"*100)
			self.formatted.run()
		
		if self.start_stage <= 3 <= self.max_stage:
			print("-"*100)
			print("\n\nPIPELINE: STARTING TRUSTED ZONE (STAGE 3)\n\n")
			print("-"*100)
			self.trusted.run()
		
		if self.start_stage <= 4 <= self.max_stage:
			print("-"*100)
			print("\n\nPIPELINE: STARTING EXPLOITATION ZONE (STAGE 4)\n\n")
			print("-"*100)
			self.exploitation.run()

		if self.start_stage <= 5 <= self.max_stage:
			print("-"*100)
			print(f"\n\nPIPELINE: STARTING ANALYSIS (STAGE 5) ({self.analysis})\n\n")
			print("-"*100)
			if self.analysis == "both":
				self.analysis_query.run()
				self.analysis_embeddings.run()
			elif self.analysis == "query":
				self.analysis_query.run()
			elif self.analysis == "embeddings":
				self.analysis_embeddings.run()

		print("-"*100)
		print("\n\nPIPELINE ENDED\n\n")
		print("-"*100)
		if self.spark is not None:
			# Stop the Spark session if it was created
			self.spark.stop()