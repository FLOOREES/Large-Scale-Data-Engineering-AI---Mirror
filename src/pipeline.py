from pyspark.sql import SparkSession

from src.landing.landing import LandingZone
from src.formatted.formatted import FormattedZone
from src.trusted.trusted import TrustedZone
from src.exploitation.exploitation import ExploitationZone

class Pipeline:
	"""
	Main class to orchestrate the data pipeline.
	"""
	def __init__(self, spark: SparkSession, max_stage: int = 4):
		assert max_stage in [1, 2, 3, 4], "max_stage must be between 1 and 4"
		self.max_stage = max_stage
		self.spark = spark
		self.landing = LandingZone() # No spark is used in landing zone
		self.formatted = FormattedZone(spark=self.spark)
		self.trusted = TrustedZone(spark=self.spark)
		self.exploitation = ExploitationZone(spark=self.spark)

	def run(self):
		self.landing.run()
		if self.max_stage >= 2:
			self.formatted.run()
		if self.max_stage >= 3:
			self.trusted.run()
		if self.max_stage >= 4:
			self.exploitation.run()