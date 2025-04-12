from pyspark.sql import SparkSession

from src.landing.landing import LandingZone
from src.formatted.formatted import FormattedZone
from src.trusted.trusted import TrustedZone
from src.exploitation.exploitation import ExploitationZone

class Pipeline:
	"""
	Main class to orchestrate the data pipeline.
	"""
	def __init__(self, spark: SparkSession):
		self.spark = spark
		self.landing = LandingZone(spark=self.spark)
		self.formatted = FormattedZone(spark=self.spark)
		self.trusted = TrustedZone(spark=self.spark)
		self.exploitation = ExploitationZone(spark=self.spark)

	def run(self):
		self.landing.run()
		self.formatted.run()
		self.trusted.run()
		self.exploitation.run()