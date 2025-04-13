from pyspark.sql import SparkSession

from landing.landing import LandingZone
from formatted.formatted import FormattedZone
from trusted.trusted import TrustedZone
from exploitation.exploitation import ExploitationZone

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
		print("\n\nPIPELINE: STARTING LANDING ZONE\n\n")
		self.landing.run()
		if self.max_stage >= 2:
			print("\n\nPIPELINE: STARTING FORMATTED ZONE\n\n")
			self.formatted.run()
		if self.max_stage >= 3:
			print("\n\nPIPELINE: STARTING TRUSTED ZONE\n\n")
			self.trusted.run()
		if self.max_stage >= 4:
			print("\n\nPIPELINE: STARTING EXPLOITATION ZONE\n\n")
			self.exploitation.run()