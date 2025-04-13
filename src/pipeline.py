from pyspark.sql import SparkSession

from landing.landing import LandingZone
from formatted.formatted import FormattedZone
from trusted.trusted import TrustedZone
from exploitation.exploitation import ExploitationZone
from analysis.model import Model
from analysis.visualize_catalan_affordability import CatalanAffordabilityVisualizer

class Pipeline:
	"""
	Main class to orchestrate the data pipeline.
	"""
	def __init__(self, spark: SparkSession, max_stage: int = 5, analysis: str = "map"):
		assert max_stage in [1, 2, 3, 4, 5], "max_stage must be between 1 and 4"
		assert analysis in ["map", "model"], "analysis must be either 'map' or 'model'"

		self.analysis = analysis
		self.max_stage = max_stage
		self.spark = spark
		self.landing = LandingZone() # No spark is used in landing zone
		self.formatted = FormattedZone(spark=self.spark)
		self.trusted = TrustedZone(spark=self.spark)
		self.exploitation = ExploitationZone(spark=self.spark)
		self.map = CatalanAffordabilityVisualizer(spark=self.spark)
		self.model = Model(spark=self.spark)

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

		if self.max_stage >= 5:
			print(f"\n\nPIPELINE: STARTING ANALYSIS ({self.analysis})\n\n")
			if self.analysis == "map":
				self.map.run()
			elif self.analysis == "model":
				self.model.run()