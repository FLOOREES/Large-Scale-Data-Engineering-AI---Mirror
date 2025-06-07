from pyspark.sql import SparkSession

from utils import generate_gif

from landing.landing import LandingZone
from formatted.formatted import FormattedZone
from trusted.trusted import TrustedZone
from exploitation.exploitation import ExploitationZone
from analysis.model import Model
from analysis.visualizer import CatalanAffordabilityVisualizer

class Pipeline:
	"""
	Main class to orchestrate the data pipeline.
	"""
	def __init__(self, spark: SparkSession, start_stage: int = 1, max_stage: int = 5, analysis: str = "both"):
		assert start_stage in [1, 2, 3, 4, 5], "start_stage must be between 1 and 5"
		assert max_stage in [1, 2, 3, 4, 5], "max_stage must be between 1 and 5"
		assert start_stage <= max_stage, "start_stage cannot be greater than max_stage"
		assert analysis in ["model", "visualizer", "both"], "analysis must be either 'model' or 'visualizer' or 'both'"

		self.analysis = analysis
		self.start_stage = start_stage
		self.max_stage = max_stage
		self.spark = spark
		self.landing = LandingZone() # No spark is used in landing zone
		self.formatted = FormattedZone(spark=self.spark)
		self.trusted = TrustedZone(spark=self.spark)
		self.exploitation = ExploitationZone(spark=self.spark)
		self.visualizer = CatalanAffordabilityVisualizer(spark=self.spark)
		self.model = Model(spark=self.spark)

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
				self.visualizer.run()
				self.model.run()
				generate_gif()
			elif self.analysis == "visualizer":
				self.visualizer.run()
				generate_gif()
			elif self.analysis == "model":
				self.model.run()

		print("-"*100)
		print("\n\nPIPELINE ENDED\n\n")
		print("-"*100)
		self.spark.stop()