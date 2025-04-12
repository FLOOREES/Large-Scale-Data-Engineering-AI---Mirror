from pyspark.sql import SparkSession

from src.trusted.idescat import IdescatTrustedZone
from src.trusted.lloguer import LloguerTrustedZone
from src.trusted.rfdbc import RFDBCTrustedZone

class TrustedZone:
	def __init__(self, spark: SparkSession):
		self.spark = spark
		self.idescat = IdescatTrustedZone(spark=self.spark)
		self.lloguer = LloguerTrustedZone(spark=self.spark)
		self.rfdbc = RFDBCTrustedZone(spark=self.spark)

	def run(self):
		self.idescat.run()
		self.lloguer.run()
		self.rfdbc.run()