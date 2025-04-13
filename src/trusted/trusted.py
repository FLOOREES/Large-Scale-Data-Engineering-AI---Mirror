from pyspark.sql import SparkSession

from trusted.idescat import IdescatTrustedZone
from trusted.lloguer import LloguerTrustedZone
from trusted.rfdbc import RFDBCTrustedZone

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