from pyspark.sql import SparkSession

from formatted.idescat import IdescatFormattedZone
from formatted.lloguer import LloguerFormattedZone
from formatted.rfdbc import RFDBCFormattedZone

class FormattedZone:
	def __init__(self, spark: SparkSession):
		self.spark = spark
		self.idescat = IdescatFormattedZone(spark=self.spark)
		self.lloguer = LloguerFormattedZone(spark=self.spark)
		self.rfdbc = RFDBCFormattedZone(spark=self.spark)
	
	def run(self):
		try:
			self.idescat.run()
			self.lloguer.run()
			self.rfdbc.run()
		except Exception as e:
			print(f"ERROR IN FORMATTED ZONE: {e}")
		finally:
			print("FORMATTED ZONE COMPLETED")