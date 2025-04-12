from src.landing.idescat import IdescatLandingZone
from src.landing.lloguer import LloguerLandingZone
from src.landing.rfdbc import RFDBCLandingZone

class LandingZone:
	def __init__(self):
		self.idescat = IdescatLandingZone()
		self.lloguer = LloguerLandingZone()
		self.rfdbc = RFDBCLandingZone()
	
	def run(self):
		try:
			self.idescat.run()
			self.lloguer.run()
			self.rfdbc.run()
		except Exception as e:
			print(f"ERROR IN LANDING ZONE: {e}")
		finally:
			print("LANDING ZONE COMPLETED")