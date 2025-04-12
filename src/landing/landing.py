from src.landing.idescat import IdescatLandingZone
from src.landing.lloguer import LloguerLandingZone
from src.landing.rfdbc import RFDBCLandingZone

idescat = IdescatLandingZone()
lloguer = LloguerLandingZone()
rfdbc = RFDBCLandingZone()

try: 
	idescat.run()
	lloguer.run()
	rfdbc.run()
except Exception as e:
	print(f"ERROR IN LANDING ZONE: {e}")
finally:
	print("LANDING ZONE COMPLETED")