from src.landing.idescat import Idescat
from src.landing.lloguer import Lloguer
from src.landing.rfdbc import RFDBC

idescat = Idescat()
lloguer = Lloguer()
rfdbc = RFDBC()

try: 
	idescat.run()
	lloguer.run()
	rfdbc.run()
except Exception as e:
	print(f"ERROR IN LANDING ZONE: {e}")
finally:
	print("LANDING ZONE COMPLETED")