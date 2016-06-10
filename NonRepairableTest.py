
import reliability as r

fail = False

try:
	a = r.NonRepairable(-1, 1)
except:
	print("alpha < 0 caught")
	fail = True

try:
	a = r.NonRepairable(1, -1)
except:
	print("beta < 0 caught")

if fail:
	print("Test Failed")