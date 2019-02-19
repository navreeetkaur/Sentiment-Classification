import sys
import json
from sklearn.metrics import *

TEST_FILE = sys.argv[1]
OUT_FILE = sys.argv[2]

y_pred=[]
for line in (open(OUT_FILE, 'r')):
    y_pred.append(float(line.strip()))

y_true=[]
i=0
for line in (open(TEST_FILE, 'r')):
	# if i>=50000:
	# 	break
	line = json.loads(line)
	y_true.append(line['ratings'])
	i+=1
    
print(mean_squared_error(y_true, y_pred))
