import numpy as np
import os
files = os.listdir(os.path.join("not_sammy"))
for i in range(len(files)):
	os.rename(os.path.join("not_sammy", files[i]), os.path.join("not_sammy", "{}.jpg".format(i)))
