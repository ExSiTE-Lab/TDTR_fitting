#TDTR_fitting can either be edited to update the thermal properties matrix, specify the filename, etc, or it can be called externally like so. This makes it very convenient to analyze data en masse, with varying projects able to always reference the same code file (and always be using the most up-to-date code available). 

import numpy as np
from TDTR_fitting import *


	   #C       Kz	  d	 Kr
PLSB_SiO2=[[2.42e6, 110.,  80e-9,  "Kz"],
           [        1/150e6             ], # we default to using TBR. you can change this by running 'setVar("useTBR",False)'
           [1.63e6,  1.35,  1.0,    "Kz"]]

# Set global variables within TDTR_fitting.py like this. e.g. the thermal properties matrix, or the parameters we want to fit. 
setVar("tp",PLSB_SiO2)
setVar("tofit",["Kz2","R1"])

# some things are defaulted, but it does no harm to set them explicitly. if you wanted to fit SSTR data for example, the "solve" function below will work, but you would change the mode to "SSTR"
# setVar("mode","TDTR") 

# Use glob to search for fles matching the pattern. "files" below is a list
import glob
direc="testscripts/2019_10_23_PLSB/"
pattern="*SiO2*"
files=glob.glob(direc+pattern) 

# we can solve these individually, one at a time. note that experimental conditions (spot sizes, modulation frequency) are read in automatically
# for f in files:
#	print(solve(f))

setVar("verbose",["comprehensiveUncertainty"]) # if you want the code to spit out progress updates, you can list functions here
# you can also process all files at once, including calculating uncertainty in multiple ways! 
print( comprehensiveUncertainty(files) )



