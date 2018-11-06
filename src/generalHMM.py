"""
***********************************************************
File: generalHMM.py
Author: Luke Burks
Date: November 2018

Implements a high level HMM with configurable options for
state, action, and observation sizes
Implements Viterbi and the forward algorithm
***********************************************************
"""


__author__ = "Luke Burks"
__copyright__ = "Copyright 2018, Luke Burks"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Luke Burks"
__email__ = "clburks9@gmail.com"
__status__ = "Development"



import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import yaml


class HMM:

	def __init__(self,inputFile=None): 

		#from config import *

		if(inputFile is not None):
			with open(inputFile,'r') as stream:
				cfg = yaml.load(stream); 
			self.states = cfg['States']
			self.obs = cfg['Observations']; 
			self.Iprob = cfg['Initial_Probability']; 
			self.Tprob = cfg['Transition_Probability']; 
			self.Oprob = cfg['Observation_Probability']; 
		else:
			self.states = []; 
			self.obs = []; 
			self.Iprob = {}; 
			self.Tprob = {}; 
			self.Oprob = {}; 


	def display(self):

		print("States:" + str(self.states)); 
		print(""); 
		print("Observations:"+str(self.obs)); 
		print(""); 
		print("Initial Probability: "); 
		print(self.Iprob); 
		print(""); 
		print("Transition Probability: "); 
		print(self.Tprob); 
		print(""); 
		print("Observation Probability: "); 
		print(self.Oprob); 




