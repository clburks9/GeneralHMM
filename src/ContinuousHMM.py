"""
***********************************************************
File: ContinuousHMM.py
Author: Luke Burks
Date: November 2018

Subclasses a general HMM for continuous states

For both categorical (discrete/softmax) obs and continuous

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

from generalHMM import HMM
from gaussianMixtures import Gaussian,GM 
from softmaxModels import Softmax


class CHMM(HMM):

	def __init__(self,inputFile=None): 

		super().__init__(inputFile);

		self.Oprob = GM(self.Oprob['Means'],self.Oprob['Vars'],self.Oprob['Weights']); 


	

	def simulate(self,steps,initState=None):

		if(initState is None):
			initState = self.states[0]

		states = []; 
		obs = []; 

		states.append(initState); 


		for step in range(0,steps):
			#get new state
			keys,vals = zip(*self.Tprob[states[-1]].items());
			states.append(np.random.choice(keys,p=vals));


			newGM = GM()
			newGM.addG(self.Oprob[self.states.index(states[-1])]); 
			obs.append(newGM.sample(1)[0]);

		return states,obs; 


	def forward(self,obs):
		alpha = []; 
		alpha.append({k:v*self.Oprob[self.states.index(k)].pointEval(obs[0]) for k,v in self.Iprob.items()}); 

		for i in range(1,len(obs)):
			alpha.append({key:1 for key in self.states}); 

			for s in self.states:
				alpha[i][s] = self.Oprob[self.states.index(s)].pointEval(obs[i])*sum(self.Tprob[sprime][s]*alpha[i-1][sprime] for sprime in self.states); 

		#alpha.remove(alpha[0]); 
		alpha.insert(0,self.Iprob);
		return alpha;

	def backward(self,obs):


		beta = []; 
		for i in range(0,len(obs)+1):
			beta.append({key:1 for key in self.states}); 

		for i in range(len(obs)-1,-1,-1):
			#print(obs[i])
			for s in self.states:
				beta[i][s] = sum(self.Tprob[s][sprime]*self.Oprob[self.states.index(sprime)].pointEval(obs[i])*beta[i+1][sprime] for sprime in self.states);


		return beta; 



	def forwardBackward(self,obs):



		alpha = self.forward(obs); 
		beta = self.backward(obs); 

		posterior = []; 
		for t in range(0,len(alpha)):
			posterior.append({key:1 for key in self.states}); 

			for s in self.states:
				posterior[t][s] = alpha[t][s]*beta[t][s]/sum(alpha[t][sprime]*beta[t][sprime] for sprime in self.states); 


		return posterior; 






	def viterbi(self,obs,initState = None):
		#From Wikipedia page on viterbi
		V=[{}]; 
		for st in self.states:
			V[0][st] = {"prob":self.Iprob[st]*self.Oprob[self.states.index(st)].pointEval(obs[0]),"prev":None};

		for t in range(1,len(obs)):
			V.append({}); 
			for st in self.states:
				max_tr_prob = max(V[t-1][prev]["prob"]*self.Tprob[prev][st] for prev in self.states); 

				for prev in self.states:
					if(V[t-1][prev]["prob"]*self.Tprob[prev][st] == max_tr_prob):
						max_prob = max_tr_prob*self.Oprob[self.states.index(st)].pointEval(obs[t]); 
						V[t][st] = {"prob":max_prob,"prev":prev}; 
						break;
		opt = []; 
		max_prob = max(value["prob"] for value in V[-1].values()); 
		previous = None; 
		for st, data in V[-1].items():
			if data["prob"] == max_prob:
				opt.append(st); 
				previous = st; 
				break; 

		for t in range(len(V)-2,-1,-1):
			opt.insert(0,V[t+1][previous]["prev"]); 
			previous = V[t+1][previous]["prev"]; 

		opt.insert(0,initState); 
		return opt; 


def testSimulate(h):
	states,obs = h.simulate(10,'Fall'); 
	print(states); 
	print(obs); 
	
def testForward(h):
	states,obs = h.simulate(30,'Fall'); 

	alphas = h.forward(obs);

	for i in range(0,len(alphas)):
		suma = 0; 
		for s in h.states:
			suma += alphas[i][s]; 
		for s in h.states:
			alphas[i][s]/=suma; 
	

	allProbs = {key:[] for key in h.Iprob.keys()}; 

	for a in alphas:
		for key,value in a.items():
			allProbs[key].append(value); 

	keys = h.states;  
	cols = ['r','g','b','c','m','y','k']; 

	for i in range(0,len(keys)):
		plt.plot(allProbs[keys[i]],color=cols[i]); 
	for i in range(0,len(states)-1):
		plt.axvline(i,color=cols[keys.index(states[i])],ls='--'); 

	perCorFB = 0; 
	for i in range(1,len(states)):

		m = max(alphas[i],key=alphas[i].get); 
		if(m==states[i]):
			perCorFB += 1; 
	perCorFB /=len(states); 
	print("Forward Accuracy: {0:.2f}%".format(perCorFB*100))


	plt.show(); 

def testForwardBackward(h):
	states,obs = h.simulate(30,'Fall'); 

	probs = h.forwardBackward(obs); 

	allProbs = {key:[] for key in h.Iprob.keys()}; 

	for a in probs:
		for key,value in a.items():
			allProbs[key].append(value); 

	keys = h.states;  
	cols = ['r','g','b','c','m','y','k']; 

	for i in range(0,len(keys)):
		plt.plot(allProbs[keys[i]],color=cols[i]); 
	for i in range(0,len(states)-1):
		plt.axvline(i,color=cols[keys.index(states[i])],ls='--'); 

	perCorFB = 0; 
	for i in range(1,len(states)):

		m = max(probs[i],key=probs[i].get); 
		if(m==states[i]):
			perCorFB += 1; 
	perCorFB /=len(states); 
	print("F/B Accuracy: {0:.2f}%".format(perCorFB*100))

	plt.show();

def testViterbi(h):
	states,obs = h.simulate(30,'Fall'); 
	
	seq = h.viterbi(obs,'Fall'); 

	perCor = 0; 
	for i in range(0,len(states)):
		if(states[i] == seq[i]):
			perCor += 1; 
	perCor /=len(states); 
	print("Viterbi Accuracy: {0:.2f}%".format(perCor*100))
	

if __name__ == '__main__':
	h = CHMM('../yaml/conConTest.yaml');  

	#testSimulate(h); 
	#testForward(h); 
	#testForwardBackward(h); 
	testViterbi(h); 


	# a = Softmax();

	# a.buildGeneralModel(1,4,[[0,1],[1,2],[2,3]],np.matrix([-1,55,-1,65,-1,70]).T)

	# print(a.weights); 
	# print(a.bias); 

	# a.plot1D(low=0,high=100,res=100); 



