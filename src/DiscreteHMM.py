"""
***********************************************************
File: DiscreteHMM.py
Author: Luke Burks
Date: November 2018

Subclasses a general HMM for discrete states, actions, 
and observations.

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

class DHMM(HMM):

	def __init__(self,inputFile=None): 

		super().__init__(inputFile);




	def forward(self,obs,normed=False):
		alpha = []; 
		alpha.append({k:v*self.Oprob[k][obs[0]] for k,v in self.Iprob.items()}); 
		#print(alpha[0])

		for i in range(1,len(obs)):
			alpha.append({key:1 for key in self.states}); 

			for s in self.states:
				alpha[i][s] = self.Oprob[s][obs[i]]*sum(self.Tprob[sprime][s]*alpha[i-1][sprime] for sprime in self.states); 
			
			if(normed):
				suma = sum([alpha[i][st] for st in self.states]); 
				#print(suma,alpha[i])
				for st in self.states:
					alpha[i][st] /= suma; 
				#print(alpha[i],sum([alpha[i][s] for s in self.states])); 


		alpha.insert(0,self.Iprob);

		return alpha; 



	def backward(self,obs,normed=False):

		
		beta = []; 
		for i in range(0,len(obs)):
			beta.append({key:1 for key in self.states}); 

		

		for i in range(len(obs)-2,-1,-1):
			for s in self.states:
				beta[i][s] = sum(self.Tprob[s][sprime]*self.Oprob[sprime][obs[i]]*beta[i+1][sprime] for sprime in self.states); 
				if(normed):
					suma = sum([beta[i][st] for st in self.states]); 
					for st in self.states:
						beta[i][st] /= suma; 


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



	def simulate(self,steps,initState):

		states = []; 
		obs = []; 

		states.append(initState);
		keys,vals = zip(*self.Oprob[initState].items()); 
		obs.append(np.random.choice(keys,p=vals)); 

		for step in range(0,steps):

			keys,vals = zip(*self.Tprob[states[-1]].items());
			states.append(np.random.choice(keys,p=vals));

			keys,vals = zip(*self.Oprob[states[-1]].items()); 
			obs.append(np.random.choice(keys,p=vals));

		return states,obs


	def viterbi(self,obs,initState=None):
		#From Wikipedia page on viterbi
		V=[{}]; 
		for st in self.states:
			V[0][st] = {"prob":self.Iprob[st]*self.Oprob[st][obs[0]],"prev":None};

		for t in range(1,len(obs)):
			V.append({}); 
			for st in self.states:
				max_tr_prob = max(V[t-1][prev]["prob"]*self.Tprob[prev][st] for prev in self.states); 

				for prev in self.states:
					if(V[t-1][prev]["prob"]*self.Tprob[prev][st] == max_tr_prob):
						max_prob = max_tr_prob*self.Oprob[st][obs[t]]; 
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


		return opt; 


	def baumWelch(self,obs,maxIter = 100):
		
		print("*********************************************************************************************")
		print("Warning!!!!! The Baum-Welch Method has not been completed, and will just output junk answers"); 
		print("*********************************************************************************************")

		for count in range(0,maxIter):

			alpha = self.forward(obs,normed=True)[1:]; 
			beta = self.backward(obs,normed=True); 
			
			# for t in range(0,len(alpha)):
			# 	suma = sum(alpha[t].values()); 
			# 	#print(alpha[t].values())
			# 	for key in alpha[t].keys():
			# 		alpha[t][key] /= suma; 

			# for t in range(0,len(beta)):
			# 	suma = sum(beta[t].values()); 
			# 	for key in beta[t].keys():
			# 		beta[t][key] /= suma; 

			gamma = []; 
			for i in range(0,len(alpha)):
				gamma.append({key:0 for key in self.states}); 


			for t in range(0,len(alpha)):
				for s in self.states:
					numer = alpha[t][s]*beta[t][s]; 
					denom = 0; 
					for sprime in self.states:
						denom += alpha[t][sprime]*beta[t][sprime]; 
					gamma[t][s] = numer/denom; 

			chi = []; 
			for i in range(0,len(alpha)):
				chi.append({key:{key:0 for key in self.states} for key in self.states}); 

			for t in range(0,len(alpha)-1):
				for s in self.states:
					for sprime in self.states:
						numer = alpha[t][s]*self.Tprob[s][sprime]*self.Oprob[sprime][obs[t+1]]*beta[t+1][sprime];
						denom = 0; 
						for s2 in self.states:
							for s3 in self.states:
								denom += alpha[t][s2]*self.Tprob[s2][s3]*self.Oprob[s3][obs[t+1]]*beta[t+1][s3];
						chi[t][s][sprime] = numer/denom; 

			self.Iprob = gamma[0]; 
			

			for s in self.states:
				for sprime in self.states:
					self.Tprob[s][sprime] = sum([chi[t][s][sprime] for t in range(0,len(chi)-1)])/sum(gamma[t][s] for t in range(0,len(chi)-1)); 
			

			# for i in range(0,len(gamma)):
			# 	print(gamma[i]); 

			for s in self.states:
				for o in self.obs:
					numer = 0; 
					denom = 0; 
					for t in range(0,len(gamma)):
						if(obs[t] == o):
							numer += gamma[t][s]; 
						denom += gamma[t][s]; 
					self.Oprob[s][o] = numer/denom; 

		print("Initial:")
		print(self.Iprob); 
		
		print()
		print("Transition:"); 
		for s in self.states:
			st = ""; 
			for sprime in self.states:
				st = st + sprime + ": " + "{0:.2f}, ".format(self.Tprob[s][sprime])
			print(s + ": " + st); 

		print();

		print("Observation:"); 
		for s in self.states:
			st = ""; 
			for o in self.obs:
				st = st + o + ": " + "{0:.2f}, ".format(self.Oprob[s][o])
			print(s + ": " + st); 





def testForwardBackward(h):
	obs = ['Cloudy','Cloudy','Rainy','Sunny']; 
	alpha = h.forward(obs)
	both = h.forwardBackward(obs); 
	
	fig,axarr = plt.subplots(2); 

	#Forward Plot
	probs = {'Sunny':[],'Cloudy':[],'Rainy':[]}

	for a in alpha:
		for key,value in a.items():
			probs[key].append(value); 

	keys = probs.keys(); 

	
	for key in keys:
		axarr[0].plot(probs[key]); 
	axarr[0].legend(keys); 
	axarr[0].set_title("Forward"); 



	#Both Plot
	probs = {'Sunny':[],'Cloudy':[],'Rainy':[]}

	for a in both:
		for key,value in a.items():
			probs[key].append(value); 

	keys = probs.keys(); 

	
	for key in keys:
		axarr[1].plot(probs[key]); 
	axarr[1].legend(keys); 
	axarr[1].set_title("Both"); 


	plt.show(); 


def testViterbi(h):
	obs = ['Cloudy','Cloudy','Rainy','Sunny']; 
	seq = h.viterbi(obs,'Cloudy'); 

	print(seq); 

def testSimulate(h):
	states,obs = h.simulate(1000,'Sunny');

	counts = Counter(states); 
	for key in counts.keys():
		counts[key]/=len(states); 

	#Histogram should be 9/14,4/14,1/14 if using original values
	plt.bar(counts.keys(),counts.values()); 
	plt.axhline(y=9/14,color='k',ls='--'); 
	plt.axhline(y=4/14,color='k',ls='--'); 
	plt.axhline(y=1/14,color='k',ls='--'); 
	plt.show(); 


def simAndViterbiTest(h):
	states,obs = h.simulate(1000,'Spring');

	seq = h.viterbi(obs,'Spring'); 
	
	perCor = 0; 
	for i in range(0,len(states)):
		if(states[i] == seq[i]):
			perCor += 1; 
	perCor /=len(states); 
	print("Viterbi Accuracy: {0:.2f}%".format(perCor*100))


def simAndForwardBackward(h):

	init = np.random.choice(list(h.Iprob.keys()),p=list(h.Iprob.values()))

	states,obs = h.simulate(25,init); 

	probs = h.forwardBackward(obs); 

	perCorFB = 0; 
	for i in range(1,len(states)):
		m = max(probs[i],key=probs[i].get); 
		if(m==states[i]):
			perCorFB += 1; 
	perCorFB /=len(states); 
	print("F/B Accuracy: {0:.2f}%".format(perCorFB*100))		

	seq = h.viterbi(obs,'Sunny'); 
	
	#seq.remove(seq[0])

	perCorV = 0; 
	for i in range(0,len(states)):
		if(states[i] == seq[i]):
			perCorV += 1; 
	perCorV /=len(states); 
	print("Viterbi Accuracy: {0:.2f}%".format(perCorV*100))


	#allProbs = {'Sunny':[],'Cloudy':[],'Rainy':[]}
	allProbs = {key:[] for key in h.Iprob.keys()}; 

	for a in probs:
		for key,value in a.items():
			allProbs[key].append(value); 

	#keys = ['Sunny','Cloudy','Rainy'];
	keys = h.states;  
	cols = ['r','g','b','c','m','y','k']; 


	for i in range(0,len(keys)):
		plt.plot(allProbs[keys[i]],color=cols[i]); 

	for i in range(0,len(states)-1):
		plt.axvline(i,color=cols[keys.index(states[i])],ls='--'); 
		plt.axvline(i+0.25,color=cols[keys.index(seq[i])],ls=':')


	plt.legend(keys); 
	plt.title("F/B"); 
	plt.show();


def testBaumWelch(h):
	hBW = DHMM(); 
	numStates = 4; 
	numObs = 4; 
	obs = ['Sun','Snow','Rain','Tornado'];
	states = []; 
	for i in range(0,numStates):
		states.append(str(i)); 
	hBW.states = states; 
	hBW.obs = obs; 
	#Initialize Uniform Transitions, Initials, and Observations
	tmat = {}; 
	for i in range(0,numStates):
		tmat[str(i)] = {}; 
		suma = 0; 
		for j in range(0,numStates):
			tmat[str(i)][str(j)] = np.random.random(); 
			suma += tmat[str(i)][str(j)]; 
		for j in range(0,numStates):
			tmat[str(i)][str(j)] /= suma; 
	hBW.Tprob = tmat; 

	imat = {}; 
	suma = 0; 
	for i in range(0,numStates):
		imat[str(i)] = np.random.random(); 
		suma += imat[str(i)];
	for i in range(0,numStates):
		imat[str(i)] /= suma; 
	hBW.Iprob = imat; 

	omat = {}; 
	for i in range(0,numStates):
		omat[str(i)] = {}; 
		suma = 0; 
		for o in obs:
			omat[str(i)][o] = np.random.random(); 
			suma += omat[str(i)][o]; 
		for o in obs:
			omat[str(i)][o] /= suma; 
	hBW.Oprob = omat; 

	#Get training data
	timeSteps = 1000; 
	init = np.random.choice(list(h.Iprob.keys()),p=list(h.Iprob.values()))
	states,obs = h.simulate(timeSteps-1,init); 

	
	#do Baum-Welch
	hBW.baumWelch(obs,maxIter = 1000); 

	#t1 = h.forward(obs); 
	# t2 = h.forward(obs,True); 

	# for i in range(0,len(t1)):

	# 	st = ""; 
	# 	for sprime in h.states:
	# 		st = st + sprime + ": " + "{0:.2f}, ".format(t1[i][sprime])

	# 	print(st); 

	# 	st = ""; 
	# 	for sprime in h.states:
	# 		st = st + sprime + ": " + "{0:.2f}, ".format(t2[i][sprime])
		
	# 	print(st); 
	# 	print(); 




if __name__ == '__main__':
	h = DHMM('../yaml/baumTest.yaml');  
	

	#h.display()

	#simAndViterbiTest(h); 
	#testViterbi(h); 
	#testSimulate(h); 
	#testForwardBackward(h); 
	#simAndForwardBackward(h); 

	testBaumWelch(h); 



