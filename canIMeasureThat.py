# Answer the question "can i measure [parameter] using [technique], and if so, under what conditions?"
# eg:
#  from canIMeasureThat import *
#  importMatrix("tpropsMatrixFile.txt")
#  setVar("tofit",["G1","Kz2"]) # or whatever thermal properties you'll be fitting for. the first is what we run preductUncert on
#  checkTechnique("mfTDTR") # or whatever technique you want to check

from TDTR_fitting import *
import itertools,time,os

upperAlphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
sFrqs=[100,1e3,5e3,10e3,25e3,50e3] ; tFrqs=[1e6,2.1e6,4e6,8.4e6]
sRads=[.5e-6,1e-6,1.5e-6,3e-6,5e-6,7.5e-6,10e-6,15e-6] ; tRads=[5e-6,7.5e-6,10e-6,15e-6]

freqs={"PWA":sFrqs,"TDTR":tFrqs,"SSTR":[1e2,1e3,1e4,1e5,1e6,1e7],"FDTR":[1]}
rads={"PWA":sRads,"TDTR":tRads,"SSTR":sRads,"FDTR":sRads}

setVar("fitRisePWA",True) ; setVar("sumNPWA",5e3)
nw={True:7,False:40}[os.uname()[1]=='everest'] ; setVar("quietWarnings",True)
setVar("useTBR",False)

threshold=2.5 # as a %

def main():
	#global freqs ; freqs["SSTR"]=[500,10e6]
	#importMatrix("testscripts/testing48/.calmats/Fiber_Si_cal_matrix.txt")
	importMatrix("testscripts/testing48/.calmats/GeSi_matrix.txt")
	setVar("tofit",[,"G1","G1","Kz2"])
	#checkTechnique("mfTDTR+SSTR")
	#checkTechnique("mfSSTR")
	#checkTechnique("mfPWA")
	checkTechnique("mfTDTR")

shutil.rmtree("canIMeasureThat_cache/",ignore_errors=True)
os.makedirs("canIMeasureThat_cache")

singleCounter=0
def single(name):
	global singleCounter ; singleCounter+=1
	tp_orig=getVar("tp")									# save off tprops
	v=getParam(getVar("tofit")[0])								# get initial value
	vr=predictUncert(addedNoise=.003,regen=True,threshold=threshold/100,nworkers=nw)	# run hypothetical contours
	setVar("tp",tp_orig)									# restore tprops
	p=np.round((vr[1]-v)/v*100,2) ; v=np.round(v,2) ; vr=[np.round(vr[0]),np.round(vr[1])]	# calculate percent error from bounds
	logPrint((name,v,vr,p,"%"))								# print to screen AND log to file
	shutil.move("tdtrcache","canIMeasureThat_cache/"+name+"-"+str(singleCounter))		# move tdtrcache (with fake files and contour results) to cache folder

def logPrint(text):
	f=open("canIMeasureThat_cache/log.txt",'a')
	if type(text)==tuple:
		text=" ".join([str(v) for v in text])
	f.write(text+"\n")
	print(text)

def checkTechnique(technique):
	technique=technique.split("+")		# eg "mfTDTR+SSTR"
	#allCombinations=[]
	#for t in technique:
	#	experiment="".join( [c for c in t if c in upperAlphabet ] ) # "mfTDTR" --> "TDTR"
	#	f=freqs[experiment] ; r=rads[experiment]
	#	options=[[experiment],f,r]				# [ [ TDTR ],[ N Frequencies ],[ M radii ] ] --> itertools.product --> list of combinations
	#	allCombinations=allCombinations+list(itertools.product(*options))
	#	print(len(allCombinations))
	#print(allCombinations)
	
	# say we have 16 different options for TDTR (4 frequencies, 4 radii), and 48 for SSTR (6 frequencies and 8 radii), we could *conceivably* end up doing simultaneous solving of up to 64 experiments (mrmfTDTR+mrmfSSTR would be solving all of them at once!). alternatively, we could be looping through up to 64 
	
	#for i in range(len(allComb

	#return	



	types=[] ; modifiers=[] ; frequencies=[] ; radii=[] #passed to ss2types and so on. each element is a multi-solve. some elements may be a list, in which case, we'll loop through

	for t in technique: # "mfTDTR"
		experiment="".join( [c for c in t if c in upperAlphabet ] ) # "mfTDTR" --> "TDTR"
		modifier=t.replace(experiment,"") # "mfTDTR" - "TDTR" = "mf"
		fs=freqs[experiment] ; rs=rads[experiment]
		if "mf" in modifier and "mr" in modifier: # mrmf superpower
			for f in fs:
				for r in rs:
					types.append(experiment) ; frequencies.append(f) ; radii.append(r) ; modifiers.append(modifier)
		elif "mf" in modifier:
			for f in fs:
				types.append(experiment) ; frequencies.append(f) ; radii.append(rs) ; modifiers.append(modifier)
		elif "mr" in modifier:
			for r in rs:
				types.append(experiment) ; frequencies.append(fs) ; radii.append(r) ; modifiers.append(modifier)
		else:
			types.append(experiment) ; frequencies.append(fs) ; radii.append(rs) ; modifiers.append(modifier)

	#print(types,modifiers,frequencies,radii) 
	# "mfTDTR+SSTR" --> types = ['TDTR', 'TDTR', 'TDTR', 'TDTR', 'SSTR'] (4 TDTR frequencies + 1 SSTR)
	# frequencies = [1000000.0, 2100000.0, 4000000.0, 8400000.0, ["6 SSTR frequencies"]] (4 TDTR frequencies to do simultaneously, then loop through the 6 SSTR
	# radii= [["4 TDTR radii"], [...], [...], [...], ["8 SSTR radii"]] (4 and 8 TDTR and SSTR radii to loop through
	loopable=[ type(e)==list for e in frequencies+radii] # [False, False, False, False, True, True, True, True, True, True]
	iters=[ e for e in frequencies+radii if type(e)==list ] # pick out the list-type elements from above, which we'll iterate through
	iterIs=[i for i,e in enumerate(frequencies+radii) if type(e)==list ] # [4, 5, 6, 7, 8, 9] and the index of those lists in frequencies+radii
	iterated=list(itertools.product(*iters)) # all combinations of those iterable lists
	nf=len(frequencies) ; nr=len(radii)
	for vals in iterated: # eg (100.0, 5e-06, 5e-06, 5e-06, 5e-06, 5e-07), corresponding to SSTR freq, TDTR 1 rad..., SSTR rad, indices from iterIs
		#print(vals)
		ts=types
		fs=[] ; rs=[]
		for n in range(nf):
			if loopable[n]: # frequencies has a list which we're iterating through, so check iterIs, and grab that entry from vals
				i=iterIs.index(n)
				fs.append(vals[i])
			else:		#singular value in frequencies, so use that
				fs.append(frequencies[n])
		for n in range(nr):
			if loopable[nf+n]:
				i=iterIs.index(nf+n)
				rs.append(vals[i])
			else:
				rs.append(radii[n])
		#finally, skip any same-technique different-radii if "mr" modifier is not used. or vice versa for "mf". eg, see below for a valid iter result outside "mr"
		# ['TDTR', 'TDTR', 'TDTR', 'TDTR', 'SSTR'] [1000000.0, 2100000.0, 4000000.0, 8400000.0, 100.0] [5e-06, 5e-06, 5e-06, 7.5e-06, 7.5e-06]
		keep=True
		for ti,fi,ri,m in zip(ts,fs,rs,modifiers):
			for tj,fj,rj in zip(ts,fs,rs):
				if ti==tj and "mf" not in m and fi!=fj:
					#print("not mf, but dif f, skip:",ti,tj,fi,fj)
					keep=False
				if ti==tj and "mr" not in m and ri!=rj:
					#print("not mr, but dif r, skip:",ti,tj,ri,rj)
					keep=False
		if keep:
			#print(ts,fs,rs)
			setVar("ss2rpus",rs) ; setVar("ss2rprs",rs) ; setVar("ss2Types",ts) ; setVar("ss2fms",fs)
			logPrint(" / ".join([str(e).replace(" ","") for e in zip(ts,fs,rs)]))
			name="+".join(technique)
			single(name)

if __name__=='__main__':
	main()

