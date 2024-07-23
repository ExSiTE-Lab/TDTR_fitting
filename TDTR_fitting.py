# -*- coding: utf-8 -*- #need this to accept unicode shenanigans below. TDTR_fitting.py is written by Thomas W. Pfeifer
# v0.163 (goes with 0.73)

#USE CASE 1: STANDALONE: edit the below thermal properties, measurement parameters, and fitting settings, and execute "python TDTR_fitting.py" to have the file specified read in, fitting performed, and the specified thermal properties returned. 
#USE CASE 2: EXTERNALLY CALLED: import this code into another python script via "from TDTR_fitting import *". Then, be sure to set the various thermal properties , measurement parameters, and fitting settings in your code (via the function "setVar(variableName,value)"), and call "solve" yourself. This allows you to process many files at once. 

# TODO: sparse contours, where it keeps running and adding resolution, "stop once you're satisfied"
# TODO: measureContour1Axis() supports RMXY via magicModifiers. is that good enough? (then can scrub measureContours and friends)
# TODO: should measureContours2 do contouring in the 3 axes? (currently contours in x-y plane only, but could just as easily do x-z and y-z)
# TODO: ss2() and solveSimultaneous() are duplicative. scrub solveSimultaneous and friends. also get rid of FD-TDTR from the gui
# TODO: only once ss2 results are published can we publish this code (or, scrub all mention of ss2 from gui *and* this. gross. nah)
# TODO: should each dataset type and associated functions be put into classes? probably. the parallel list sort of structure that ss2 uses to track datafiles and measurement types and so on is pretty abhorrent. classes just weren't all that necessary prior to simultaneous fitting, where we only ever dealt with one dataset and measurement type at a time...


# TODO: i'm a clown who apparently can't decide between the verbiage "show", "display", or "plot" for plotting things. this should be cleaned up
# TODO: should find some sanity on what things are passable vs what things are global variables (eg, layerNames is only used by sensitivity(), so shouldn't that be passable? tp is used all over, so global for that makes sense. i think there are other cases like layerNames though too)
# TODO warn() and conditionalPrint() made. warn should probably have level-specific suppression instead of a single quietWarnings flag

#THERMAL PROPERTIES
#     C       Kz/G     d         Kr
tp=[[ 2.42e6, 120.00,  85.49e-9, "Kz" ], # layer 1 (metal coating)
    [         1/4e9                ], # interface 1
    [ 2.64e6, 35,  1.,       "Kz" ]] # layer 2 (bottom layer)
#     J/m³/K  W/m⁽²⁾/K m         W/m/K

#MEASUREMENT PARAMETERS
Pow=1. #pump power measurement (literally only matters for Tz(). everywhere else, either ratio is used, or magnitudes/X/Y signals are normalized)
#(the following do nothing if running under standalone mode, as actual pump and probe radii and modulation frequency are read from file. these serve as the defaults if externally called however)
rprobe=(10.0+10.0)/4.*1e-6 
rpump=(20.0+20.0)/4.*1e-6 
fp=80e6 #pulse frequency, in Hz (note, this is NOT read from standard data files, but is unlikely to change)
fm=8.4e6 #moduluation frequency, in Hz
minimum_fitting_time=200e-12 ; minimum_fitting_frequency=1e3
maximum_fitting_time=None #60e-12
time_normalize="3000e-12"
gamma=1 # power absorbed = P/gamma
useTBR=True


#FITTING SETTINGS
tofit=['Kz2',"R1"]
filesToRead=["testscripts/DATA/2019_12_13_HQ/12132019_Al2O3_Cal_1_HQ_113202_8400000.txt"]
fitting="R"

#You may also set the upper and lower bounds for fitting variables here. We define them based on type, rather than dependant on guess values (guess/20 - guess*20 might not even be okay for K, but it is quite excessive for C, for example). beware: we'll crash if your guess is outside the bounds. 
LBUB = { "C":(.1e6,10e6)    , "Kz":(.01,1500000)        , "Kr":(.01,1500000)           , "G":(1e3,5e9)  , "R":(1/750e6,1/5e6) , 
         "d":(5e-9,1)       , "rpump":(1e-6,30e-6)      , "rprobe":(1e-6,30e-6)         , "gamma":(0,1e12) , "tshift":(-.1,.1)   ,
         "chopwidth":(0,25) , "yshiftPWA":(-1e-2,1e-2), "slopedPhaseOffset":(-2*3.14159,2*3.14159), "variablePhaseOffset":(0,0)    , "alpha":(0,1),
         "expA":(-10000,10000)   , "expB":(-200,0)          , "expC":(0,10000) , "anisotropy":(0.00000001,10000000) , "dutyCycle":(0,100) }
lbs={ k:LBUB[k][0] for k in LBUB.keys() }
ubs={ k:LBUB[k][1] for k in LBUB.keys() }


#lbs={ "C":.1e6 , "Kz":.01   , "Kr":.01   , "G":1e3   , "R":1/750e6 , 
#"d":5e-9 , "rpu":1e-6  , "rpr":1e-6  , "gamma":0   ,"tshift":-.1,
#"chopwidth":0,"yshiftPWA":-1e-2,"sphase":-2*3.1416,"phase":0,"alpha":0,
#"expA",1}
#ubs={ "C": 10e6 , "Kz":15000 , "Kr":15000 , "G":750e6 , "R":1/5e6   ,  "d":1.   , "rpu":30e-6 , "rpr":30e-6 , "gamma":1e12,"tshift":.1,"chopwidth":25,"yshiftPWA":1e-2,"sphase":2*3.1416,"phase":0,"alpha":1}

#WHAT'S THE SCOOP ON THE MATH? READ HERE:
# Jiang's, Qian's, and Yanga's "Tutorial, Time-domain thermoreflectance (TDTR) for thermal property characterization of bulk and thin film materials"
# Schmidt's, Cheaito's, and Chiesa's "A frequency-domain thermoreflectance method for the characterization of thermal properties"
# Braun's, Olson's, Gaskins' and Hopkins' "A steady-state thermoreflectance method to measure thermal conductivity"
# Carslaw & Jaeger's "Conduction of Heat in Solids", Section 3.7, page 110
# Vᵢₙ(t𝘥)  = Re(Z(ω)) , Vₒᵤₜ(t𝘥) = Im(Z(ω))							#Jiang eq 2.21/2.22 (modified) 
# Z(ω)= Σ ΔT(ωₘ+n*ωₚ)*exp(i*n*ωₚ*t𝘥 ) (pulsed) or Z(ω)=ΔT(ωₘ) (CW)				#Jiang eq 2.19  / Schmidt eq 2/10
#	ΔT(ω)=A₁ ∫ Ĝ(k,ω)*exp(-π²*k²*w₀²)*2*π*k*dk ; from 0 to ∞				#Jiang eq 2.18
#	ΔH(ω)=A₁/2*π ∫ k*Ĝ(k,ω)*exp(-k²*(rᵣ²+rᵤ²)/8)*dk ; from 0 to ∞				#Schmidt eq 8, equivalent to ΔT(ω)
#	L(r,ω)=1/2*π ∫ k*Ĝ(k,ω)*exp(-k²*(rᵣ²+rᵤ²)/8)*J₀(k*r)*dk					#Braun eq A2, equivalent to ΔT(ω), J₀ is bessel func
#			(ps, use scipy.special.jv for bessel, it's very fast)
#		A₁ is pump power average, w₀=√(½(rᵣ²+rᵤ²)), rᵣ rᵤ are probe and pump powers	#Jiang eq 2.10+, Jiang eq 2.18+
#		Ĝ(k,ω)=-D/C 									#Jiang eq 2.9 / Schmidt eq 7
#			[N]ₙ[M]ₙ...[R]₁[N]₁[M]₁=[[A B][C D]] where 1 is upper, n is lower	#Jiang eq 2.8
#				[N] = | 1  1 | | exp(λ*L)    0     |				#Jiang eq 2.5 ( to prevent explosion, divide each 
#				      |-γᵢ γᵢ| |    0    exp(-λ*L) |				                element by exp(λ*L) if L is large )
#				[M] =_1_  | γᵢ -1 |						#Jiang eq 2.6
#				     2*γᵢ | γᵢ  1 |
#				[R] = | 1 -1/G |						#Jiang eq 2.7
#				      | 0   1  |
#					γ=K𝘻*λ							#Jiang eq 2.4+
#					λ²=4*π²*k²*η+i*ω*C/K𝘻					#Jiang eq 2.2+
#					η=K𝘳/K𝘻							#Jiang eq 2.1+
#		or	[M]ₙ...[M]₁=[[A B][C D]] where 1 is upper, n is lower			#Schmidt eq 6
#				[M] = |    cosh(q*L)   -1/(K𝘻*q)*sinh(q*L) |			#Schmidt eq 3 ( note: to prevent explosion, divide each
#				      | -K𝘻*q*sinh(q*L)     cosh(q*L)      |			                element by cosh(q*L). sinh/cosh=tanh )
#					q²=(K𝘳*k²+C*i*ω)/K𝘻					#Schmidt eq 4
#Whether using Jiang's or Schmidt's, overflows should be expected in exp(λ*L), cosh, and sinh (at least for your semi-infinite final layer). Instead, divide through by exp(λ*L) or cosh (sinh/cosh becomes tanh). This is safe for traditional FDTR/FDTR since the final result we're after is -D/C, and the end result is the same. Beware however: if you want the backside temp of a layer, we replace -D/C above with -A𝘴𝘶𝘣*D𝘧𝘶𝘭𝘭/C𝘧𝘶𝘭𝘭+B𝘴𝘶𝘣. in this case, tanh may still be safe if L is very very small (cosh≈1), but can result in incorrect values for A𝘴𝘶𝘣 and B𝘴𝘶𝘣. division by exp(λ*L) is never safe in this case.

# CODE STRUCTURE:
# solve  > [ solveTDTR , solveFDTR , solvePWA , solveSSTR ]
#	readFile > [ readTDTR , readFDTR , readPWA , readSSTR ]
#	curve_fit or least_squares
#		func > [ TDTRfunc , FDTRfunc , PWAfunc , SSTRfunc ]
#	 		setTofitVals > setParam (sets each fitted parameter with values passed)
#			popGlos
#			delTomega > Gkomega
#	resultsPlotter > func()
# Tz > choptp > biMatrix (replaces Gkomega)
# comprehensiveUncertainty
#	solve(), record stdev
# 	perturbUncertainty > perturb each param and then solve()
# 	generateHeatmap > generate 2D list of fitted parameters > func(), record error() for each
# 	measureContour1Axis > generate 1D list of single fitted parameter > solve(), record residual for each
# sensitivity > perturb each param and then func() and subtract


#GENERAL STUFF
import numpy as np
import sys,math,time,copy,os,re,glob,shutil,scipy.io,traceback
try: # if the folder exists, we use that code, if not, we fall back to niceplot
	#sys.path.insert(1,"../niceplot") # found within this same directory. see testing58
	sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../niceplot/') )
except: # for proof that this works (and that we use the other niceplot direc first)
	pass
#from plotter import *
pi=np.pi
verbose=[] ; quietWarnings=False
autorpu=True ; autorpr=True ; autofm=True ; autoFailed=False
if os.name=='posix':
	from multiprocessing import set_start_method
	try:
		set_start_method("fork")
	except:
		pass
# linux default is fork, mac/win default is spawn. we actually want fork here (copies everything before running, rather than all subprocesses in pool access save globals) https://pythonspeed.com/articles/python-multiprocessing/

### MATH STUFF ###
#Ĝ(k,ω)=-D/C #Jiang eq 2.9 / Schmidt eq 7
#Consider a function for Ĝ(k,ω) that takes a single k and a single ω; this is quite slow. (and you can find this in v0.3). instead, accept lists of k values, and lists of ω values, and return a 2D array structured as Ĝ[nth k][nth ω]. refer to v0.5 if you want a verbose version of this code. 
def Gkomega(ks="",omegas="",partial=False):
	global Kzs,Krs,Cs,ds,Gs,Rs
	omegas=np.asarray(omegas) # WATCH OUT! if "omegas" is a list instead of a numpy array, ω[ω==0]=newval doesn't do what you think! it just sets the 
	omegas[omegas==0]=1e-6	# 0th element to newval. try it yourself. a=[1,2,0,4] ; a[a==0]=7 --> [7,2,0,4]. "a==0" is False, a[False] is 0th element
	lenk=len(ks);leno=len(omegas)
	ks=np.outer(ks,np.ones(leno)) #creates 2D array, k[nth k,nth ω]
	omegas=np.outer(np.ones(lenk),omegas) #creates 2D array, ω[nth k,nth ω]
	ones=1.*np.ones((lenk,leno));zeros=0.*np.ones((lenk,leno))
	if partial:
		ct=0
		while max(ds)>1e-6: # partial code below, with cosh and sinh instead of tanh, will have problems with thick slices with low thermal 
			i=np.where(np.asarray(ds)>1e-6)[0][0]	# conductivity, eg if you're calling T(r,z) for a deep volume. To handle that, we'll
			#print(ct,"WAS:",Kzs,Krs,Cs,ds,Gs)	# chop any layers bigger than 1um, cloning the thermal properties, zero TBR, and a 
			Kzs.insert(i,Kzs[i]) ; Krs.insert(i,Krs[i]) ; Cs.insert(i,Cs[i]) ; Gs.insert(i,np.inf) ; Rs.insert(i,0) # new layer thickness updated
			ds.insert(i,.9e-6) ; ds[i+1]-=.9e-6
			#print(ct,"NOW:",Kzs,Krs,Cs,ds,Gs)
			ct=ct+1
	lenl=len(Kzs);leni=len(Gs)
	#and start working our way through Jiang/Schmidt et al
	Ml=[] ; Mi=[]
	for i in range(lenl):
		Kz=Kzs[i];Kr=Krs[i];C=Cs[i];d=ds[i]
		q=np.sqrt((Kr*ks**2.+C*1j*omegas)/Kz) #q²=(K𝘳*k²+C*i*ω)/K𝘻 #Schmidt eq 4
		if partial: #certain sets of parameters (eg, small Kzs (.01) or large ds (1um)) will cause overflows in cosh and sinh. instead, divide through by exploding cosh and use tanh. 𝑢𝑛𝑙𝑒𝑠𝑠 you're using this for T(r,z) where ΔT𝘣𝘢𝘤𝘬𝘴𝘪𝘥𝘦(ω) requires each element of this matrix rather than simply the ratio of -D/C. note: explosions are most commonly found doing contours with semi-silly ranges. this will merely produce a nan in the output file, which will be treated as residual=1 upon import, so not a huge deal. (fyi, use "np.seterr(all='raise', under='ignore')", then "try:" and "except:" to debug this stuff. beware cases also exist where cosh,sinh,tanh are unexploded, but there is an explosion inside TPmatmul, so check there too).
			M=[[     np.cosh(q*d)      , -1./Kz/q*np.sinh(q*d) ], # |    cosh  -1/K/q*sinh | #Schmidt eq 3
			   [ -1.*Kz*q*np.sinh(q*d) ,      np.cosh(q*d)     ]] # | -K*q*sinh    cosh    |
		else:
			np.seterr(under='ignore')
			M=[[        ones           , -1./Kz/q*np.tanh(q*d) ], # |     1    -1/K/q*tanh | #Schmidt eq 3, ÷ cosh
			   [ -1.*Kz*q*np.tanh(q*d) ,           ones        ]] # | -K*q*tanh     1      |
		Ml.append(M)
	for i in range(leni):
		R=[[ ones , -Rs[i] ], # | 1 -1/G | #Jiang eq 2.7
		   [ zeros,  ones  ]] # | 0   1  | # (note: if G=∞ (zero thermal boundary resistance), you just have an identity matrix! )
		Mi.append(R)
	#finally, stack up those matrices
	ABCDs=[1]*(len(Ml)+len(Mi)) # start with an empty list of appropriate length, then "interlace"
	if lenl>leni: #Standard, lay1 intf1 lay2 intf2....layN, meaning len(Ms)=len(Rs)+1. so stack them like: M1 R1 M2 R2 M3 R3....MN
		ABCDs[::2]=Ml ; ABCDs[1::2]=Mi
	else: #atypical, but used for buried heating: Q_down sees standard, but Q_up sees just an interface first, meaning len(Ms)=len(Rs), so stack them like: R1 M1 R2 M2....MN
		ABCDs[::2]=Mi ; ABCDs[1::2]=Ml
	ABCD=np.identity(2) #we'll construct [M]ₙ...[R]₂[M]₂[R]₁[M]₁ as we go, tacking each new one on to the left
	for entry in ABCDs:
		ABCD=TPmatmul(entry,ABCD)
	return ABCD
#END: Ĝ(k,ω)=-D/C #Jiang eq 2.9 / Schmidt eq 7

#ΔT𝘴𝘶𝘳𝘧𝘢𝘤𝘦(ω)=A₁/2*π ∫ k*Ĝ(k,ω)*exp(-k²*(rᵣ²+rᵤ²)/8)*dk ; from 0 to ∞ #Schmidt eq 8 (analagous to Jiang's ΔT𝘴𝘶𝘳𝘧𝘢𝘤𝘦(ω)=A₁ ∫ Ĝ(k,ω)*exp(-π²*k²*w₀²)*2*π*k*dk)
from scipy.integrate import trapz,simps,romb,quad
from scipy.special import jv,erf
pumpShape="gaussian" ; probeShape="gaussian" ; xoff=10e-6
hybridFactors=[1] # TODO currently hybridFactors are used in the order: gaussian,tophat,ring,offset. ideally we'd follow whatever order is in pumpShape (e.g. pumpShape="ring+tophat" would reverse the order)
#@profile
def delTomega(omegas,gkomega="",radii="",integration="trapz"):
	ks=np.linspace(kmin,kmax,ksteps) #to integrate, we set up up a list of x values, pass them to f(x), and then integrate numerically by summing the area of each trapezoid. 2⁶+1=65, lab code uses 50 { [k,wk]=lgwt(50,0,10/sqrt(w0^2 + w1^2)); }
	if len(gkomega)==0: #passing in Ĝ(k,ω) is how we hijack the same code for T𝘴𝘶𝘳𝘧𝘢𝘤𝘦 (Ĝ(k,ω)=-D/C) and ΔT𝘣𝘢𝘤𝘬𝘴𝘪𝘥𝘦 (use -A𝘴𝘶𝘣*D𝘧𝘶𝘭𝘭/C𝘧𝘶𝘭𝘭+B𝘴𝘶𝘣 instead)
		if measureAt!=0 or depositAt!=0:
			#ks=np.linspace(kmin,kmax,ksteps)	# used for integrating ∫ stuff dk
			gkomega=biMatrix(omegas,ks)
		else:
			gkomega=Gkomega(ks,omegas) #returns 2D list, gkomega[nth k,nth ω]
			ABCD=gkomega
			gkomega=-1.*ABCD[1,1]/ABCD[1,0]
	r1={False:rpump,True:rprobe}[rpump=="rpr"] ; r2={False:rprobe,True:rpump}[rprobe=="rpu"] #if rpump set to "rpr", inherit probe radius for pump (and vice versa)
	# Hankel transform of a gaussian spot:
	# p(r)=2*A/π/rᵤ²*exp(-2*r²/rᵤ²) [Jiang 2.10] -> H(k)=2π∫f(r)*J(k*r*2*π)*k dr -> p(k)=A*exp(-π²k²rᵤ²/2) [Jiang 2.11]
	# p(r)=2*A/π/rᵤ²*exp(-2*r²/rᵤ²) [Jiang 2.10] -> H(k)=∫f(r)*J(k*r)*k dr -> p(k)=A/2/π*exp(-k²rᵤ²/8) [Schmidt 5]
	if probeShape=="gaussian" or r2==0:
		Hprobe=np.exp(-1.*ks**2.*(r2**2.)/8.)
	else:
		rs=np.linspace(0,r2*5,1000) ; pr=np.zeros(1000)
		if "tophat" in probeShape:
			pr[rs<=r2]=np.ones(len(rs[rs<=r2]))*1/(np.pi*r2**2) # tophat heating, 1 inside ro, 0 outside
		#if "gaussian" in probeShape:
		#	pr[:]=2/np.pi/r2**2*np.exp(-2*rs**2/r2**2)
		dr=rs[1]-rs[0] ; integ=np.sum(pr*rs)*dr*2*np.pi
		pr*=1/integ*2*np.pi
		Hprobe=np.sum( pr*jv(0, np.outer(ks,rs))*rs , axis=1)*dr # [ nth k, nth r], flattened in terms of r	
	#print("sum Hprobe",np.sum(Hprobe))
	if pumpShape=="gaussian":
		Hpump=1/2/np.pi*np.exp(-1.*ks**2.*(r1**2.)/8.) # 1D, [nth k].
	elif pumpShape=="ring":
		Hpump=1/2/np.pi*jv(0,r1*ks) # ring heating source, "infinitely thin", hankeled analytically
	else:
		rs=np.linspace(0,r1*5,1000) ; pu=np.zeros((len(pumpShape.split("+")),1000)) ; ct=0
		# ALTERNATIVE SPOT SHAPES: UNCOMMENT SOME OF THESE LINES FOR NUMERICALLY-SOLVED GAUSSIAN, TOPHAT, OR RING
		if "gaussian" in pumpShape:
			pu[ct,:]=2/np.pi/r1**2*np.exp(-2*rs**2/r1**2) ; ct+=1 # replicated gaussian pump, but Hankelling numerically
		if "tophat" in pumpShape:
			pu[ct,rs<=r1]=np.ones(len(rs[rs<=r1]))*1/(np.pi*r1**2) ; ct+=1 # tophat heating, 1 inside ro, 0 outside
		if "ring" in pumpShape:
			i=np.argmin(np.abs(rs-r1)) ; pu[ct,i]=1/(2*np.pi*r1)/(rs[i]-rs[i-1]) ; ct+=1 # ring heating source, "infinitely thin"
		elif "offset" in pumpShape:
			pu[ct,:]=np.exp(-1.*(rs-xoff)**2./(2.*r1**2.))
		pu=[ p*hf for p,hf in zip(pu,hybridFactors) ] ; pu=np.sum(pu,axis=0)

		dr=rs[1]-rs[0]
		integ=np.sum(pu*rs)*dr*2*np.pi
		#print("INTEGRAL PUMP:",integ) # V(r,θ)=∫ ∫ z(r,θ)*r dr dθ, 0 < r < ∞, 0 < θ < 2π
		pu*=1/integ
		Hpump=np.sum( pu*jv(0, np.outer(ks,rs))*rs , axis=1)*dr # [ nth k, nth r], flattened in terms of r	
		#plot([rs],[pu],xlabel="radius (m)",ylabel="pump intensity (-)") ; return
	# integrand of ΔT(ω) = A₁/2*π ∫ [ k*Ĝ(k,ω)*Hₚᵤ*Hₚᵣ ] dk, aka, Schmidt's H(ω)=A₁/2*π ∫ [ k*Ĝ(k,ω)*exp(-k²*(rᵣ²+rᵤ²)/8) ] dk.
	integrand=ks[:,None]*gkomega*Hpump[:,None]*Hprobe[:,None] # [ k, ω ] Any var without a given dimension uses "None" to expand to the proper shape
	if len(radii)!=0:
		integrand=integrand[:,:,None]*jv(0,np.outer(ks,radii))[:,None,:] # [ k, ω, r]. if list of radii is passed, we add a dimension on return
	# ΔT(ω) = ∫ k*Ĝ(k,ω)*Hₚᵤ*Hₚᵣ dk or ΔT(ω,r) = ∫ k*Ĝ(k,ω)*Hₚᵤ*J(k*r) dk
	integrandr=integrand.real #separate real and imaginary parts so we can integrate separately
	integrandi=integrand.imag
	delTr=A1*integrate(integrandr, dx=ks[1]-ks[0],axis=0,itype=integration) #and integrate using scipy's trapz. you might also use simps/romb (romb needs 2ⁿ+1 steps)
	delTi=A1*integrate(integrandi, dx=ks[1]-ks[0],axis=0,itype=integration)*1j
	return delTr+delTi # [ ω, (r) ]
#END: #ΔT𝘴𝘶𝘳𝘧𝘢𝘤𝘦(ω)=A₁/2*π ∫ k*Ĝ(k,ω)*exp(-k²*(rᵣ²+rᵤ²)/8)*dk ; from 0 to ∞ #Schmidt eq 8

depositAt=0 ; measureAt=0 ; alpha=0 # BIDIRECTIONAL: PROBING AND HEATING LOCATION NOT NECESSARILY SAME AS MEASUREMENT LOCATION
# The following is always true for a given layer:			
# | T𝑏𝑎𝑐𝑘𝑠𝑖𝑑𝑒 | = | A B | | T𝑠𝑢𝑟𝑓𝑎𝑐𝑒 | per Jiang eq 2.8	-->	In normal TDTR, we can use an adiabatic backside boundary condition: Q𝑏𝑎𝑐𝑘𝑠𝑖𝑑𝑒=0
# | Q𝑏𝑎𝑐𝑘𝑠𝑖𝑑𝑒 |   | C D | | Q𝑠𝑢𝑟𝑓𝑎𝑐𝑒 | or / Schmidt eq 6	       which yields T𝑠𝑢𝑟𝑓𝑎𝑐𝑒=-D/C*Q𝑠𝑢𝑟𝑓𝑎𝑐𝑒  via the second equation left bottom.
# 	or: 
# T𝑏𝑎𝑐𝑘𝑠𝑖𝑑𝑒=A*T𝑠𝑢𝑟𝑓𝑎𝑐𝑒+B*Q𝑠𝑢𝑟𝑓𝑎𝑐𝑒 , Q𝑏𝑎𝑐𝑘𝑠𝑖𝑑𝑒=C*T𝑠𝑢𝑟𝑓𝑎𝑐𝑒+D*Q𝑠𝑢𝑟𝑓𝑎𝑐𝑒	              Next, consider the goal of measuring the backside temperature of a subslice. We can still
#							    find T𝑠𝑢𝑟𝑓𝑎𝑐𝑒 in the traditional manner: T𝑠𝑢𝑟𝑓𝑎𝑐𝑒=-Dᵢⱼ/Cᵢⱼ*Q𝑠𝑢𝑟𝑓𝑎𝑐𝑒
# |  i  |  j  | 					   and then plug that in to the T𝑏𝑎𝑐𝑘𝑠𝑖𝑑𝑒 equation:
# |     |      semi-infinite BC				  T𝑏𝑎𝑐𝑘𝑠𝑖𝑑𝑒=(-Aᵢ*Dᵢⱼ/Cᵢⱼ+Bᵢ)*Q𝑠𝑢𝑟𝑓𝑎𝑐𝑒
# |      measure here
#  deposit heat here					Finally, add an additional layer, with bidirectional heat flow. Consider heat flow both
#						      to the left and the right, Q𝑙𝑒𝑓𝑡+Q𝑟𝑖𝑔ℎ𝑡=Q𝑑𝑒𝑝𝑜𝑠𝑖𝑡𝑒𝑑. The classic T𝑠(Q𝑠) expression still
# |  i  |  j  |  k  |				    holds true for subslice i: Tᵢⱼ=-Dᵢ/Cᵢ*Q𝑙𝑒𝑓𝑡, and similarly, Tᵢⱼ=-Dⱼₖ/Cⱼₖ*Q𝑟𝑖𝑔ℎ𝑡. We now have 
# |     |     |      semi-infinite BC		  a system of equations, and can find: Tᵢⱼ=-(Dᵢ*Dⱼₖ)/(Dᵢ*Cⱼₖ+Cᵢ*Dⱼₖ)*Q𝑑𝑒𝑝𝑜𝑠𝑖𝑡𝑒𝑑.
# |     |      measure here			Recall this was the first step for finding T𝑏𝑎𝑐𝑘𝑠𝑖𝑑𝑒, so we can also incorporate:
# |      deposit heat here		      Tⱼₖ=(-Aⱼ*Dⱼₖ/Cⱼₖ+Bⱼ)*Q𝑟𝑖𝑔ℎ𝑡 or in terms of Tᵢⱼ: Tⱼₖ=(Aⱼ-Bⱼ*Cⱼₖ/Dⱼₖ)*Tᵢⱼ.
#  semi-infinite BC			    After some arduous algebra, we can find 
#					 ΔTⱼₖ = (Aⱼ - Bⱼ*Cⱼₖ/Dⱼₖ) * (-Dᵢ*Dⱼₖ ) / ( Dᵢ*Cⱼₖ + Cᵢ*Dⱼₖ ) * Q𝘥𝘦𝘱𝘰𝘴𝘪𝘵𝘦𝘥. 
# 				      And recall thin Hankel space, so we''l plug (Aⱼ-Bⱼ*Cⱼₖ/Dⱼₖ)*(-Dᵢ*Dⱼₖ)/(Dᵢ*Cⱼₖ+Cᵢ*Dⱼₖ) for Ĝ(k,ω) in 
#				  ΔT(ω)=A₁/2*π ∫ k*Ĝ(k,ω)*exp(-k²*(rᵣ²+rᵤ²)/8)*dk. Finally, considering the case where points ij and jk are switched,
#			      the math above holds true, except that matrix j is inverted (as heat flows in the opposite direction). alternatively,
#			  the layers can simply be reversed prior to computing the matrix, and matrices i and k are also switched with each other.
# biMatrix returns (Aⱼ-Bⱼ*Cⱼₖ/Dⱼₖ)*(-Dᵢ*Dⱼₖ)/(Dᵢ*Cⱼₖ+Cᵢ*Dⱼₖ), where points ij and jk are where heat is deposited and measured respectively. This is plugged straight in as Ĝ(k,ω) in ΔT(ω)=A₁/2*π ∫ k*Ĝ(k,ω)*exp(-k²*(rᵣ²+rᵤ²)/8)*dk for bidirectional or da,ma!=0
def biMatrix(omegas,ks): #,seenDepths=[]	see testing29 for more rigourous algebraic derivations
	global tp
	tpOrig=copy.deepcopy(tp)
	#chop tp into sub-matrices
	tp_i=list(reversed( choptp(tpOrig,0,min(depositAt,measureAt)) ))	#|<--i--[H]--j--[M]--k--...|
	tp_j=choptp( tpOrig,min(depositAt,measureAt),max(depositAt,measureAt) )	#|--i--[H]--j-->[M]--k--...|
	if depositAt>measureAt:							#|--i--[M]<--j--[H]--k--...|
		tp_j=list(reversed(tp_j))
	tp_k=choptp(tpOrig,max(depositAt,measureAt),20)
	
	conditionalPrint("biMatrix","da="+str(depositAt)+", ma="+str(measureAt)+", splitting tp into:\n"
		+"tp_i"+str(tp_i)
		+"tp_j"+str(tp_j)
		+"tp_k"+str(tp_k))
	#compute ABCD matrix for each
	# I
	if depositAt==0:
		ABCD_i=np.identity(2)
	else:
		tp=tp_i
		popGlos()
		ABCD_i=Gkomega(ks,omegas)
	#J
	if depositAt==measureAt:
		ABCD_j=np.identity(2)
	else:
		tp=tp_j
		popGlos()
		ABCD_j=Gkomega(ks,omegas,partial=True) #J needs to be partial (note division of terms for all the rest: (Aⱼ - Bⱼ*Cⱼₖ/Dⱼₖ) * (-Dᵢ*Dⱼₖ ) / ( Dᵢ*Cⱼₖ + Cᵢ*Dⱼₖ ) )
	#K
	tp=tp_k
	popGlos()
	ABCD_k=Gkomega(ks,omegas)

	if depositAt>measureAt: #|--i--[M]<--j--[H]--k--...|  instead of  |--i--[H]--j-->[M]--k--...| : simply exchange i and k and use the same math!
		ABCD_i,ABCD_k=ABCD_k,ABCD_i
	ABCD_jk=TPmatmul(ABCD_k,ABCD_j)
	# (Aⱼ - Bⱼ*Cⱼₖ/Dⱼₖ) * (-Dᵢ*Dⱼₖ ) / ( Dᵢ*Cⱼₖ + Cᵢ*Dⱼₖ )
	Aj=ABCD_j[0,0] ; Bj=ABCD_j[0,1] ; Cjk=ABCD_jk[1,0] ;  Djk=ABCD_jk[1,1] ; Ci=ABCD_i[1,0] ; Di=ABCD_i[1,1]
	gkomega=(Aj-Bj*Cjk/Djk)*(-Di*Djk)/(Di*Cjk + Ci*Djk)
	conditionalPrint("biMatrix", "gkomega "+{True:"has",False:"does not have"}[np.isnan(gkomega).any()]+" nans")
	#if np.isnan(gkomega).any():
	#	print(np.where(np.isnan(gkomega)))
	tp=copy.deepcopy(tpOrig)

	return gkomega

def choptp(tp,fromDepth,toDepth):
	fromDepth,depth=min(fromDepth,toDepth),max(fromDepth,toDepth) #if you feed us in the wrong order, we'll correct it for you (beware, if you actually mean from deep to shallow, you'll need to reversed() it yourself
	if fromDepth==toDepth: # properties literally don't matter, it's a layer with zero thickness
		return [[C_Sapph,K_Sapph,0,"Kz"]]
	ds=getCol(2) ; cumdepth=np.cumsum(ds) #layers 0 to 10 (d=10), 10 to 30 (d=20), 30 to 100 (d=70), 100 to inf (d=inf), cumdepth gives 10,30,100,inf
	fromLayer=np.where(cumdepth>fromDepth)[0][0] #layers 0 to 10, 10 to 30, 30 to 100, 100 to inf. want to inspect d=1? cumsum is 10,30,100,inf, we're *inside* the 0th. convention: if we say deposit heat AT an interface, it technically goes in layer below (fromdepth=d_interface), so if d=10, well, now we want to 1nth layer. 
	fromID=fromLayer*2 # layer number (0,1,2,3...) to index in tprops (layers on 0,2,4,6...)
	if toDepth>cumdepth[-1]:
		toDepth=cumdepth[-1]
	toLayer=np.where(cumdepth>=toDepth)[0][0] #layers 0 to 10, 10 to 30, 30 to 100, 100 to inf. want to inspect d=1? cumsum is 10,30,100,inf, we're *inside* the 0th. this time, d=10, there's no point in keeping the extra zero-thickness layer that we'll end up with if we reject cumdepth=todepth.
	toID=toLayer*2
	newtp=copy.deepcopy(tp[fromID:toID+1])
	th_0=cumdepth[fromLayer]-fromDepth #new thickness: depth of the original interface to our right, minus how deep we actually are
	th_N=toDepth-(cumdepth[toLayer]-ds[toLayer]) #new thickness: how deep we are, minus the depth of original interface to our left (gotcha: if toLayer is 0, we don't want cumdepth[-1], ie, total
	#print(th_0,th_N,fromDepth,toDepth)
	if len(newtp)>1:
		newtp[0][2]=th_0
		newtp[-1][2]=th_N
	else: #in the case where we end up with only one layer, th_0 is "distance from right interface" whereas th_N is "distance from left interface", ie, we need overlap!
		newtp[0][2]=th_0+th_N-newtp[0][2]
	#print("newtp",newtp)
	return newtp

#Z(t𝘥)= Σ ΔT(ωₘ+n*ωₚ)*exp(i*n*ωₚ*t𝘥) #from -∞ to ∞, Jiang eq 2.21/2.22 (modified) / Schmidt eq 2. 
#def Z(tds): #takes a list of time delay values
#	sumbits=(delTplus)*np.exp(1j*omegaP*np.outer(tds,ns))*convergeAccelerator #[which time delay,which n]
#	return np.sum(sumbits,axis=1)
#Note that while Jiang states Vᵢₙ(t𝘥)=½Σ(ΔT(ωₘ+n*ωₚ)+ΔT(-ωₘ+n*ωₚ))*exp(i*n*ωₚ*t𝘥) and Vₒᵤₜ(t𝘥)=-i*½Σ(ΔT(ωₘ+n*ωₚ)-ΔT(-ωₘ+n*ωₚ))*exp(i*n*ωₚ*t𝘥), simply taking the real and imaginary parts of ΣΔT(ωₘ+n*ωₚ)*exp(i*n*ωₚ*t𝘥) yields the same result. 
#END: Z(t𝘥)= Σ ΔT(ωₘ+n*ωₚ)*exp(i*n*ωₚ*t𝘥) #from -∞ to ∞, Jiang eq 2.21/2.22 (modified) / Schmidt eq 2.
insertInterfaceDepths=True
# MAJOR CHANGE WITH VERSION 132: WE NOW ALLOW PASSING A LIST OF OMEGAS, AND RETURNING A LIST OF T(z,r) FOR EACH OMEGA. WHY? THIS ALLOWS EASY PASSAGE OF BUNCHES OF FREQUENCIES, E.G. FOR SQUARE WAVE HEATING, WHERE THE SQUARE WAVE IS DONE VIA A SUM OF SINES
#returns EITHER a nD x nR matrix of temperatures for a T(r,z) profile, OR, given a single radius and depth, return the individual temperature there.
def Tz(mindepth=0,maxdepth=1500e-9,dsteps=50,rsteps=1,td="CW",maxradius=0,full=False,r=-1,d=-1,gif=False,omegas=""): #see testing29 for, testing, examples, and more rigourous algebraic derivations
	global rprobe,depositAt,measureAt,minimum_fitting_time
	if depositAt!=0 or measureAt!=0:	# suppose we're doing bidirectional analysis (material/Al/SiO2, backside pump/probe). there's a HIGH 
		dz=maxdepth-mindepth		# CHANCE we don't want to gen T(r,z) in the surface 2μm or so, when we deposited heat 1mm down
		mindepth=depositAt-dz/2
		maxdepth=depositAt+dz/2
	#copy off old settings
	rpr_old=rprobe ; depAt_old=depositAt ; measAt_old=measureAt ; tmin_old=minimum_fitting_time
	rprobe=0 ; popGlos()
	conditionalPrint("Tz","preparing to run with the following parameters:",pp=True)
	ks=np.linspace(kmin,kmax,ksteps)
	if len(omegas)==0:
		if td=="CW":
			omegas=np.asarray([omegaM]) #no sidebands required for CW
		else:
			minimum_fitting_time=td #min_t controls number of sidebands summed over, nmin/nmax, popGlos() populates them
			popGlos()
			ns=np.arange(nmin,nmax+1)
			convergeAccelerator=np.exp(-pi*ns**2./nmax**2.) #same stuff as in delTomega(), but we do it here so we can compute Ĝ(k,ω) once per layer, regardless of how many radii
			omegas=omegaM+ns*omegaP #1D list of ω=ωₘ+n*ωₚ values to pass into ΔT(ω)
	if maxradius==0:
		maxradius=rpump
	if isinstance(d,(list,np.ndarray)) and isinstance(r,(list,np.ndarray)):
		depths=d ; radii=r ; dsteps=len(depths) ; rsteps=len(radii)
	elif r>=0 and d>=0:
		if np.amin(np.absolute(d-np.cumsum(getCol(2))))<1e-11: #if passed depth is on (or ultra-close to) an interface, bump it upwards (shallower)
			d+=1e-10
		depths=[d] ; radii=[r] ; dsteps=1 ; rsteps=1
	else:
		depths=np.linspace(mindepth,maxdepth,dsteps) ; radii=np.linspace(0,maxradius,rsteps)
		#update depths with points just-above and just-below each interface within bounds
		if insertInterfaceDepths:
			depths=list(depths)
			for d in np.cumsum(getCol(2)):
				if d<mindepth or d>maxdepth:
					continue
				depths.append(d-1e-10) ; depths.append(d+1e-10)
			depths=sorted(depths) ; depths=np.asarray(depths) ; dsteps=len(depths)
	conditionalPrint("Tz","depths:"+str(depths)+"\nradii:"+str(radii))
	#and start processing all depths and radii
	Ts=np.zeros((len(omegas),dsteps,rsteps),dtype=complex)
	for d in range(dsteps):
		measureAt=depths[d]
	#	gkomega=biMatrix(omegas,ks) # ΔT(ω) = A₁/2*π ∫ k* [ Ĝ(k,ω) ] *Hₚᵤ*Hₚᵣ dk. this is -D/C for ΔT𝘴𝘶𝘳𝘧𝘢𝘤𝘦(ω), or (Aⱼ-Bⱼ*Cⱼₖ/Dⱼₖ)*(-Dᵢ*Dⱼₖ)/(Dᵢ*Cⱼₖ+Cᵢ*Dⱼₖ) for point "j" within stack [i][j][k]. 
		dto=delTomega(omegas,radii=radii)#,gkomega=gkomega) # [ ω, r ], pass list of radii, get radii dimension in return
		#if len(factors)==0:
		#	Ts[d,:]=dto[0,:]
		#else:
		#	Ts[d,:]=np.sum(dto,axis=0)
		Ts[:,d,:]=dto[:,:]
		#if td=="CW":
		#	Ts[d,:]=dto[0,:] # [ d, r ] <-- [ ω, r ], single frequency
		#else:
		#	sumbits=dto[:,:]*np.exp(1j*omegaP*td*ns)[:,None]*convergeAccelerator[:,None] # [ ω or n, r ]
		#	T=np.sum(sumbits,axis=0)  # [ ω, r ]
		#	Ts[d,:]=T[:]
	#actual temp is magnitude of real and imag "signal", for steady state
	T=(Ts.real**2.+Ts.imag**2.)**.5
	#T=Ts.real
	#restore old settings
	rprobe=rpr_old ; depositAt=depAt_old ; measureAt=measAt_old ; minimum_fitting_time=tmin_old
	if full:
		return T,depths,radii,Ts #returning the full-on real+i*imag parts allows someone to, say, get instantaneous T(t) for the mod-CW heating. see KZF validation/why not heater modulation/testing.py
	return T,depths,radii # index ordering: T[d,r],depths[d],radii[r]
### END MATH STUFF ###

### MATH HELPERS ###
def TPmatmul(mat1,mat2): #replaces numpy.matmul, which does not accept lists as A-Dn
	A1=mat1[0][0];B1=mat1[0][1];C1=mat1[1][0];D1=mat1[1][1] #|A1 B1| |A2 B2| __ | r1c1 r1c2 | 
	A2=mat2[0][0];B2=mat2[0][1];C2=mat2[1][0];D2=mat2[1][1] #|C1 D1| |C2 D2| -- | r2c1 r2c2 |
	return np.asarray([[A1*A2+B1*C2,A1*B2+B1*D2],[C1*A2+D1*C2,C1*B2+D1*D2]])
	
def integrate(matrix,dx,axis,itype="trapz"): #given an N D matrix of y values, integrate along given axis
	if itype=="trapz":
		return trapz(matrix,dx=dx,axis=axis)
	if itype=="simps":
		return trapz(matrix,dx=dx,axis=axis)
	if itype=="romb":
		return trapz(matrix,dx=dx,axis=axis)

def RSQ(dataYs,funcYs): # R² value: https://en.wikipedia.org/wiki/Coefficient_of_determination
	meanYs=np.mean(dataYs)
	SST=sum( (dataYs-meanYs)**2 ) #SSₜₒₜ=Σ(yᵢ-ȳ)²
	SSR=sum( (dataYs-funcYs)**2 ) #SSᵣₑₛ=Σ(yᵢ-fᵢ)²
	return SSR / SST #R²=1-SSᵣₑₛ/SSₜₒₜ, closer to 1 is better (or, leave off the "1-", closer to 0 is better)

def MSE(dataYs,funcYs): # Mean Squared Error: https://en.wikipedia.org/wiki/Mean_squared_error
	return sum( (dataYs-funcYs)**2 ) / len(dataYs) #1/N*Σ(Yᵢ-Fᵢ)²

def RES(dataYs,funcYs): # √( Σ[(Fᵢ-Yᵢ)²] / Σ[Fᵢ²] ) : Eq 23 from Feser & Cahill, Rev. Sci. Instrum. 83, 104901 (2012)
	return np.sqrt( sum( (funcYs-dataYs)**2 ) / sum(funcYs**2) )

def RESF(dataYs,funcYs): # RES(), but with outliar filtering (points more than 3x standard deviations are ignored)
	s=np.std(funcYs-dataYs) ; m=np.mean(funcYs-dataYs) # why use mean? f=ones, d=zeros, std(f-d)=0, all points are >3s off
	mask=np.zeros(len(dataYs)) ; mask[abs(funcYs-dataYs-m)>s*3]=1
	conditionalPrint("RESF","std: "+str(s)+", tossed "+str(len(mask[mask==1]))+"/"+str(len(funcYs))+" datapoints")
	return RES(dataYs[mask!=1],funcYs[mask!=1])

def RESindv(dataYs,funcYs):
	return max( [ RES(np.asarray([y]),np.asarray([f])) for y,f in zip(dataYs,funcYs) ] )

def RMS(dataYs,funcYs): # √( Σ[(Fᵢ-Yᵢ)²] / n )
	return np.sqrt(MSE(dataYs,funcYs))

def NRMS(dataYs,funcYs): # √( Σ[(Fᵢ-Yᵢ)²] / n ) / Σ[Fᵢ]/n
	return RMS(dataYs,funcYs)/np.mean(funcYs)

def NMSE(dataYs,funcYs):
	return NRMS(dataYs,funcYs)**2

def STD(dataYs,funcYs): # traditional σ=√(Σ[(xₙ-x̄)²]/N) -> 1/N*Σ[((Fᵢ-Yᵢ)/Yᵢ)²] for two sets of data: Scott, Thermal Conductivity Manipulation Through...
	return np.sum( ((funcYs-dataYs)/dataYs)**2 ) / len(dataYs) # and Wang et al , Phys Rev B 88, 075310 (2013)

def SDY(dataYs,funcYs): # we propose this test to explore the non-absolute difference between data and curve: where MSE et al might conflate random noise (curve bisects noisy data) and a bogus trend (curve under- or over-shoots data), here we don't take the absolute value of each difference. 
	return sum((dataYs-funcYs)/funcYs)#/len(dataYs)
	# trouble is, f(x)=mean(data) yields a perfect SDY! not what we want. (half data above, half data below)
	# while "bisecting the data" *is* our goal, you might imagine the case where perturbing a parameter changes the slope of the data *about the midpoint of the line*, which means SDY will not change! we'd be creating an astonishingly-bad fit, but not catching it here. 

def SLO(dataYs,funcYs): # "slope check", does the slope of the data, and func, match?
	ts=np.linspace(0,1,len(dataYs))
	solvedParams, parm_cov = curve_fit(yemxpb, ts, dataYs)
	m_data=solvedParams[0]
	#dlin=yemxpb(ts,*solvedParams)
	solvedParams, parm_cov = curve_fit(yemxpb, ts, funcYs)
	m_func=solvedParams[0]
	#flin=yemxpb(ts,*solvedParams)
	#plot([ts,ts,ts,ts],[dataYs,funcYs,dlin,flin],markers=["ko","k-","g:","b:"])
	return abs(m_data-m_func)/abs(m_data) # "by what percent did the slope change"
	
def CUR(dataYs,funcYs):
	ts=np.logspace(0,1,len(dataYs))
	solvedParams, parm_cov = curve_fit(quadratic, ts, dataYs)
	a_data=solvedParams[0]
	dfit=quadratic(ts,*solvedParams)
	solvedParams, parm_cov = curve_fit(quadratic, ts, funcYs)
	a_func=solvedParams[0]
	ffit=quadratic(ts,*solvedParams)
	lplot([ts,ts,ts,ts], [dataYs,funcYs,dfit,ffit], markers=["ko","k-","g:","b:"], title=str(a_data)+","+str(a_func)+"->"+str(abs(a_data-a_func)/abs(a_data)))
	return abs(a_data-a_func)/abs(a_data) # "by what percent did the slope change"

def CUR2(dataYs,funcYs):
	ts=np.logspace(0,1,len(dataYs))
	dD=np.gradient(dataYs,ts) ; dF=np.gradient(funcYs,ts)
	ddD=np.gradient(dD,ts) ; ddF=np.gradient(dF,ts)
	cD=np.mean(ddD) ; cF=np.mean(ddF)
	#ts=np.linspace(0,1,len(dataYs))
	lplot([ts,ts], [ddD,ddF], title=str(cD)+","+str(cF)+"->"+str(abs(cD-cF)/abs(cF)))
	return abs(cD-cF)/abs(cF)

def CUR3(dataYs,funcYs):
	def expo(ts,A,tau,C):
		return A*np.exp(-ts/tau)+C
	ts=np.logspace(0,1,len(dataYs))
	solvedParams, parm_cov = curve_fit(expo, ts, dataYs)
	a_data=solvedParams[1]
	dfit=expo(ts,*solvedParams)
	solvedParams, parm_cov = curve_fit(expo, ts, funcYs)
	a_func=solvedParams[1]
	ffit=expo(ts,*solvedParams)
	#plot([ts,ts,ts,ts],[dataYs,funcYs,dfit,ffit],markers=["ko","k-","g:","b:"],title=str(a_data)+","+str(a_func)+"->"+str(abs(a_data-a_func)/abs(a_data)))
	return abs(a_data-a_func)/abs(a_data) # "by what percent did the slope change"


def error(dataYs,funcYs):
	return RESF(dataYs,funcYs)
	#return SDY(dataYs,funcYs)
	#return SLO(dataYs,funcYs)
	#return CUR(dataYs,funcYs)
	#a,b,c=CUR3(dataYs,funcYs)/5,SLO(dataYs,funcYs)/4,RES(dataYs,funcYs)
	#print("error:",a,b,c)
	#return max(b,c) # "What percent change in curvature is visible by eye, which we would reject a fit for?" "what percent change in slope is visible by eye, which we would reject a fit for?" "what absolute deviation is visible by eye, which we would reject a fit for?" and so on. we only use to find "all the fits" for our cals which we dislike (and the tightened range of cal values we'll accept). 

# Criteria 1: error metric should not scale with funcion and data: consider y=[3,4,5], f=[2,4,6] and y=[0.3,0.4,0.5], f=[0.2,0.4,0.6]
#if you need nonscalability (error metric scales as you scale dataset would be bad), avoid MSE, RSQ, RMS. 
#consider 
# RSQ: 2/2 vs .02/.02		# PASS
# MSE: 2/3 vs .02/3		# FAIL
# RES: 2/56 vs .02/.56		# PASS
# RMS: √2/√3 vs √2*.1/√3	# FAIL
# NRMS: √2/√3/4 vs √2/√3/4	# PASS
# NMSE: 1/24 vs 1/24		# PASS
# STD: (1/9+1/25)/3		# PASS
#
# Criteria 2: error metric should not explode if function or data is centered around zero: consider y=[-3,-2,2,3], f=[-2,-1,1,2]
# RSQ:				# PASS
# RES: 				# PASS
# NRMS: ∞			# FAIL		trouble is, a raw integral of a zero-centered non-zero function 
# NMSE: ∞			# FAIL			can be zero (below-axis cancels above-axis)
# STD: ultrasmall?		# i don't know
#
# Repeat criteria 2: try with y=[3,4,5]-n, f=[2,4,6]-n
#	RES	STD	RSQ
# -1	.24	.10	c			trouble is, any occurrence of zero (or ultralow) value in y causes an explosion
# -2	.32	.37	o
# -3	.43	∞	n
# -4	.50	nan	s
# -5	.43	∞	t
# -6	.32	.37	a n t
### END MATH HELPERS ###

### SETUP AND SOLVING ###
def popGlos(): #populate globals used in various places for solving, eg, ω in rad/s from f in Hz, list of C,Kz,d,Kr,G values, 
	conditionalPrint("popGlos",str(tp))
	global wo,omegaM,omegaP,nmin,nmax,Cs,Kzs,ds,Krs,Gs,Rs,kmax,kmin,ksteps,ks,A1
	A1=Pow/gamma
	r1={False:rpump,True:rprobe}[rpump=="rpr"] ; r2={False:rprobe,True:rpump}[rprobe=="rpu"] #if rpump set to "rpr", inherit probe radius for pump (and vice versa)
	wo=math.sqrt(.5*r1**2.+.5*r2**2.) #w₀=√(½(w₁²+w₂²)) #Jiang eq 2.18+
	omegaM=fm*2.*pi;omegaP=fp*2.*pi #ωₘ and ω𝘴  are modulation frequency and laser pulse frequency #Jiang eq 2.10+
	# for TDTR: while Jiang states a summing of n from -∞ to ∞ (Vᵢₙ(t𝘥) and Vₒᵤₜ(t𝘥), Jiang eqs 2.21 and 2.22), there is a point where additional ns add little value. Cahill's "Analysis of heat flow in layered..." uses a bounds of 10*τ/t (Cahill eq 21+), where τ is the inverse of the pulse frequency: τ=1/fₚ=2*π/ωₚ, yielding ±20*π/(ωₚ*t), where ωₚ is the laser pulse frequency. Meanwhile, Cheito's code uses bounds of ±4*π/(ωₚ*tₛ) { M=round(4*pi/(omega_s*min_t));n=-M:M }, where tₛ is the minimum time where a TDTR response fitting begins. Note, using ±4*π/(ωₚ*tₛ) yields ±0.2% wiggles over Cheito's code. ±2*4*π/(ωₚ*tₛ) matches Cheito's code (odd), and ±3*4*π/(ωₚ*tₛ) undercuts Cheito's wiggles by 0.15% (nearly no wiggles). Cahill "Analysis of heat flow in layered...", Eq 20+, exp(-π(f/fₘₐₓ)²) is used to accelerate convergence. read about the need for these "sidebands" here: https://en.wikipedia.org/wiki/Dirac_comb (our pulses are a dirac comb, represented as a series of sines; the number of sine waves. the number of sine waves required to accurately approximate the signal from the pulses is determined by their rate and how close to the pulses you care to be accurate. 
	nmax=int(2./(fp*minimum_fitting_time)*3);nmin=-1*nmax #here, we take Cheito's ±4*π/(ωₚ*tₛ), replace ωₚ=2*π*fₚ, and use ±2/(fₚ*tₛ), x3
	if nmax>200000: #unsure why, but excessive N's seems to yield whacky results (try T(r,z) at low time delays)
		nmax=200000
	#nmax*=20 ; nmin*=20
	kmax=10./math.sqrt({True:max(r1,xoff),False:r1}["offset" in pumpShape]**2.+r2**2.);kmin=0.0 #S's H(ω) is required if you're using S's layers! ditto with integration bounds.
	cutw=2;hi=11;hf=6
	if fm<10**-cutw:
		ksteps=2**hi+1
	elif fm>10**cutw:
		ksteps=2**hf+1
	else:
		e=np.log10(fm)
		n=int((hf-hi)/(2*cutw)*e+(hf+hi)/2)
		ksteps=2**n+1
#	update material properties: (set up numpy lists (allows easy elementwise math, like "C/Kz" for each layer all at once) for each property)
	if len(tp[0])==4: #typical: lay1 intf1 lay2...layN, layer props are in even (0 - N), interface props are in odd (1 - N-1)
		Cs=getCol(0) ;  Kzs=getCol(1) ; ds=getCol(2) ; Krs=getCol(3)
		if useTBR:
			Rs=getCol(0,"odds")# ; Rs=[ v*1e-9 for v in Rs ]
			Gs=[ 1/np.float64(v) for v in Rs ] # casting to np.float64 allows div by zero: https://stackoverflow.com/questions/62264277/get-infinity-when-dividing-by-zero liable to have zero R, but never zero G
		else:
			Gs=getCol(0,"odds") ; Rs=[ 1/v for v in Gs ]
	else: #We may also find partial stacks: G/lay/G/lay...
		Cs=getCol(0,"odds") ;  Kzs=getCol(1,"odds") ; ds=getCol(2,"odds") ; Krs=getCol(3,"odds")
		if useTBR:
			Rs=getCol(0) ; Gs=[ 1/v for v in Rs ]
		else:
			Gs=getCol(0) ; Rs=[ 1/v for v in Gs ]
	#print("Rs",Rs,"Gs",Gs)
	if "Kz" in list(Krs): #put "Kz" in your Krs column for anisotropic. here, it's just a list referencing another list; anisotropy always enforced!
		#Krs=Kzs # see below: we can be more clever about this: something like Krs[Kzs=="Kz"]=Kzs[Kzs=="Kz"], to allow mixing of iso and aniso
		Krs=[ Kz if Kr=="Kz" else Kr for Kz,Kr in zip(Kzs,Krs)] # Any place where Krs contains "Kz", we pull the value from Kzs
			
	conditionalPrint("popGlos","found: 0<k<"+str(kmax)+"/"+str(ksteps)+", "+str(nmin)+"<n<"+str(nmax),pp=True)

def getCol(c,evensOrOdds="evens"): #return a column from the thermal properties matrix, tp. TODO consider using list comprehension: [x[0] for x in l]
	i1={"evens":0,"odds":1}[evensOrOdds]
	everyother=np.asarray(tp[i1::2])
	if len(np.shape(everyother))<2: #happens when tp is "too short" to find this param, eg, single-layer and we ask for G
		return []
	column=list(everyother[:,c])
	conditionalPrint("getCol",str(column))
	column=[ v if v=="Kz" else float(v) for v in column] # for each value in the column that is NOT "Kz", cast it to float
	#if "Kz" not in column:
	#	column=list(map(float,column))
	return column

def readResultFile(datafile): # datafile is NOT the results file. the datafile has the measurement data. the results file name is infered from the datafile name. 
	#print(datafile)
	direc=datafile.split("/")[:-1]+[callingScript,datafile.split("/")[-1]]
	#print(direc)
	outFile="/".join(direc).replace(".txt",".out")
	if not os.path.exists(outFile): # no resultsfile, skip
		return None,None
	lines=open(outFile,'r').readlines()
	if len(lines)==0 or " ".join(tofit)+"\n" != lines[0] or len(tofit)==0: # results line empty, no fitted params now, or different fitted params
		return None,None
	#print(tofit,lines)
	r,e=lines[1].split(";") 	# "r=[1.23,45.6];e=[residual,[.12,.26]]"
	r = [ float(v) for v in r.split("=")[1].replace("[","").replace("]","").split(",") if len(v)>0 ]
	e = e.split("=")[1].replace("[","").replace("]","").split(",")
	e = [ float(e[0]), [ float(v) for v in e[1:] ] ]
	return r,e
def writeResultFile(datafile,r,e):
	conditionalPrint("writeResultFile",datafile)
	#datafile=datafile.rstrip("/")
	datafile=datafile.replace('\\\\','/').replace('\\','/')
	direc="/".join( datafile.split("/")[:-1]+[callingScript] )
	#if datafile[0]=="/":
	#	direc="/"+direc
	outFile=direc+"/"+datafile.split("/")[-1]

	os.makedirs(direc,exist_ok=True)
	conditionalPrint("writeResultFile",outFile)
	with open(outFile,'w') as f:
		#print(r,e)
		f.write(" ".join(tofit)+"\n")
		rstr="r=["+",".join([str(v) for v in r])+"];"
		estr="e=["+str(e[0])+",["+",".join([str(v) for v in e[1]])+"]]"
		f.write(rstr+estr)
		f.write("\n"+"tp="+str(tp)+",rpr="+str(rprobe)+",rpu="+str(rpump))

# SOLVING FUNCTIONS
# generic solve(), sensitivity(), perturbUncertainty(), measureContour1Axis() functions check variable "mode" to check which solve function to call (solveTDTR, solveSSTR), and resultsPlotter() (called by all solve functions) checks variable "mode" to decide how to plot the resilts. it's a bit sketchy though: e.g. user imports TDTR_fitting, calls solveSSTR manually, means solveSSTR needs to re-set "mode" so resultsPlotter can check it. easy to mess up for future solver functions (TODO: find some sanity here)
mode="TDTR"
def solve(fileToRead,plotting="show",refit=True):
	# First, check if this file has already been solved for
	#print(fileToRead,callingScript)
#	direc=fileToRead.split("/")[:-1]+[callingScript,fileToRead.split("/")[-1]]
#	#print(direc)
#	outFile="/".join(direc).replace(".txt",".out")
#	if not refit and os.path.exists(outFile): # do NOT refit, and has already been fitted:
#		lines=open(outFile,'r').readlines() #; print(lines)
#		if " ".join(tofit)+"\n" == lines[0]:
#			r,e=lines[1].split(";") 	# "r=[1.23,45.6];e=[residual,[.12,.26]]"
#			r = [ float(v) for v in r.split("=")[1].replace("[","").replace("]","").split(",") ]
#			e = e.split("=")[1].replace("[","").replace("]","").split(",")
#			e = [ float(e[0]), [ float(v) for v in e[1:] ] ]
#			return r,e
	r,e=readResultFile(fileToRead)
	if not refit and r is not None:
		conditionalPrint("solve","refit==False: returning results from file")
		#print("REFIT=False")
		return r,e

	solveFunc={"TDTR":solveTDTR,"SSTR":solveSSTR,"FDTR":solveFDTR,"PWA":solvePWA}[mode]
	#if mode=="TDTR" and type(fileToRead)==list: # TODO is this even necessary? we used to have it in perturbUncertainty(), but solveTDTR still has it...
	#	solveFunc=solveSimultaneous
	conditionalPrint("solve","calling "+str(solveFunc))
	try:
		r,e=solveFunc(fileToRead,plotting)
	except RuntimeError:
		print("SOLVE ENCOUNTERED AN ERROR")
		r=[np.nan]*len(tofit) ; e=[np.nan,r]
		#return r,e

	stack=traceback.format_stack()
	mc1as=[ ( "mc1aWorker" in l or "genContour2D" in l ) for l in stack ] # we do NOT want to save off the results (for later re-use) if solve() was just
	#print(stack)	# called by the fast contours code! 
	if True not in mc1as:
		writeResultFile(fileToRead,r,e)
	return r,e

def func(xs,*parameterValues,store=False,addNoise=False): # x axis points (time delays for TDTR, pump powers for SSTR, and so on). and a list of parameter values corresponding to tofit. func(*listVariable) notation pops list values out. var=["cat","dog"], func(*var) allows func() to hear func("cat","dog"). not passing anything for paremeterValues simply generates the decay function with the thermal property matrix as-is.
	f={"TDTR":TDTRfunc,"SSTR":SSTRfunc,"pSSTR":SSTRfunc,"FDTR":FDTRfunc,"PWA":PWAfunc,"FD-TDTR":TDTRfunc}[mode]
	return f(xs,*parameterValues,store=store,addNoise=addNoise)
lastRead="" ; lastData=[] # prevent pounding the shit out of the disk if we're re-reading the same file (e.g. flat3DContour)
def readFile(fileToRead,reread=False):
	global lastRead,lastData
	if not reread and fileToRead==lastRead:
		return lastData
	f={"TDTR":readTDTR,"SSTR":readSSTR,"FDTR":readTDTR,"PWA":readPWA,"FD-TDTR":readTDTR}[mode]
	lastRead=fileToRead ; lastData=f(fileToRead)
	return lastData
def fileAverager1(filesToRead,fileOut): # WARNING: IT IS THE USER'S RESPONSIBILITY TO ENSURE ALL FILES ARE THE SAME TYPE, COLLECTED UNDER THE SAME EXPERIMENTAL CONDITIONS (eg, frequencies and spot sizes)
	vals=[] ; global fitting,time_normalize ; fitting_old=fitting ; tnorm_old=time_normalize
	for f in filesToRead:
		if mode=="TDTR":
			fitting="X" ; time_normalize=""
			ts,xs=readFile(f)
			fitting="Y"
			ts,ys=readFile(f)
			vals.append([ts,xs,ys])
		elif mode=="SSTR":
			P,M=readFile(f)
			ones=np.ones(len(P)) ; zeros=np.zeros(len(P))
			vals.append([P,zeros,zeros,zeros,M,zeros,zeros,zeros,ones])
		#else:
		#	xs,ys=readFile(f)
		#	vals.append([xs,ys])
	fitting=fitting_old ; time_normalize=tnorm_old
	vals=np.mean(vals,axis=0)
	#print(vals)
	saveGen(vals,fileOut)

def combinedFilename1(listOfFiles,fileOut=""):
	conditionalPrint("combinedFilename","received:"+str(listOfFiles))
	direc="/".join(listOfFiles[0].split("/")[:-1]) # select the first file's directory (no guarantee all files exist in same folder!)
	listOfFiles=[ f.split("/")[-1] for f in listOfFiles ]
	for i in range(len(listOfFiles[0])):	# loop through all characters of the first filename
		c=listOfFiles[0][i]
		for f in listOfFiles:		# loop through all other filenames
			if i>=len(f) or c!=f[i]:# and check the ith character
				fileOut+="x"	# if non-matching, set to "x". 
				break
		else:				# for-else: if we didn't break, all files' ith character matches, so keep it
			fileOut+=c
	return direc+"/"+fileOut

def combinedFilename(listOfFiles,fileOut=""):
	conditionalPrint("combinedFilename","received:"+str(listOfFiles))
	# slice out first file's directory (in case files are in different places, default to first's)
	direc="/".join(listOfFiles[0].split("/")[:-1])
	listOfFiles=[ f.split("/")[-1] for f in listOfFiles ]

	# basic sub-function to loop through all files checking a given substring
	def isCommon(substr,lst):
		for f in lst:
			if substr not in f:
				return False
		return True

	# loop through length-4 chunks. if it exists in all filenames, increment length and retry (so we capture >4 length substrings too).
	substr="" ; i=-1
	while i<=len(listOfFiles[0])-4:		# increment i (instead of "for i in range", because we may want to skip some chars if we found a match)
		i+=1 ; N=3			# reset substring length to 3, so we can quit if it's 4 or more
		#print("substrSearchStart, i =",i,N,len(listOfFiles[0])-i+1)
		while N<len(listOfFiles[0])-i+1:
			sub=listOfFiles[0][i:i+N]
			#print(i,N,sub)
			if isCommon(sub,listOfFiles):
				N+=1		# increment length ("abc" matched, so now check "abcd")
			else:
				sub=sub[:-1]	# ( if it didn't match, remove the last non-matching character )
				break
		if N>3:
			substr+=sub		# add the substring we checked against whatever else we've found. or, "*" if no match
			i+=N
		else:
			if len(substr)==0 or substr[-1]!="x": # no match (and either nothing in substr yet, or pevious char is not x)
				substr+="x"
	#print(substr)
	return direc+"/"+substr


def fileAverager2(filesToRead,fileOut=""):
	conditionalPrint("fileAverager","preparing:"+str(filesToRead))
	if len(fileOut)==0:
		#for i in range(len(filesToRead[0])):
		#	c=filesToRead[0][i]
		#	for f in filesToRead:
		#		if c!=f[i]:
		#			fileOut+="x"
		#			break
		#	else:
		#		fileOut+=c
		fileOut=combinedFilename(filesToRead,fileOut)	

		fileOut=fileOut.replace(".txt","_AVG.txt")

	header=[] ; averaged=[] ; delim="\t"
	files=[ open(f,'r') for f in filesToRead ]	# simultaneously open all files
	lines=[ f.readlines() for f in files ]		# and read all lines from all files
	for i,l in enumerate(lines[0]):				# loop through first file
		for c in l:								# inspecting every character
			if c not in "0123456789e,.-\t\n ":	# if not a number, it's a header!
				break 							# (stop inspecting all chars) 
		else:							 		# if we finished inspecting the whole line (didn't break)
			delim={True:",",False:"\t"}["," in l]
			break 								# then this line has data, not header, so stop checking lines
		header.append(l)							# if we DID find a header (broke on chars), add to header
	for j in range(i,len(lines[0])):	 			# loop through all DATA lines, starting on i
		data=[ l[j] for l in lines ]
		data=[ [ float(v) for v in d.split(delim) ] for d in data ]
		data=np.mean(data,axis=0)
		data=[ str(v) for v in data ]
		averaged.append(delim.join(data)+"\n")
	with open(fileOut,'w') as f:
		for h in header:
			f.write(h)
		for l in averaged:
			f.write(l)
	return fileOut			

# depending on fileType passed we either: 
# raw: average the raw values, line-by-line/column/by-column between files (breaks if files are different lengths, e.g., one TDTR scan has picosecond ultrasonics data in it)
# fSSTR/TDTR: generate a new "standardized" point spacing in the x axis, and perform interpolation
# "ignoreOutliers" only works for fSSTR: we check the slope of each dataset, excluding any with 1.5 standard deviation from the mean
def fileAverager(filesToRead,fileOut="",fileType="raw",ignoreOutliers=False):
	conditionalPrint("fileAverager","running for fileType="+fileType+", on files: "+",".join(filesToRead))
	# regardless of method, read all lines, check file lengths, and pick out the file header
	lines=[ open(f).readlines() for f in filesToRead ] # read all lines from all files [whichfile][whichline]
	length=[ len(l) for l in lines ] # number of lines per each file

	# pick out the header
	averaged=[]
	for n in range(length[0]):
		line=[ l[n] for l in lines ]
		if len(set(line))==1: # line is identical in all files, e.g., header
			averaged.append(line[0])
		else:
			break

	# "raw" method literally just averages floats line-by-line
	if fileType=="raw":
		if len(set(length))!=1:
			print("ERROR: file-averager received files of differing length, which is incompatible with filetype \"raw\"")
			return
		length=length[0]
		for n in range(n,length): # we already found the end of the header at line n, so start there
			line=[ l[n] for l in lines ]
			if len(set(line))==1: # line is identical in all files, e.g., header
				averaged.append(line[0])
			else:
				vals=[ [ float(v) for v in l.split() ] for l in line ]
				vals=np.asarray(vals)
				vals=np.mean(vals,axis=0)
				vals=[ str(v) for v in vals ]
				averaged.append("\t".join(vals)+"\n")

	ignored=[]
	# "fSSTR" reads fiber SSTR files, does lin-interpolate, and then generates new points based on the interpolation. this is useful if you have datasets with differing number of measurement points
	if fileType=="fSSTR":
		Ps,Ms,funs=[],[],[]
		for f in filesToRead:
			P,M=readFiberSSTR(f)
			Ps.append(P) ; Ms.append(M)
			funs.append(interp1d(P,M,kind="quadratic"))
		if ignoreOutliers:
			# Ps and Ms are list of datapoints, for each file. fine mean slope of each file
			slopes = [ np.mean(M/P) for P,M in zip(Ps,Ms) ] ; avg_s=np.mean(slopes) ; std_s=np.std(slopes)
			print(slopes,avg_s,std_s)
			# check each slope. anything beyond 1.5 standard deviations is considered an outlier here??
			mask=np.zeros(len(slopes)) ; mask[slopes<avg_s-1.5*std_s]=-1 ; mask[slopes>avg_s+1.5*std_s]=1
			print(mask)
			# filter Ps,Ms, to where mask==0. similar to "Ps=Ps[mask==0]" but Ps may be ragged (datasets may differ in number of points)
			Ps=[ P for P,m in zip(Ps,mask) if m==0 ]
			Ms=[ M for M,m in zip(Ms,mask) if m==0 ]
			funs=[ f for f,m in zip(funs,mask) if m==0 ]
			ignored=[ f for f,m in zip(filesToRead,mask) if m!=0 ]
			if len(ignored)>0:
				print("ignoring",len(ignored),"outliers:",ignored)
		Pmin=max([min(p) for p in Ps]) ; Pmax=min([max(p) for p in Ps]) ; nPows=max([len(p) for p in Ps])
		P=np.linspace(Pmin,Pmax,nPows)
		M=[ fun(P) for fun in funs ]
		M=np.mean(M,axis=0)

		for p,m in zip(P,M):
			p,m=str(np.round(p,6)),str(np.round(m,6))
			averaged.append( p+"\t0\t0\t0\t"+m+"\t0\t0\t0\t1\t1\n")# PuX PuXStd PuY PuYStd PrX PrXStd PrY PrYStd AuxIn1 AuxIn2

	# "TDTR" reads TDTR files, does quadratic-interp, to put all points onto a "standardized" log-spacing of time-points. this is useful if you have datasets with differing number of points, e.g. one scan is pico-acoustics plus thermal, and other scans are just thermal
	elif fileType=="TDTR":
		toplotX=[] ; toplotY=[] ; toplotMKR=[]
		rise=np.linspace(-15e-12,10e-12,26)
		a=np.log(10e-12)/np.log(10);b=np.log(5.49e-9)/np.log(10)
		thermal=np.logspace(a,b,50)
		ts=np.concatenate((rise,thermal))
		xs=[] ; ys=[] ; aux1=[] ; aux2=[]
		for f in filesToRead:
			tsf,xsf,ysf,auxesf=readTDTRdata(f)
			toplotX.append(tsf) ; toplotMKR.append("ko")
			toplotY.append({"R":-ysf/xsf,"M":(ysf**2+xsf**2)**.5,"X":xsf,"Y":ysf}[fitting]) 
			interp=interp1d(tsf,xsf,kind="quadratic") ; xs.append(interp(ts))
			ts2=np.linspace(-15e-12,5.49e-9,10000) ; x2=interp(ts2)
			#toplotX.append(ts2) ; toplotY.append(x2) ; toplotMKR.append("-")
			interp=interp1d(tsf,ysf,kind="quadratic") ; ys.append(interp(ts))
			y2=interp(ts2) ; toplotX.append(ts2) ; toplotMKR.append("-")
			toplotY.append({"R":-y2/x2,"M":(y2**2+x2**2)**.5,"X":x2,"Y":y2}[fitting])
			interp=interp1d(tsf,auxesf[:,0],kind="quadratic") ; aux1.append(interp(ts))
			interp=interp1d(tsf,auxesf[:,1],kind="quadratic") ; aux2.append(interp(ts))
		xs=np.mean(xs,axis=0) ; ys=np.mean(ys,axis=0) ; aux1=np.mean(aux1,axis=0) ; aux2=np.mean(aux2,axis=0)
		toplotX.append(ts) ; toplotMKR.append("ro")
		toplotY.append({"R":-ys/xs,"M":(ys**2+xs**2)**.5,"X":xs,"Y":ys}[fitting])
		#lplot(toplotX,toplotY,markers=toplotMKR) #; sys.exit()
		for t,x,y,a1,a2 in zip(ts,xs,ys,aux1,aux2):
			r=(x**2+y**2)**(1/2) ; p=np.arctan2(y,x)*180/np.pi
			t,x,y,r,p,a1,a2=[ str(v) for v in [t*1e12,x,y,r,p,a1,a2] ]
			averaged.append(t+"\t"+t+"\t"+x+"\t"+y+"\t"+r+"\t"+p+"\t"+a1+"\t"+a2+"\n") # Pos Delay(ps) X Y R Phi Auxin0 Auxin1

	if len(fileOut)==0:
		fileOut=combinedFilename(filesToRead,fileOut)
		fileOut=fileOut.replace(".txt","_AVG.txt")
	with open(fileOut,'w') as f:
		for l in averaged:
			f.write(l)
	return fileOut,ignored

#from scipy.optimize import curve_fit #[param1Result,param2result,...],[residual,stdev]
from scipy.optimize import curve_fit
# we wrap scipy.optimize.curve_fit in order to handle scaling (curve_fit calls least_squares, but lsq really doesn't like when the fitting parameters are significant orders of magnitude off in scale, e.g. thermal conductivity is typically 1-100 W/m/K whereas thermal boundary resistance is typically ~5e-9 m²K/W, and typically scaled by the user and presented as 5 m²K/GW. fitting both (conductivity and unscaled TBR) may be fragile, so we scale the terms to get them within similar orders of magnitude. 
# Two strategies:
# 1. take advantage of least_squares' x_scale argument, and take advantage of the fact that curve_fit appears to cascade this argument through to least_squares. this helps, but is imperfect (e.g. check out Research/IBB/GaN- C N Ga/Data Analysis Playground/GaN-CNGa2.py > fitAll. set REGEN=True, and then run "python3 GaN-CNGa2.py | grep bad | wc -l". 70 bad fits with x_scale off, 28 with x_scale on. pretty sizable difference, but fitting for TBC still beats it, with only 16 bad fits. You can also try using gui.py to fit the Sapphire cals from 2021_06_15_HQ, with useTBR=yes. the results will be highly sensitive to your guess values: 5e-9 m²K/W for R1 gives bad cals, but 6.5e-9 m²K/W gives good cals. fitting for TBC with a guess of 200e6 W/m²/K gives good fits too, despite being identical numerically to 5e-6 m²K/W. simply removing x_scale actually seems to help with the cals, but does not help with the GaN-CNGa2.py > fitall() troubles).
# 2. do the scaling ourselves. We can either update every single "func" (TDTRfunc, SSTRfunc, FDTRfunc, PWAfunc), or their shared child functions (e.g. popGlos), to handle TBRs as 1e0 order of magnitude numbers with m²K/GW instead of 1e-9 order of magnitude numbers with units of m²K/W. OR, we can "simply" wrap the function, so curve_fit sees unity order of magnitudes. 
def curvefit(fun,x,y,p0,bounds,bonusArgs=""): # DIY version of scipy.optimize.curve_fit, which doesn't allow x_scale
	p0=list(p0)
	for i,p in enumerate(p0):
		if p<bounds[0][i]:
			p0[i]=bounds[0][i]
		elif p>bounds[1][i]:
			p0[i]=bounds[1][i]

	# factDict={'rp':1e6,'Kz':1,'Kr':1,'G':1e-6,'C':1e-6,'d':1e9,'R':1e9}
	# STRATEGY 1, SIMPLY PASS x_scale THROUGH curve_fit TO least_squares. BUT THIS DOESN'T ALWAYS BEHAVE....
	#x_scale=[ 1/factDict[p[:-1]] for p in tofit ]
	#results=curve_fit(fun,x,y,p0=p0,bounds=bounds)#,x_scale=x_scale)
	# STRATEGY 2, DO THE SCALING OURSELVES
	# e.g. suppose we're fitting for TBR, a good value might be 5e-9 m²K/W (or 5 m²K/GW). curve_fit doesn't love nano[numbers], so we'll scale them up. instead of updating TDTRfunc (passed into here as "fun") (and also editing SSTRfunc and PWAfunc and FDTRfunc, or maybe some other child function of theirs like popGlos), we'll simply wrap TDTRfunc. TDTRfunc should still receive 5e-9 [m²K/W], but we want curve_fit to see 5 [m²K/GW]
	# for this to work, we need to scale up p0 (5 [m²K/GW]) and bounds (.1 to 20 [m²K/GW]), wrap fun and scale back down before fun is called. then at the end, curve fit will return a best result (6.25 [m²K/GW]) so we need to scale that back down too (returning 6.25e-9 [m²K/W]). 
	scale=np.asarray( [ getScaleUnits(p)[0] for p in tofit ] )	
	def wrapped(xs,*params): # e.g. "fun" might be TDTRfunc(ts,*parameterValues,store=False,addNoise=False,whackyFunc=None)
		params/=scale 
		return fun(xs,*params)
	p0*=scale ; bounds=np.asarray(bounds)*scale # [[lb1,lb2,lb3...],[ub1,ub2,ub3...]]
	conditionalPrint("curvefit","bounds: "+str(bounds)+", p0: "+str(p0))
	#print(x,y,p0,bounds)
	results,error=curve_fit(wrapped,x,y,p0=p0,bounds=bounds)
	return results/scale,error

# wrapper for something like: lsqout=minimize(ss3hwrapped2, tuple(guesses), bounds=tuple(list(zip(*bnds))) 
def minimizee(fun,p0,bounds,method=None):
	scale=np.asarray( [ getScaleUnits(p)[0] for p in tofit ] )	
	#print("scale",scale)
	def wrapped(params):
		conditionalPrint("minwrapped",str(params)) #; sys.exit()
		params/=scale
		return fun(params)
	p0*=scale ; bounds=np.asarray(bounds)*scale
	bnds=[ [ bounds[0][i] , bounds[1][i] ] for i in range(len(tofit)) ]
	#print("p0,bounds,bnds",p0,bounds,bnds) #; sys.exit()
	lsqout=minimize(wrapped,x0=tuple(p0),bounds=tuple(bnds),method=method) ; lsqout['x']/=scale
	return lsqout

# HELP! cals too low, check pump/probe overlap. too high, check offsets. noisy, check allignment. 
def solveTDTR(fileToRead,plotting="show",ts='',data='',skipSolved=False): # "plotting" options include: show, save, showsens, savesens, none
	conditionalPrint("solveTDTR",fileToRead)
	global mode ; mode="TDTR"
	incrementCounter("solve")
	if skipSolved:
		conditionalPrint("solveTDTR","skipSolved = True: checking for figfile")
		figFile="/".join(fileToRead.split("/")[:-1])+"/"+callingScript+"/pics/"+fileToRead.split("/")[-1]+".png"
		if os.path.exists(figFile):
			return np.zeros(len(tofit)),0
	if len(ts)>0 and len(fileToRead)>0: # custom ts,data was passed, but we still may want to read in rpu/rpr/fm
		A,B=readTDTR(fileToRead)
	elif len(fileToRead)>0:
		ts,data=readTDTR(fileToRead) #go get data from file. (also sets global processing parameters like spot size, modulation frequency)
	if len(ts)<3:
		conditionalPrint("solveTDTR","ts is too short! not enough data!"+str(ts)+str(data))
		return np.zeros(len(tofit)),[0,[0]]
	conditionalPrint("solveTDTR","prepared to run with the following parameters:",pp=True)
	guesses=getTofitVals() #get guesses from starting thermal property matrix
	bnds=lookupBounds()
	conditionalPrint("solveTDTR","guesses / bounds : "+str(guesses)+" , "+str(bnds))
	if len(tofit)>0:
		#x_scale=[ 1/factDict[p[:-1]] for p in tofit ] ; print(x_scale) # testing: xscale seems to help, so let's generalize it
		solvedParams, parm_cov = curvefit(TDTRfunc, ts, data, p0=tuple(guesses),bounds=tuple(bnds))#,x_scale=x_scale) #solve it
		sigmas=np.sqrt(np.diag(parm_cov)) # this is is the standard deviation for each parameter: "if we had random noise in our data, the best-fit could be something other than the real (un-noised) function"
	else:
		solvedParams=[] ; sigmas=[0]
	conditionalPrint("solveTDTR","found: "+str(solvedParams)+","+str(sigmas))
	residual=resultsPlotter(fileToRead,ts,data,solvedParams,plotting)
	return solvedParams,[residual,sigmas]

# TDTR: a series of pulses heats the sample, and probes the temperature rise some time later:
# Vᵢₙ(t𝘥)  = Re(Z(ω)) , Vₒᵤₜ(t𝘥) = Im(Z(ω))							#Jiang eq 2.21/2.22 (modified) 
# Z(ω)= Σ ΔT(ωₘ+n*ωₚ)*exp(i*n*ωₚ*t𝘥 ) (pulsed) or Z(ω)=ΔT(ωₘ) (CW)				#Jiang eq 2.19  / Schmidt eq 2/10
#	ΔT(ω)=A₁ ∫ Ĝ(k,ω)*exp(-π²*k²*w₀²)*2*π*k*dk ; from 0 to ∞				#Jiang eq 2.18
#		Ĝ(k,ω)=-D/C 									#Jiang eq 2.9 / Schmidt eq 7
#		or Ĝ(k,ω)=(Aⱼ-Bⱼ*Cⱼₖ/Dⱼₖ)*(-Dᵢ*Dⱼₖ)/(Dᵢ*Cⱼₖ+Cᵢ*Dⱼₖ) 				bidirectional
# note that this Σₙ ΔT(ωₙ) stuff is effectively the same "sum of sines" fourier series stuff we do for PWA! 
expA=100 ; expB=-1 ; expC=0
def TDTRfunc(ts,*parameterValues,store=False,addNoise=False,whackyFunc=None):

	if "expA" in tofit:
		vals=[]
		for L in ["A","B","C"]:
			if "exp"+L in tofit:
				i=tofit.index("exp"+L) ; vals.append(parameterValues[i])
			else:
				vals.append(getVar("exp"+L))
		A,B,C=vals
		return A*np.exp(ts*1e9*B)+C

	#def whackyFunc(ts,xs,ys):
	#	xs+=yemxpb(ts,1e6,.1)
	#	return ts,xs,ys

	incrementCounter("TDTRfunc")

	if len(parameterValues)==len(tofit): 
		setTofitVals(parameterValues)
	popGlos()

	nonzeroAbsorption="gradient"
	global alpha
	# NON-ZERO OPTICAL PENETRATION DEPTH: discretize into 10 locations to dump heat, with a weighted average for signal
	if nonzeroAbsorption=="gradient" and alpha!=0:
		global depositAt ; depositAt_old=depositAt ; alpha_old=alpha
		conditionalPrint("TDTRfunc","alpha="+str(alpha)+", recursing")
		depths=np.linspace(0,2*alpha,10) ; expos=np.exp(-depths/alpha) ; alpha=0
		results=np.zeros((10,len(ts))) ; expos/=sum(expos)
		#lplot([depths],[expos]) ; sys.exit()
		for i in range(10):
			depositAt=depths[i]
			results[i,:]=TDTRfunc(ts)*expos[i]
		depositAt=depositAt_old ; alpha=alpha_old
		return np.sum(results,axis=0)
	# CAHILL TRICK FOR NON-ZERO OPTICAL PENETRATION DEPTHS: 
	elif nonzeroAbsorption=="cahill" and alpha!=0:
		global tp ; alpha_old=alpha ; tp_old=tp
		conditionalPrint("TDTRfunc","alpha="+str(alpha)+", adding layer")
		tp=[[tp[0][0],tp[0][1],tp[0][2],tp[0][3]],[{True:0,False:np.inf}[useTBR]]]+tp
		tp[0][0]*=alpha*1e9 ; tp[0][2]=1e-9 ; alpha=0 
		result=TDTRfunc(ts)
		tp=tp_old ; alpha=alpha_old
		return result
	
	if nmax>1000:
		warn("TDTRfunc","Warning: summing over "+str(nmax*2)+" sidebands. please check that your pulse frequency is correct, and consider a larger minimum fitting time for better performance.")

	ns=np.arange(nmin,nmax+1)			# used for summing over many frequencies
	omegas=omegaM+ns*omegaP 			# [ ωₙ ] , 1D list of ω=ωₘ+n*ωₚ values to pass into ΔT(ω). 
	if len(omegas)>10000:				# performance considerations: all at once is faster, but heavier on RAM if too many ω values
		delTplus=np.zeros(len(omegas),dtype=np.complex128)
		for i in range(int(np.ceil(len(omegas)/10000))):
			print("processing delTomega in pieces:",i+1,"/",int(np.ceil(len(omegas)/10000)))
			i1=i*10000 ; i2=(i+1)*10000
			delTplus[i1:i2]=delTomega(omegas[i1:i2])
	else:
		delTplus=delTomega(omegas)
	conditionalPrint("TDTRfunc","using parameters:",pp=True)

	convergeAccelerator=np.exp(-pi*ns**2./nmax**2.)	# [ ωₙ ], each n represents a frequency, Cahill eq 20+, exp(-πf²/fₘₐₓ²)

	#Z(t𝘥)= Σ ΔT(ωₘ+n*ωₚ)*exp(i*n*ωₚ*t𝘥) #from -∞ to ∞, Jiang eq 2.21/2.22 (modified) / Schmidt eq 2. 
	sumbits=delTplus[None,:]*np.exp(1j*omegaP*np.outer(ts,ns))[:,:]*convergeAccelerator[None,:] # [ t, ωₙ ]
	z=np.sum(sumbits,axis=1) # [ t ], sum over all ωₙ 
	#Note that while Jiang states Vᵢₙ(t𝘥)=½Σ(ΔT(ωₘ+n*ωₚ)+ΔT(-ωₘ+n*ωₚ))*exp(i*n*ωₚ*t𝘥) and Vₒᵤₜ(t𝘥)=-i*½Σ(ΔT(ωₘ+n*ωₚ)-ΔT(-ωₘ+n*ωₚ))*exp(i*n*ωₚ*t𝘥), simply taking the real and imaginary parts of ΣΔT(ωₘ+n*ωₚ)*exp(i*n*ωₚ*t𝘥) yields the same result. 
	#Vᵢₙ(t𝘥)  = Re(Z(ω)) , Vₒᵤₜ(t𝘥) = Im(Z(ω))							#Jiang eq 2.21/2.22 (modified) 
	xs=z.real ; ys=z.imag

	# what is the purpose of noise? statistical analysis (no scan is perfect, what is it's fundamental noise level? if we find a residual value r, what percent of that is fundamental noise, vs incorrect fitting params?)
	if addNoise: 
		xs*=noise(ts,addNoise)
	if whackyFunc is not None:
		ts,xs,ys=whackyFunc(ts,xs,ys)
	if store:
		saveGen([ts,xs,ys],store)
	
	result = { "R":-xs/ys , "X":normalize(ts,xs) , "Y":normalize(ts,ys) , "M":normalize(ts,(xs**2.+ys**2.)**.5) }[fitting] #dict as 1 line switch-case
	conditionalPrint("TDTRfunc",fitting+" result:"+str(result))
	return result

def noise(xs,resid): # given a series of data, add noise to it (used for generating hypothetical datasets for predictUncert)
	n=np.random.normal(1,.05,size=len(xs)) ; o=np.ones(len(xs)) # gaussian noise for each datapoint, centered around 1, and a list of 1s
	n=np.absolute(n) ; n[::2]*=-1 # every other noisification should be above / below the line. gui.py > ctrl+d will check delta between dataset and data, and every other point being clearly above/below is a clear indicator that something is afoot
	res=RES(n,o)
	n-=1 ; n*=resid/res ; n+=1 # noise is 1+/-σ. we want to scale σ by desired/current
	conditionalPrint("noise","Adding noise: "+str(n))
	return n

def saveGen(txy,fname,delim="\t"):
	conditionalPrint("saveGen","saving "+mode+" dataset to file "+fname)
	header="# thermal properties matrix and other parameters:\n"
	if "Cs" in globals(): # don't crash if unpopulated (eg, we use saveGen for fileAverager)
		header=header+"#  Cs: "+str(Cs)+"\n#  Kzs: "+str(Kzs)+"\n#  ds: "+str(ds)
		header=header+"\n#  Krs: "+str(Krs)+"\n#  Gs: "+str(Gs)+"\n#  (Rs: "+str(Rs)+")\n#  nmax: "+str(nmax)+"\n"
	header=header+"#  fm: "+str(fm)+"\n#  fp: "+str(fp)+"\n#  rpump: "+str(rpump)+"\n#  rprobe: "+str(rprobe)+"\n#  tm: "+str(minimum_fitting_time)+"\n#  tn: "+str(time_normalize)+"\n"
	direc="/".join(fname.split("/")[:-1])
	if len(direc)>0:
		os.makedirs(direc, exist_ok=True)
	with open(fname,'w') as f:
		f.write(header)
		for row in zip(*txy):
			f.write(delim.join([str(v) for v in row])+"\n")

#TODO: need to test FDTR solving (pulsed and CW) : If you are a user of FDTR, i would appreciate if you sent me some data files for testing. (see testing23 for examples)
# Discussion: how does phase correction with FDTR work? we can monitor the pump signal, and use that
# as one phase-correction step (in drift in phase of the pump is thus captured), however there may still
# be some other systematic phase (e.g. what if there is a time lag between pump acquisition and probe
# acquisition). For now, we'll assume the systematic phase is linear with frequency (a constant time lag,
# vs a uniformly-varying period duration dependant on frequency, will produce a phase lag proportional to
# frequency). This can be accounted for as a correction applied to the data, or as a correction applied
# to the model, or both. There is also the issue of wrapping (a high phase lag is represented as a phase
# lead), and this wrapping will result in instability in fitting. To account for both, for example, we
# might arbitraryly set the phase of the highest-frequency datapoint to zero, then let the model do the 
# same, or we might arbitrarily set the highest-frequency datapoint to pi/2 phase lag (and ditto for the
# model). These will give the same fits, BUT, different residual values (it's not a "scaling" of data, 
# it's a "shifting" of data. dY=0.01 when Y=1 is 1% error, but dY=.01 (same dY, data and model are both
# just shifted) when Y=.1 is 10% error). 
# instead, when fitting for phase, we effectively need to fit the data to the model (instead of model to
# data, as is the normal procedure).
slopedPhaseOffset=0 # Attempts at a theory-based phase offset scheme: laser path / electronics / etc may all create a phase offset between the "true" probe response and the "measured" probe response (or you can think of it as time delay between two supposedly-simultaneous sinusoidal signals at a given frequency). for FDTR, it is common practice to fit for the phase offset (because there is no time-delay=0 crossing to use for automatic phase correction like with TDTR)
variablePhaseOffset=0 # empirical phase offset: conceivably data may not follow slopedPhaseOffset, so instead, simply use a reference sample to generate a phase-vs-frequency offset dataset. fit for vphase on a known sample, then the point-by-point correction will be applied to all subsequent fits
# TODO phase offset may be frequency-dependent! 
def solveFDTR(fileToRead,plotting="show"):
	global mode,tofit,variablePhaseOffset ; mode="FDTR"
	conditionalPrint("solveFDTR","importing file:"+fileToRead)
	conditionalPrint("solveFDTR","phase offsets: slopedPhaseOffset: "+str(slopedPhaseOffset)+", variablePhaseOffset: "+str(variablePhaseOffset))
	#FILE READING
	fs,phis=readFDTR(fileToRead) #; phis+=variablePhaseOffset
	#global slopedPhaseOffset ; slopedPhaseOffset=0
	#phasefile=fileToRead.split("/")[:-1]
	#phasefile.append("phase.txt")
	#phasefile="/".join(phasefile)
	#if os.path.exists(phasefile):
	#	vPO=open(phasefile).readlines()[0]
	#	vPO=vPO.split(",")
	#	variablePhaseOffset=np.asarray([ float(v) for v in vPO ])
	#FITTING
	guesses=getTofitVals() ; bnds=lookupBounds() # guesses come from thermal property matrix, bounds come from ubs / lbs globals
	#if "phase" in tofit:
	#	variablePhaseOffset=FDTRfunc(fs)-phis
	#	with open(phasefile,'w') as f:
	#		f.write(",".join([str(v) for v in variablePhaseOffset]))
	#	tofit=[]
	#corrected=phis+variablePhaseOffset ; corrected[corrected>np.pi]-=2*np.pi ; corrected[corrected<-np.pi]+=2*np.pi
	#plot([fs,fs,fs],[phis,variablePhaseOffset,corrected],xscale="log")
	#phis=corrected
	#phis+=variablePhaseOffset
	#fs=fs[phis>-3] ; phis=phis[phis>-3]
	#fs=fs[phis<3]  ; phis=phis[phis<3]
	if tofit==["variablePhaseOffset"] or tofit==["phase"]:
		phi_m=FDTRfunc(fs) ; phis-=variablePhaseOffset # remove old offset
		variablePhaseOffset=phi_m-phis
		print(variablePhaseOffset) ; fs,phis=readFDTR(fileToRead)
		tofit=[] ; solvedParams=[] ; sigmas=[]
	elif tofit==["slopedPhaseOffset"] or tofit==["sphase"]: # if fitting for phase, since the offset is applied in readFDTR, we sort of do the fitting "backwards": func in curve_fit is readFDTR and the "data" is the model
		phi_m=FDTRfunc(fs)
		def readFDTRp(fs,slope):
			global slopedPhaseOffset
			slopedPhaseOffset=slope
			fs,phis=readFDTR(fileToRead) ; print(slope,phis)
			return phis
		solvedParams, parm_cov = curve_fit(readFDTRp,fs,phi_m,p0=([4]),bounds=tuple(bnds)) ; sigmas=[0]
		fs,phis=readFDTR(fileToRead)
		tofit=[] ; solvedParams=[] ; sigmas=[]
	elif len(tofit)!=0:
		solvedParams, parm_cov = curvefit(FDTRfunc, fs, phis, p0=tuple(guesses),bounds=tuple(bnds)) #solve it
		sigmas=np.sqrt(np.diag(parm_cov))
		#lsqout=least_squares(dzFDTR, tuple(guesses), bounds=tuple(bnds), args=(fs,phis,fileToRead))
		#solvedParams=lsqout['x']
		#sigmas=np.zeros(len(tofit))
	else:
		solvedParams=[] ; sigmas=[]
	#fs,phi=readFDTR(fileToRead)
	conditionalPrint("solveFDTR","solved:"+str(solvedParams))
	residual=resultsPlotter(fileToRead,fs,phis,solvedParams,plotting)
	return solvedParams,[residual,sigmas]

def calsForPhase(fileDirec,calmatDirec,materials=["Al2O3","SiO2","Quartz","Si"]): # based onf calsForSpot2, which is based on superMegaFit.py (2022_12_13_Fiber)
	aliases={"Quartz":"cSiO2"} ; setVar("fitting","P") ; setVar("mode","FDTR") # readFDTR > autos both require fitting phase and FDTR in order to import all the stuff we want
	dphis=[] ; labels=[]
	for mat in materials:					# for each cal
		files=glob.glob(fileDirec+"/*_"+mat+"_*FDTR.txt")				# n files following this naming pattern
		if mat in aliases.keys():
			files+=glob.glob(fileDirec+"/*_"+aliases[mat]+"_*FDTR.txt")
		matfile=glob.glob(calmatDirec+"/*"+mat+"_cal_matrix.txt")[0]
		importMatrix(matfile)
		for f in files:
			fs,Y=readFDTR(f)
			generated=FDTRfunc(fs)
			dphis.append(generated-Y) ; labels.append(f.split("/")[-1])
			#plot([fs,fs,fs],[Y,generated,dphis[-1]],xlabel="f (Hz)",ylabel="phi (rad)",title=mat,labels=["data","generated","gen-data"],xscale="log")
	# averaging dphi is a bit more complicated than simply mean(). mean(355,5) != 0, which is the actual correct average angle!
	# https://stackoverflow.com/questions/491738/how-do-you-calculate-the-average-of-a-set-of-circular-data
	xs=np.zeros(len(dphis[0])) ; ys=np.zeros(len(dphis[0]))
	for dp in dphis:
		xs+=np.cos(dp) ; ys+=np.sin(dp)
	dphi = np.arctan2(ys, xs)
	dphis.append(dphi) ; labels.append("averaged")
	for i in range(len(dphis)):
		dphis[i][dphis[i]>np.pi]-=2*np.pi ; dphis[i][dphis[i]<-np.pi]+=2*np.pi
	plot([fs]*len(dphis),dphis,xlabel="frequency (Hz)",ylabel="Δϕ (rad)",title="calsForSpots()",labels=[""]*len(dphis),xscale="log",yscale="log",filename="calsForPhase.png")
	f=open(fileDirec+"/phase.txt",'w')
	f.write(",".join([str(p) for p in dphi]))
	print(dphi)
	f.close()
"""
def dzFDTR(parameterValues,fs,data,filename):
	model=FDTRfunc(fs,*parameterValues)
	#corrected=data+slopedPhaseOffset*fs
	#corrected[corrected<-np.pi]+=2*np.pi
	#corrected[corrected>np.pi]-=2*np.pi
	fs,corrected=readFDTR(filename)
	conditionalPrint("dzFDTR","ran with",pp=True)
	return model-corrected
"""
def readFDTR(filename):
	autos(filename)
	data=np.loadtxt(filename,skiprows=2)

	npts,ncols=np.shape(data)
	if ncols==3: # files from TDTRfunc(save!=False)
		fs,xs,ys=np.transpose(data)
		xs=xs[fs>=minimum_fitting_frequency] ; ys=ys[fs>=minimum_fitting_frequency] ; fs=fs[fs>=minimum_fitting_frequency]
		Y = { "R":-xs/ys , "X":normalize(fs,xs) , "Y":normalize(fs,ys) , "M":normalize(fs,(xs**2.+ys**2.)**.5) , "P":np.arctan2(ys,xs)}[fitting]	
		return fs,Y

	fs,pux,puxs,puy,puys,prx,prxs,pry,prys,a1,a2=np.transpose(data)

	fmin=1e3 ; fmax=.8e7

	pux=pux[fs>=fmin] ; puxs=puxs[fs>=fmin] ; puy=puy[fs>=fmin] ; puys=puys[fs>=fmin]
	prx=prx[fs>=fmin] ; prxs=prxs[fs>=fmin] ; pry=pry[fs>=fmin] ; prys=prys[fs>=fmin]
	a1=a1[fs>=fmin] ; a2=a2[fs>=fmin] ; fs=fs[fs>=fmin]
	
	pux=pux[fs<=fmax] ; puxs=puxs[fs<=fmax] ; puy=puy[fs<=fmax] ; puys=puys[fs<=fmax]
	prx=prx[fs<=fmax] ; prxs=prxs[fs<=fmax] ; pry=pry[fs<=fmax] ; prys=prys[fs<=fmax]
	a1=a1[fs<=fmax] ; a2=a2[fs<=fmax] ; fs=fs[fs<=fmax]
	
	dphi=-np.arctan2(puy,pux)+slopedPhaseOffset*fs/1e7+variablePhaseOffset # phase offset is the angle from each recorded pump datapoint
#global variablePhaseOffset ; variablePhaseOffset=np.zeros(len(fs))
	#pux , puy = pux*np.cos(dphi)-puy*np.sin(dphi) , puy*np.cos(dphi)+pux*np.sin(dphi)
	#lplot([fs],[np.arctan2(puy,pux)])
	#sys.exit()

	#if not doPhaseCorrect:
	#dphi=0
	prx , pry = prx*np.cos(dphi)-pry*np.sin(dphi) , pry*np.cos(dphi)+prx*np.sin(dphi) # Braun "The role of compositional..." Eq 3.68
	#yfixed=
	
	#lplot([fs],[dphi],"freq","dphi") ; sys.exit()
	#fs,xs,ys=[],[],[]
	#lines=open(fileToRead).readlines()
	#for l in lines:
	#	if not isNum(l[0]):
	#		continue
	#	l=l.split("\t")
	#	f,x,y=[float(v) for v in l[:3]]
	#	fs.append(f) ; xs.append(x) ; ys.append(y)
	#fs=np.asarray(fs) ; xs=np.asarray(xs) ; ys=np.asarray(ys)
	#prx,pry=FDTRphase(fs,prx,pry)

	xs=prx ; ys=pry
	#xs=xfixed ; ys=yfixed

	xs=xs[fs>=minimum_fitting_frequency] ; ys=ys[fs>=minimum_fitting_frequency] ; fs=fs[fs>=minimum_fitting_frequency]
	Y = { "R":-xs/ys , "X":normalize(fs,xs) , "Y":normalize(fs,ys) , "M":normalize(fs,(xs**2.+ys**2.)**.5) , "P":np.arctan2(ys,xs)}[fitting]	
	#if fitting=="P":
	#	Y+=variablePhaseOffset
	return fs,Y

"""
def FDTRphase(fs,xs,ys):
	#return xs,ys
	for i in range(int(len(xs)*3/4),len(xs)):
		#i=-7
		rs , phi = np.sqrt(xs**2+ys**2) , np.arctan2(ys,xs)	
		
		pO=phi[i]/fs[i]+np.pi/2/fs[i]
		print("FDTRphase",i,phi[i],fs[i],pO)
		dphi=-pO*fs
		#dphi=-np.arctan2(ys[i],xs[i])
		#print(i,ys[i],xs[i],dphi)
		xs , ys = xs*np.cos(dphi)-ys*np.sin(dphi) , ys*np.cos(dphi)+xs*np.sin(dphi)

		#break
		#m=phi[i]/fs[i] # a "linear slope correction" applied to phase vs frequency
		#print(i,xs[i],ys[i],phi[i],fs[i],m)
		#phi/=abs(phi[i])
		#phi[phi<-np.pi]+=2*np.pi
		#phi[phi>np.pi]-=2*np.pi
	#xs , ys = rs*np.cos(phi) , rs*np.sin(phi)
	return xs,ys
"""

def FDTRfunc(fs,*parameterValues,store=False,addNoise=False):
	# Step 1: set passed parameters, infer mofulation frequency, and so on	
	if len(parameterValues)==len(tofit):
		setTofitVals(parameterValues)
	popGlos()
	conditionalPrint("FDTRfunc","using parameters:",pp=True)
	omegas=2*np.pi*fs
	Zs=delTomega(omegas)
	xs=Zs.real ; ys=Zs.imag

	#dphi=-slopedPhaseOffset*fs # phase offset is the angle from each recorded pump datapoint
	#xs , ys = xs*np.cos(dphi)-ys*np.sin(dphi) , ys*np.cos(dphi)+xs*np.sin(dphi)

	if addNoise: 
		xs*=noise(fs,addNoise)
	if store:
		saveGen([fs,xs,ys],store)
	Y = { "R":-xs/ys , "X":normalize(fs,xs) , "Y":normalize(fs,ys) , "M":normalize(fs,(xs**2.+ys**2.)**.5) , "P":np.arctan2(ys,xs)}[fitting]
	return Y#-variablePhaseOffset


plusMinus=0
tshift=0 ; chopwidth=5 ; normPWA=True ; yshiftPWA=0 ; centerY=False ; waveformPWA="square" ; fitRisePWA=False ; sumNPWA=10000 ; runAvgPWA=0 ; dutyCycle=50 ; diracIndex=0
timeMaskPWA="0:10,50:60" 	# "apply a mask to the PWA data, eg, only fitting data between 0-10% and 50-60% of the cycle"
timeNormPWA="25,75,10"		# "when normalizing the data, pick points 25 and 75% of the way along, and use those as max/min"
 # TODO: generalizing this code to handle more waveforms, also means the normalization code (written for square waves) fails and gives you really confusing results (including sign inversions and the likes). so we should fix that, or at the very least, stop defaulting to norm.
# a low-frequency square-wave heating is applied, and the rise and fall for each "heater on" / "heater off" event is measured (time dependant). The temperature response for this can be modeled as a fourier series representation of a square wave. 
# NORMALIZATION CONVENTION:		 .  _..---       .  _..---       .  _..---
# halfway through first rise is t=0	  /      |        /      |	  /      |
# 1/4th period in is normalized to 1	 /       |       /       |       /
# 3/4th period in is normalized to 0	          \     |         \     |
# noisy data can use mean(1/8<t<3/8) etc .         -..__|.         -..__|.
onefourth=1 ; threefourths=-1
from scipy.interpolate import interp1d
def PWAfunc(ts,*parameterValues,store=False,addNoise=False):
	incrementCounter("PWAfunc")

	# Step 1: set passed parameters, infer mofulation frequency, and so on	
	if len(parameterValues)==len(tofit):
		#print("try:",parameterValues)
		setTofitVals(parameterValues)
	
	popGlos()
	conditionalPrint("PWAfunc","using parameters:",pp=True)

	p=1/fm #; dt=ts[1]-ts[0]
	# step 2: fourier series of a square wave (or arbitrary waveform from pumpWaveform()!). use numpy to find fourier series of it.
	# Aₙ = fft( f(t) ) , ωₙ=2π*fftfrq( f(t )) , H(t) = Σ Re[Aₙ] * cos(ωₙ*t) - Im[Aₙ] * sin(ωₙ*t) , R(t) uses Aₙ*Z(ωₙ) instead of Aₙ
	ts_fine=np.linspace(0,p,int(sumNPWA),endpoint=False) ; dt=ts_fine[1]-ts_fine[0] # using a higher density of time points sums over more freqs
	args=[ts_fine,fm]
	conditionalPrint("PWAfunc","generating pump waveform")
	if waveformPWA=="square-gauss":
		args.append(chopwidth)
	H=pumpWaveform(*args)
	#plot([ts_fine],[H]) ; sys.exit()
	# the integral of H should NOT be 1, since measured power delivery (A1) is "energy/s" and one period may be less than a second
	# it also does not need to be scaled with power (A1), since this is included in delTomega() (it's really just a one-time scaling term)
	# and it's also not the integral we care about, since we'll later simply be *summing* every single sine (more timesteps, more sines to sum)
	H=H/np.mean(H)*dt/p
	conditionalPrint("PWAfunc","FFTing")
	fft=np.fft.fft(H) ; freq=np.fft.fftfreq(n=len(ts_fine),d=dt) # fft() -> Aₙ, fftfreq -> ωₙ === Σₙ Aₙ*sin(ωₙ)
	omegas=freq*np.pi*2
	# step 3: pass each frequency through ΔT(ω), multiply each result by prefactor
	conditionalPrint("PWAfunc","passing"+str(len(omegas))+" omegas into delTomega()")
	Zs=delTomega(omegas)*fft #; Zs=1j*np.sqrt((Zs.real**2+Zs.imag**2)) # ; Zs=1j*Zs.imag
	# step 4: since fourier series is sum of sines and cosines, temperature at a given point in time is the sum of each Aₙ*cos(ωₙ*t)+Bₙ*sin(ωₙ*t)
	def zt(ts):
	#	z=np.zeros(len(ts))
	#	for z,o in zip(Zs,omegas):	# METHOD 1: a for loop. goes easy on ram, but it's slow!
	#		zt+=-z.real*np.cos(o*(ts))-z.imag*np.sin(o*(ts))
		ot=np.outer(ts,omegas)		# METHOD 2: vectorized, twice as fast. but blows up ram if ts or omegas is yuge (eg, see testing49.py. we
		return np.sum( Zs.real*np.cos(ot)-Zs.imag*np.sin(ot) , axis=1) # can use this same code to generate a realistic TDTR T(t) plot across 1/fm
	# step 5: generate our time-dependant signal, and shift for zero-crossing at t=0, then re-generate
	ts_course=np.linspace(0,p,1000,endpoint=False) # generate one full cycle
	z=zt(ts_course)
	if z[0]>np.mean(z):
		tshift=0
	else:
		f=interp1d(z[:450],ts_course[:450]) # use just the rise to swap axes: t vs mag
		tshift=f(np.mean(z))
	#tshift=0
	z=zt(ts+tshift)
	# STEP 5 ALT: above seems to fail for unknown reasons, with arbitrary waveform 
	
	# NORMALIZATION OF THE FUNCTION: calculate zt for two points, 1/4 and 3/4th period. scale and shift to 1 and 0 respectively
	if normPWA:
		#z/=max(z)
		mint,maxt,lt=timeNormPWA.split(",")
		tmn=p*float(mint)/100 ; tmx=p*float(maxt)/100		# times we grab for normalization (which points on the curve become 0, 1)
		Tmn=zt(tmn) ; Tmx=zt(tmx)
		Tmn,Tmx=min(Tmn,Tmx),max(Tmn,Tmx) 			# triangle), so let's not flip the data.
		z-=Tmn							# shift and scale, min --> 0
		z/=(Tmx-Tmn)						# max --> 1

		

		#tmxmn=np.asarray([p*float(mint)/100,p*float(maxt)/100])
		#ztmxmn=list(sorted(list(zt(tmxmn+tshift))))
		#z-=ztmxmn[1] ; z/=(ztmxmn[0]-ztmxmn[1]) # shift/scale, data between 0 and 1
		#z*=(onefourth-threefourths) ; z+=threefourths # shift/scale data between N3 and N1
		#t,z=normalizePWA(ts,z)
	if centerY:
		z-=np.mean(z)

	if addNoise: 
		z*=noise(ts,addNoise)
	if store:
		saveGen([ts,z],store,delim="; ")

	return z
	#return np.roll(z,int(len(z)/2))

waveformReference="/media/Alexandria/U Virginia/Research/Various Code/runTR/20220505_washer_monitorPu_162628_PWA.txt"
def pumpWaveform(*args):
	def square(ts,f):
		ys=np.zeros(len(ts))#-1
		p=1/f
		ys[ts%p<p*dutyCycle/100]=1
		return ys
	def fourierSquare(ts,f,N=100):
		ys=np.zeros(len(ts))
		p=1/f
		for n in range(1,N,2):
			ys+=4/np.pi*1/n*np.sin(n*np.pi*2*ts/p) # f(x)=4/π Σ 1/n sin(nπx/L) https://mathworld.wolfram.com/FourierSeriesSquareWave.html
		return ys
	def gaussEdgeSquare(ts,f,p,duty=dutyCycle):
		if p==0:
			return square(ts,f)
		w=1/f*p/100						# gaussian template, and its integral
		derivative=np.zeros(len(ts))				#          .-.                        ____ 1
		derivative+=Gaussian(ts,1,0,w)				#         ' | '                     .'
		derivative-=Gaussian(ts,1,1/f*duty/100,w)		#       .'  |  '.       -->        |
		derivative+=Gaussian(ts,1,1/f,w)			# ___.-'    | w   '-._____   0 ___.'
		# positive gaussian centered around "rise", negative gaussian centered around "fall"
		#plot([ts],[derivative])
		ys=np.cumsum(derivative)
		ys-=min(ys) ; ys/=max(ys)
		return ys
	def triangle(ts,f,slantiness=50):
		p=1/f
		ys=np.zeros(len(ts))
		mask=np.zeros(len(ts))
		mask[ts%p<=p*slantiness/100]=1 # rising
		ys[mask==1]= ts[mask==1]/(p/2)
		ys[mask==0]=-ts[mask==0]/(p/2)+2
		return ys
	def dirac(ts,fm):
		ys=np.zeros(len(ts))
		ys[diracIndex]=1
		return ys
	def dirac2(ts,fm):
		ys=np.zeros(len(ts))
		n=fp/fm ; N=int(len(ts)/n) #; print(n,N)
		ys[0::N]=1 ; ys[ts>=max(ts)/2]=0
		#lplot([ts],[ys],plotting="show")
		return ys
	def dirac3(ts,fm):
		ys=np.zeros(len(ts))
		n=fp/fm ; N=int(len(ts)/n) #; print(n,N)
		ys[0::N]=1 ; ys*=np.sin(ts*fm*2*np.pi)/2+.5
		#lplot([ts],[ys],plotting="show")
		return ys
	def sine(ts,fm):
		return np.sin(ts*fm*2*np.pi)/2+.5
	def arbitrary(ts,fm): # measure the pump response (pump reference photodetector), and use the REAL waveform!
		#ts2,Ts2=readPWA(waveformReference) # read times and voltages from file. BEWARE, this may or may not do normalization!
		ts2,Ts2=[],[]
		lines=open(waveformReference).readlines()
		for l in lines:
			if not isNum(l[0]):
				continue
			l=l.split(";")
			ts2.append(float(l[0])) ; Ts2.append(float(l[1]))
		ts2=np.asarray(ts2) ; Ts2=np.asarray(Ts2)

		Ts2=np.append(Ts2,Ts2[0])		# PWA data is technically cyclic, so append first datapoint (avoids interpolation problems between last 
		ts2=np.append(ts2,ts2[-1]+(ts2[1]-ts2[0])) # and first point)
		ts2*=(1/fm)/(ts2[-1])		# rescale times as read-in so they match the frequency (just in case...it's never perfect)
		#plot([ts2],[Ts]) ; sys.exit()
		#print(ts,ts2,fm)
		f=interp1d(ts2,Ts2)
		Ts=f(ts)
		Ts-=np.amin(Ts) #; Ts/=np.trapz(Ts,x=ts) ; Ts*=(1/fm/2) # Ts goes from 0 to 1 for a square wave. we only have to worry about the zero point, because pumpWaveform handles the integration
		#plot([ts],[Ts]) ; sys.exit()

		return Ts

	f={ "square":square , "square-gauss":gaussEdgeSquare , "dirac":dirac , "dirac2":dirac2 , "dirac3":dirac3 , "triangle":triangle , "sine":sine , "arbitrary":arbitrary }[waveformPWA]

	H=f(*args)
	H/=np.trapz(H,args[0])*args[1] # area under, normalized by period (2Hz just has 2 sine waves per second, area under each should then be half)
	#plot([args[0]],[H]) ; sys.exit()
	return H

#@profile
def readPWA(fileToRead,fileToSubtract=""):
	# IMPORT FILE
	def parseFile(fileToRead):
		ts,Ts=[],[]
		lines=open(fileToRead).readlines()
		for l in lines:
			if not isNum(l[0]):
				continue
			l=l.split(";")
			ts.append(float(l[0])) ; Ts.append(float(l[1]))
		ts=np.asarray(ts) ; Ts=np.asarray(Ts)
		return ts,Ts
	ts,Ts=parseFile(fileToRead)
	if len(fileToSubtract)>3:
		ts2,Ts2=parseFile(fileToSubtract)
		Ts-=Ts2
	# INFER FREQUENCY
	if autofm:
		global fm
		fm=1/(len(ts)*(ts[1]-ts[0]))
	else:
		fm_data=1/(len(ts)*(ts[1]-ts[0]))
		ts/=fm/fm_data
	p=1/fm
	# DATA PROCESSING
	Ts=runningAverage(Ts,int(runAvgPWA)) #; Tmax=np.amax(Ts) ; Tmin=np.amin(Ts)
	#mask=np.zeros(len(Ts)) ; mask[ np.abs(Ts-np.mean(Ts)) <= (Tmax-Tmin)/3 ]=1 # "all datapoints within +/-50% of the mean" aka, crossovers
	#plot([ts[mask==1]],[Ts[mask==1]]) ; sys.exit()
	#print(mask)
	#def fittableTriangle(ts,A,B): # /\/\/\/  A is height, B is horizontal offset, p is the periodicity
	#	ys=np.zeros(len(ts)) ; mask=np.zeros(len(ts))
	#	# /\, positive B --> \/\, or negative B --> /\/
	#	m=A/p*2 # slope = rise / run
	#	mask[(ts-B)%p<=p/2]=1 # 1 = rising
	#	ys[mask==1] = m*( (ts[mask==1]-B)%p-p/4 ) # positive slope, centered on (x=p/2,y=0)
	#	ys[mask==0] =-m*( (ts[mask==0]-B)%p-3*p/4 )
	#	return ys
	#Xs=[] ; Ys=[]
	#for A in range(1,4):
	#	for B in np.linspace(-p/3,p/3,5):
	#		Xs.append(ts)
	#		Ys.append(fittableTriangle(ts,A,B,p))
	#plot(Xs,Ys,markers=rainbowMarkers("-",15)) ; sys.exit()
	#solvedParams, parm_cov = curve_fit(fittableTriangle, ts[mask==1], Ts[mask==1])
	#plot([ts,ts[mask==1],ts[mask==1]],[Ts,Ts[mask==1],fittableTriangle(ts[mask==1],*solvedParams)]) ; sys.exit()
	#slopes=np.gradient(Ts[mask==1],ts[mask==1])
	#t0=ts[mask==1][np.argmax(slopes)]
	#print(t0)
	#print(slopes)
	ts,Ts=normalizePWA(ts,Ts)
	
	#print(p,max(ts))
	#ts-=p/2
	#print(ts)

	radiifile="/".join(fileToRead.split("/")[:-1])+"/"+"radii.txt"
	if os.path.exists(radiifile):
		importRadii(radiifile)
		conditionalPrint("readPWA","radii file found: "+radiifile+" > "+str(rpump)+","+str(rprobe))
	else:
		conditionalPrint("readPWA","no radii file found: "+radiifile+", no auto-radii")				


	return ts,Ts

def normalizePWA(ts,Ts):
	# NORMALIZATION OF THE DATA: first step is to shift and wrap in time; halfway through rise should be t=0
	ts-=min(ts) ; tm=max(ts) # "normalize" times between 0 and tm
	t_low=ts[Ts<=np.mean(Ts)] ; t_high=ts[Ts>=np.mean(Ts)]	# consider the following 2 cases: data starts high, vs low
	tshift=0						# '''|    .''''         .''''|		to detect, "chop" to keep top and bottom halves
	if t_low[-1]-t_low[0] < t_high[-1]-t_high[0]: 		#    |    |      or     |    |		of the data. then check the duration of each.
		tshift=t_low[-1]				#    .____|          ___|    .____	duration(t_low)<duration(t_high) says former
	else:							#					case, so to start halfway up the rise, we'll 
		tshift=t_high[0]				# select the last point of t_low to shift by. latter case, select first point of t_high
	ts-=tshift
#	#print(ts[Ts<=np.mean(Ts)])
#	t_l=ts[Ts<=np.mean(Ts)] ; ts-=t_l[-1]		# "chop" to keep bottom half of data, shift so rise is at t=0
	iz=np.argmin(np.absolute(ts)) ; ts[:iz]+=tm	# find index of new zero, toss negatives to around to the right
	Ts=np.roll(Ts,-iz) ; ts=np.roll(ts,-iz)		# "reorder" datapoints too: tossed values to to the end of the dataset
	# NORMALIZATION OF THE DATA: the mean around 1/4 and 3/4th period. scale and shift to 1 and 0 respectively
	if normPWA:
		percents=np.linspace(0,100,len(ts))
		mint,maxt,lt=[ float(v) for v in timeNormPWA.split(",")	] # eg "25,75,10"  says use data between 20-30% cycle for max, and 70-80% as min
		mask1=np.ones(len(ts)) ; mask1[percents<mint-lt/2]=0 ; mask1[percents>mint+lt/2]=0
		mask2=np.ones(len(ts)) ; mask2[percents<maxt-lt/2]=0 ; mask2[percents>maxt+lt/2]=0
		Tmn=np.mean(Ts[mask2==1]) ; Tmx=np.mean(Ts[mask1==1])	# depending on waveform, Tmn may actually be lower than Tmx (eg, asymmetric 
		Tmn,Tmx=min(Tmn,Tmx),max(Tmn,Tmx) 			# triangle), so let's not flip the data.
		Ts-=Tmn							# shift and scale, min --> 0
		Ts/=(Tmx-Tmn)						# max --> 1
		#Ts*=(onefourth-threefourths) ; Ts+=threefourths		# shift/scale data between N3 and N1
		#plot([ts,ts[mask1==1],ts[mask2==1]],[Ts,Ts[mask1==1],Ts[mask2==1]],markers=["ko","ro","bo"]) ; sys.exit()
	if centerY:
		Ts-=np.mean(Ts)
	Ts+=yshiftPWA
	#if centerY:
	#	Ts-=np.mean(Ts)
	#plot([ts],[Ts]) ; sys.exit()
	#Ts_trim=Ts ; ts_trim=ts
	#Ts_trim=Ts[Ts<0.5] ; ts_trim=ts[Ts<0.5]
	#if fitRisePWA:
	return ts,Ts
	#Ts_trim=Ts[Ts<np.mean(Ts)] ; ts_trim=ts[Ts<np.mean(Ts)]
	#return ts_trim,Ts_trim


def dzTrimming(parameterValues,xs,ys,mask=""): # why? we'd like to be able to insert data-trimming (don't have curve_fit just subtract PWAfunc (full curve) and readPWA (full curve), but let us select only subsections of the curves. you'll call: lsqout=least_squares(dzTrimming, tuple(guesses), bounds=tuple(bnds), args=(ts,Ts,mask))
	f=func(xs,*parameterValues)
	if len(mask)==0:
		return ys-f
	return ys[mask==1]-f[mask==1]

# how do we get the same sigmas returned by curve_fit, but for least_squares? https://stackoverflow.com/questions/42388139/how-to-compute-standard-deviation-errors-with-scipy-optimize-least-squares
def lsqSigmas(res):
	J = res.jac
	cov = np.linalg.inv(J.T.dot(J))
	cov*=res["mse"]
#	U, s, Vh = np.linalg.svd(res.jac, full_matrices=False)
#	tol = np.finfo(float).eps*s[0]*max(res.jac.shape)
#	w = s > tol
#	cov = (Vh[w].T/s[w]**2) @ Vh[w]  # robust covariance matrix
#	chi2dof = np.sum(res.fun**2)/(res.fun.size - res.x.size)
#	chi2dof=np.sum
#	cov *= chi2dof
#	#perr = np.sqrt(np.diag(cov))
	perr = np.sqrt(np.diag(cov))     # 1sigma uncertainty on fitted parameters
	return perr

subtractPWA="/media/Alexandria/U Virginia/Research/DATA/Pfeifer_Thomas/2022_05_27_Fiber/250C-1000C/prOff2.txt"
def solvePWA(fileToRead,plotting="show",fileToSubtract=""):
	global mode ; mode="PWA"
	#fileToSubtract=subtractPWA
	ts,Ts=readPWA(fileToRead)#,fileToSubtract)
	
	# PREPARE THE MASK
	mask=np.zeros(len(ts)) ; percent=np.linspace(0,100,len(ts)) #; mask[int(len(ts)/2):]=1
	for pair in timeMaskPWA.split(","):
		mn,mx=pair.split(":") ; minimask=np.ones(len(ts)) # we have a mask, but for each pair of mn < keepvals < mx, we don't want to
		minimask[percent<float(mn)]=0 ; minimask[percent>float(mx)]=100 # overwrite the old mask (eg, if pairs out of order). so create
		mask[minimask==1]=1	# a temporary mask for each pair ("1s, unless outside the bounds given"), then apply that to main mask

	# AND SOLVE
	guesses=getTofitVals() ; bnds=lookupBounds() # guesses come from thermal property matrix, bounds come from ubs / lbs globals
	

	if len(tofit)>0:
		# THE OLD WAY TO SOLVE: let curve_fit compare readPWA to the output of PWAfunc
		#solvedParams, parm_cov = curvefit(PWAfunc, ts, Ts, p0=tuple(guesses),bounds=tuple(bnds)) #solve it
		#sigmas=np.sqrt(np.diag(parm_cov))
		# NEW, use timeMaskPWA (eg "0:10,50:60") as percentage bounds, create a mask, and use that to let dzTrimming() compare only a subset of the data to a subset of the function
		lsqout=least_squares(dzTrimming, tuple(guesses), bounds=tuple(bnds), args=(ts,Ts,mask))
		if 0 in mask: # resultsPlotter just plots data, and output of func(), so we pass trimmed too, xs,ys,datalabel,marker,index (in stack)
			bonusCurves=[[ts[mask==1],Ts[mask==1],"fitted","go",1]] 
		else:
			bonusCurves=''
		solvedParams=lsqout['x'] ; sigmas=[0]
	else:
		solvedParams=[] ; sigmas=[0] ; bonusCurves=''
	if len(mask)==len(ts):
		residual=resultsPlotter(fileToRead,ts,Ts,solvedParams,plotting)#,bonusCurves=bonusCurves,mask=mask)
	else:
		residual=resultsPlotter(fileToRead,ts,Ts,solvedParams,plotting,bonusCurves=bonusCurves,mask=mask)

	return solvedParams,[residual,sigmas]

sstrDelay=300e-12
# HOW DO WE HANDLE PULSED VS CW SSTR? sneakily. if the file selected has "TDTR" in the name, we read it in using the TDTR code (readpSSTR > readTDTRdata), set a global for time delay (sstrDelay), and set mode="pSSTR" (readSSTR). we then use pulsed code for the SSTR function (SSTRfunc)
def SSTRfunc(Ps,*parameterValues,store=False,addNoise=False):
	if len(parameterValues)==len(tofit): 
		setTofitVals(parameterValues)
	popGlos()
	conditionalPrint("SSTRfunc","using parameters:",pp=True)

	# Z(ω)=A1*stuff(ω)
	slope=delTomega([fm*np.pi*2])/A1 # Z(ω)=P/gamma*ΔT(ω), delTomega includes P/gamma term (A1). so if we're going to manually include it, we need to divide

	if "p" in mode:
		conditionalPrint("SSTRfunc","(pulsed)")
		ts=np.asarray([sstrDelay])
		ns=np.arange(nmin,nmax+1)			# used for summing over many frequencies
		omegas=omegaM+ns*omegaP 			# [ ωₙ ] , 1D list of ω=ωₘ+n*ωₚ values to pass into ΔT(ω). 
		delTplus=delTomega(omegas)
		#conditionalPrint("TDTRfunc","using parameters:",pp=True)
		convergeAccelerator=np.exp(-pi*ns**2./nmax**2.)	# [ ωₙ ], each n represents a frequency, Cahill eq 20+, exp(-πf²/fₘₐₓ²)
		#Z(t𝘥)= Σ ΔT(ωₘ+n*ωₚ)*exp(i*n*ωₚ*t𝘥) #from -∞ to ∞, Jiang eq 2.21/2.22 (modified) / Schmidt eq 2. 
		sumbits=delTplus[None,:]*np.exp(1j*omegaP*np.outer(ts,ns))[:,:]*convergeAccelerator[None,:] # [ t, ωₙ ]
		z=np.sum(sumbits,axis=1) # [ t ], sum over all ωₙ 
		#Note that while Jiang states Vᵢₙ(t𝘥)=½Σ(ΔT(ωₘ+n*ωₚ)+ΔT(-ωₘ+n*ωₚ))*exp(i*n*ωₚ*t𝘥) and Vₒᵤₜ(t𝘥)=-i*½Σ(ΔT(ωₘ+n*ωₚ)-ΔT(-ωₘ+n*ωₚ))*exp(i*n*ωₚ*t𝘥), simply taking the real and imaginary parts of ΣΔT(ωₘ+n*ωₚ)*exp(i*n*ωₚ*t𝘥) yields the same result. 
		#Vᵢₙ(t𝘥)  = Re(Z(ω)) , Vₒᵤₜ(t𝘥) = Im(Z(ω))							#Jiang eq 2.21/2.22 (modified) 
		#xs=z.real ; ys=z.imag
		#slope=(xs[0]**2+ys[0]**2)**(1/2)/A1
		slope=z/A1
	else:
		conditionalPrint("SSTRfunc","(CW)")

	Zs=Ps/gamma*slope
	#for P in Ps:
	#	A1=P/gamma		# BUG WITH THIS STRATEGY: delTomega > Gkomega works, but for non-zero depths,delTomega > biMatrix > popGlos > updates A1
	#	z=delTomega([fm*np.pi*2])
	#	Zs.append(z[0])
	z=np.asarray(Zs) ; xs=z.real ; ys=z.imag
	M=(xs**2.+ys**2.)**.5
	if addNoise:
		M*=noise(Ps,addNoise)
	if store:
		ones=np.ones(len(Ps)) ; zeros=np.zeros(len(Ps))
		saveGen([Ps,zeros,zeros,zeros,M,zeros,zeros,zeros,ones],store) # Fiber SSTR is: PuX.append(l[0]) ; PuY.append(l[2]) ; PrX.append(l[4]) ; PrY.append(l[6]) ; Aux.append(l[8])
	return M

# HOW DO WE HANDLE PULSED VS CW SSTR? sneakily. if the file selected has "TDTR" in the name, we read it in using the TDTR code (readpSSTR > readTDTRdata), set a global for time delay (sstrDelay), and set mode="pSSTR" (readSSTR). we then use pulsed code for the SSTR function (SSTRfunc)
def solveSSTR(fileToRead='',plotting="show",P='',M=''): # TODO: curve_fit has the "sigma" option, to allow passing error bars in y for each datapoint!
	global mode ; mode="SSTR"
	conditionalPrint("solveSSTR","importing file:"+fileToRead)
	#FILE READING
	if len(P)==0:
		P,M=readSSTR(fileToRead)
	#FITTING
	guesses=getTofitVals() ; bnds=lookupBounds() # guesses come from thermal property matrix, bounds come from ubs / lbs globals
	if len(tofit)!=0:
		solvedParams, parm_cov = curvefit(SSTRfunc, P, M, p0=tuple(guesses),bounds=tuple(bnds)) #solve it
		sigmas=np.sqrt(np.diag(parm_cov))
	else:
		solvedParams=[] ; sigmas=[]
	conditionalPrint("solveSSTR","solved:"+str(solvedParams)+","+str(sigmas))

	residual=resultsPlotter(fileToRead,P,M,solvedParams,plotting)
	mode="SSTR" # readSSTR() may find a pSSTR file and update mode to pSSTR, so we'll reset it here
	return solvedParams,[residual,sigmas]

# This is intended as a generalized results plotting function, usable for TDTR, SSTR, FDTR, PWA. give us your results, we'll re-run fitting to generate the curve, detect mode to set labels correctly, and plot it all
def resultsPlotter(fileToRead,xs,data,solvedParams,plotting,bonusCurves='',mask=''): # bonusCurves should be quadruplet [xs1,xs2,xs3],[ys1,ys2,ys3],[dlb1,dlb2,dlb3],[mkr1,mkr2,mkr3],{optional index}
	conditionalPrint("resultsPlotter: func",func)
	ys=func(xs,*solvedParams) #; Ms-=np.mean(Ms)
	conditionalPrint("resultsPlotter: ys", ys)
	if mode=="PWAX":
		l=len(data) ; i=[ int(l*p) for p in [.15,.45,.65,.95]] #/``\,,
		data=list(data) ; ys=list(ys) ; xs=list(xs)
		data=data[:i[0]]+data[i[1]:i[2]]+data[i[3]:]
		ys =   ys[:i[0]]+  ys[i[1]:i[2]]+  ys[i[3]:]
		xs =   xs[:i[0]]+  xs[i[1]:i[2]]+  xs[i[3]:]
		data=np.asarray(data) ; ys=np.asarray(ys) ; xs=np.asarray(xs)

	if len(mask)>0:
		residuals=error(data[mask==1],ys[mask==1])
	else:
		residuals=error(data,ys)
	if plotting=="none":
		return residuals
	dlbs=["data" , ",".join([p+"="+str(scientificNotation(sp,2)) for p,sp in zip(tofit,solvedParams)]) ]
	Xs,Ys=[xs,xs],[data,ys] ; mkrs=["k.","r-"]
	if plusMinus>0:
		for sign in [-1,1]:
			SP=[ p if i>0 else p*(1+sign*plusMinus/100) for i,p in enumerate(solvedParams) ]
			ys=func(xs,*SP)
			Xs.append(xs) ; Ys.append(ys) ; dlbs.append("") ; mkrs.append("k:")
	xlabel={"TDTR":"Time (s)","FDTR":"Frequency (Hz)","SSTR":"Pump power (mW)","pSSTR":"Pump power (mW)","PWA":"Time (s)"}[mode]
	TDTRylabel={"R":"Ratio (-X/Y)","M":"Mag (V)","X":"X (V)","Y":"Y (V)","P":"Phase (rad)"}[fitting]
	ylabel={"TDTR":TDTRylabel,"FDTR":TDTRylabel,"SSTR":"Probe response (mW)","pSSTR":"Probe response (mW)","PWA":"Temperature (K)"}[mode]
	#print(traceback.format_stack())
	title=fileToRead.split("/")[-1]+",R^2 = "+str(np.round(residuals*100,2))+"%" ; filename=figFile(fileToRead,plotting)

	stack=traceback.format_stack()
	useLast = True in [ "gui" in e for e in stack ] # if this was called by the gui.py code, then use the previous matplotlib object to plot

	#print("TDTR_fitting > resultsPlotter > useLast",useLast)
	#for curve in bonusCurves:
	#	xs,ys,dlb,mkr=curve[:4]
	#	if len(curve)>4:
	#		i=curve[4]
	#		Xs.insert(i,xs) ; Ys.insert(i,ys) ; dlbs.insert(i,dlb) ; mkrs.insert(i,mkr)
	#	else:
	#		Xs.append(xs) ; Ys.append(ys) ; dlbs.append(dlb) ; mkrs.append(mkr)
	scx={True:"log",False:"linear"}[mode=="FDTR"]
	if len(mask)>0:
		Xs.insert(1,xs[mask==1]) ; Ys.insert(1,data[mask==1]) ; dlbs.insert(1,"masked") ; mkrs.insert(1,"go")
	lplot(Xs, Ys, xlabel, ylabel, title=title, filename=filename, labels=dlbs, markers=mkrs, useLast=useLast, xscale=scx)
	return residuals

def figFile(fileToRead,plotting,subfolder="pics"):
	fileToRead=fileToRead.replace("\\","/")
	direc=fileToRead.split("/")[:-1]+[callingScript,subfolder,fileToRead.split("/")[-1]+".png"]
	figFile={"show":"","save":"/".join(direc)}[plotting]
	if len(figFile)>0:
		os.makedirs("/".join(direc[:-1]),exist_ok=True)
	return figFile

from scipy.optimize import least_squares #solve
from scipy.optimize import brute
from scipy.optimize import minimize

"""
def solveSimultaneous(listOfFiles,plotting="show"): #just as solve() allows scipy.curve_fit to call into a helper TDTRfunc (which takes the parameters, handles setup, and returns resulting decay curve), we allow nlinsq to call into a different helper function which returns the dz for all the data sets (dz being how far off a given curve is). nlinsq is then trying to minimize this "error". Solving in this manner (minimizing error rather than using curve-fit) allows us to accept a list of files instead of one single file, and fit all decay curves simultaneously. "plotting" options include: show, save, none
	data=[];ts=[];fms=[]
	for fileToRead in listOfFiles: #import all files, reading in varying processing parameters (eg, modulation frequency). TODO: expand for spot size differences scan to scan. until then, last file imported rules. 
		ts,dataFromFile=readTDTR(fileToRead)
		fms.append(fm) #readTDTR set the modulation frequency as it read in the data points! so we read it here. 
		data.append(dataFromFile)
	guesses=getTofitVals() #handle guesses
	bnds=lookupBounds() #; print(bnds)
	lsqout=least_squares(solveSimultHelper, tuple(guesses), bounds=tuple(bnds), args=(data,ts,fms))
	residuals=solveSimultPlotting(lsqout,ts,data,listOfFiles,plotting)
	return lsqout["x"],[max(residuals),0] #MSE returned
	#return brute(solveSimultHelper,bnds,Ns=100)

#given a list with lists of TDTR datapoints, and a list of modulation frequencies for each (most commonly varied), return flattened function(t)-data(t) (this goes into least_squares)
def solveSimultHelper(parameterValues,listOfDecays,ts,fms): #[[file1mag1,file1mag2,...],[file2mag1...]],[t1,t2,t2, (shared by all files)], [freq1,fre2..], [Kz2val,G1val... (matching tofit)] #TODO: assumes same pump probe sizes too. should those be passable as well?
	incrementCounter("solveSimultHelper")
	results=np.zeros((len(fms),len(ts)))
	global fm
	for i in range(0,len(fms)):
		fm=fms[i]
		result=TDTRfunc(ts,*parameterValues)
		results[i,:]=result[:]
	dz=listOfDecays-results
	return dz.flatten()


import traceback # use this to use brute iaoi called by measureContour1Axis
# What is simultaneous fitting? it may be common procedure to run multiple measurements (e.g. TDTR collected at multiple modulation frequencies) or measurement types (TDTR and SSTR), and iteratively fit back and forth between the two, to fit for more unknowns than would be typical. ss2 saves you the effort of manual iteration, by solving the multiple measurements (or types) at once. 
# Simultaneous fitting works the same as normal fitting ( 1. read x and y datapoints for the curve out of a data file, also reading in various other parameters like modulation frequency, spot sizes, etc, 2. have a separate function which, given the fitted-for thermal parameters, generates the curve (and possibly also takes the data and subtracts the curve and model), 3) call curve_fit or least_squares to minimize the difference between data and curves ). However, for simultaneous fitting (multi-frequency, or multi-technique), we must save off those read-in parameters (fm, rpu, rpr), and be sure to call the approriate file-reading function, and curve-generation function. for data, we keep a nested ragged list, and we iterate through that list for curve generation. TODO: check testing41.py for how to use TODO BEWARE: an SSTR+TDTR pair, if SSTR has 10 points and TDTR has 30, will weight SSTR as 1/4th of TDTR. this is probably not what you want.
ss2Types=[] ; ss2fms=[] ; ss2rpus=[] ; ss2rprs=[] ; magicMods={} # TODO ss2Types is a really really shitty hack in order to make measureContours1Axis work without needing to cascade all args through to mc1aWorker [ setVar("ss2Types",types) ; ,measureContour1Axis(files,"Kz2",solveFunc=ss2) ]. ss2fms,rpus,rprs, is another really really shitty hack in order to make predictUncert() work, since it needs to generate fake datafiles to run measureContour1Axis() across, and we want to support multifrequency and so on.
# HOW DOES MAGICMODS WORK, IF THE USER WERE TO SET IT BEFOREHAND? dict of vars (by name), where each var's value is a list of things to set, for each file. e.g. magic={ "tp":[tpforfile1,tpforfile2...] , "fitting":[...] }
def ss2_v01(listOfFiles,listOfTypes="",plotting="show",useBrute=False,refit=True): # TODO refit does nothing right now, but we should implement it! see the refit functionality for solve()! 
	global magicMods
	conditionalPrint("ss2","preparing to run: "+" , ".join([str(l) for l in [listOfFiles,listOfTypes,ss2Types,ss2fms,ss2rpus,ss2rprs,magicMods]]))
	# settableVars: passed to ss2h, containing keys (vars) and lists of values (one per datafile), to be looped through, calling setVar(var,val)
	sV={ "fm":[] , "rpump":[] , "rprobe":[] }
	foundMM=["magicModifiers" in f for f in listOfFiles]		# some issues with file ordering from gui.py, so maybe False,False,True,False
	if True in foundMM:
		MM=listOfFiles[foundMM.index(True)]			# get the filename (including path) of magicMod file
		listOfFiles=[f for f,t in zip(listOfFiles,foundMM) if not t ] # for each file/Truth pair, only keep those files that aren't magicMods
		lines=open(MM,'r').readlines()
		for l in lines:						# eg 'tp=[C_Al,K_Al,80e-9,"Kz"],[90e6],[C_Si,K_Si,1,"Kz"]'
			#print(l)					# and you must have one entry per datafile
			ls=l.split(";")					# but you can have multiple glos to set per line, ";" delimited
			for l in ls:
				l=l.strip()				# 'tp=[[...]] ; fitting="R"' -> ' fitting="R"' -> 'fitting="R"' -> ['fitting',"R"]
				l=l.split("=")			
				var,val=l
				#print(var,val)
				if var not in sV.keys():
					sV[var]=[]
				sV[var].append(eval(val))
	else:
		# should also allow preserve any magicMods that are already in the global (allow user to populate magicMods themselves via setVar instead of requiring a magicMods file. ALSO, this is needed in order for predictUncert to work with magicMods (it'll create the fake datafiles abiding by magicMods, then we'll wipe 'em out here. not good) TODO there's definitely going to be some collisions happening here. run fitting with magicMods then without, we'll keep the old magicMods still in use! or, use different 
		for k in magicMods.keys():
			if len(magicMods[k])==len(listOfFiles):
				sV[k]=magicMods[k]

	# also need to update modes. 3 ways to get it: listOfTypes passed in, ss2Types set beforehand, or magicModifiers text file which ends up in sV already
	if len(listOfTypes)==len(listOfFiles):		# passed listOfTypes overrides all other options
		sV["mode"]=listOfTypes
	elif "mode" in sV.keys():			# next option: mode can be included in magicModifiers
		listOfTypes=sV["mode"]
	elif len(listOfTypes)==0 and len(ss2Types)==0:	# no mode given, default to whatever's in global "mode" for all
		listOfTypes=[mode]*len(listOfFiles)
	elif len(listOfTypes)==0:			# global (ss2Types) populated but nothing passed (listOfTypes)
		listOfTypes=ss2Types
	sV["mode"]=listOfTypes
		

	magicMods=sV ; conditionalPrint("sV",str(sV))
	# READ IN THE (various types of) DATA
	conditionalPrint("ss2","running for files/types: "+"; ".join([f.split("/")[-1]+","+t for f,t in zip(listOfFiles,listOfTypes)]),pp=True)
	Xs=[] ; Ys=[] ; autoFailed_local=False ; global autoFailed # local copy of autoFailed is used to record if *any* of our files autofailed
	for i,f,t in zip(range(len(listOfFiles)),listOfFiles,listOfTypes):
		fun={"TDTR":readTDTR,"SSTR":readSSTR,"PWA":readPWA,"FDTR":readFDTR,"pSSTR":readSSTR}[t]
		#print(i,f,t)
		for var in sV.keys():		# need to handle magicmods for file import too! eg, fitting="R",fitting="M"
			if var in ["fm","rpump","rprobe"]:
				continue
			#print(var)
			conditionalPrint("ss2",str(var)+str(sV)+str(i))
			setVar(var,sV[var][i])
		xs,ys=fun(f)
	#	if t=="TDTR": # this is a prototype to explore the weighting discrepency between, say, SSTR and TDTR. "scale up" TDTR data and watch
	#		ys=ys*1000	# your ss2 give different results (whether SSTR vs TDTR get better or worse or equally-bad fits)
		Xs.append(xs) ; Ys.append(ys)
		sV["fm"].append(fm) ; sV["rpump"].append(rpump) ; sV["rprobe"].append(rprobe) # TODO need more intelligent additions. plausible the user put frequencies or radii in magicModifiers.txt
		#fms.append(fm) ; rpus.append(rpump) ; rprs.append(rprobe)
		if autoFailed:
			autoFailed_local=True
	autoFailed=autoFailed_local 
#	#print(fms,rpus,rprs)
	conditionalPrint("ss2","settableVars:"+str(sV))
	if len(tofit)>0:
		# ITERATIVELY, CALL EACH APPROPRIATE MODEL FUNCTION, MINIMIZING DIFFERENCE
		guesses=getTofitVals() #handle guesses
		bnds=lookupBounds() ; conditionalPrint("ss2","guesses,bounds:"+str(guesses)+","+str(bnds))
		#lsqout=least_squares(ss2h_equalweighting, tuple(guesses), bounds=tuple(bnds), args=(Xs,Ys,sV,listOfTypes))
		#if len(tofit)>2:
		#print(traceback.format_stack())
		#truth = [ "mc1aWorker" in line for line in traceback.format_stack() ][:-1] # gotcha alert! last line of call stack is *this* line! which includes our search string! haha! 
		#print(truth)
		#if True not in truth:
		#if not useBrute:
		#	conditionalPrint("ss2","solving using function: minimize")
		stack=traceback.format_stack()
		#if True in [ "mc1aWorker" in line for line in stack ]: # TODO why just warn, instead of, say, using brute? because brute is irresponsibly slow. i'd rather not nuke the performance on "good enough"? warn, and have the user use 3D contours if they need better accuracy
		#	print("SS2+CONTOURS WARNING: USING THIS FOR CONTOURS IS DANGEROUS! LOCAL MINIMA ARE DIFFICULT TO AVOID! USE FULL ND CONTOURS INSTEAD")
		if useBrute:
			conditionalPrint("ss2","solving using function: brute")
			lsqout=brute(ss2h_equalweighting, tuple(list(zip(*bnds))), args=(Xs,Ys,sV,listOfTypes)) ; lsqout={"x":lsqout}
		else:
			#x_scale=[ 1/factDict[p[:-1]] for p in tofit ] #; print(x_scale)
			#lsqout=minimize(ss2h_equalweighting, tuple(guesses), bounds=tuple(list(zip(*bnds))), args=(Xs,Ys,sV,listOfTypes) )
			#lsqout=least_squares(ss2h_equalweighting, tuple(guesses), bounds=tuple(bnds), args=(Xs,Ys,sV,listOfTypes),x_scale=x_scale)
			#dy=ss2h(lsqout['x'],Xs,Ys,sV,listOfTypes,flatten=True)
			#lsqout["mse"]=MSE(dy,np.zeros(len(dy)))
			#sigmas=lsqSigmas(lsqout)
			# HERE IS HOW WE TRICK CURVE_FIT INTO SOLVING OUR MULTIPLE CURVES. xs is a dummy list the length of the number of datapoints (important for correct error for sigmas i think), ss2hcf just flattens the Fᵢ-Yᵢ, so the curves we're trying to get to match are, all measurements concatenated together, and zeros
			xs=[]
			for x in Xs:
				xs+=list(x)
			def ss2hcf(xs,*parameterValues):
				return ss2h(parameterValues,Xs,Ys,sV,listOfTypes)
			solvedParams,parm_cov=curvefit(ss2hcf,xs,np.zeros(len(xs)),p0=tuple(guesses),bounds=tuple(bnds))
			sigmas=np.sqrt(np.diag(parm_cov))
			lsqout={"x":solvedParams}
	else:
		lsqout={"x":[]} ; sigmas=[]
	#conditionalPrint("ss2","found:"+str(lsqout["x"])+","+str(sigmas)+","+str(residuals))
	# PLOTTING THE RESULTS
	Ys_m=ss2h(lsqout["x"],Xs,Ys,sV,listOfTypes,flatten=False)
	#Ys_mp=ss2h(lsqout["x"]*np.asarray([1.2,1]),Xs,Ys,fms,rpus,rprs,listOfTypes,flatten=False)
	#Ys_mm=ss2h(lsqout["x"]*np.asarray([.8,1]),Xs,Ys,fms,rpus,rprs,listOfTypes,flatten=False)
	residuals=[error(y,ym) for y,ym in zip(Ys,Ys_m)]
	if plotting in ["show","save"]:
	#if True:
		Xp=[] ; Yp=[] ; dlbs=[] ; mkrs=[]
		for i in range(len(Xs)):
			dlb=listOfFiles[i].split("/")[-1]+","+listOfTypes[i]
			x=np.asarray(Xs[i]) ; y=np.asarray(Ys[i]) ; ym=np.asarray(Ys_m[i])
			#ymp=np.asarray(Ys_mp[i]) ; ymm=np.asarray(Ys_mm[i])
			if len(set(listOfTypes))>1:
				xmax=max(x) ; ymax=max(abs(y))
				x/=xmax ; ym/=ymax ; y/=ymax
				#ymp/=ymax ; ymm/=ymax
			Xp.append(x) ; Yp.append(y) ; mkrs.append("o") ; dlbs.append(dlb)
			Xp.append(x) ; Yp.append(ym) ; mkrs.append("-") ; dlbs.append(dlb)
			#Xp.append(x) ; Yp.append(ymp) ; mkrs.append("r:") ; dlbs.append("")
			#Xp.append(x) ; Yp.append(ymm) ; mkrs.append("r:") ; dlbs.append("")
		valStr=", ".join([p+"="+scientificNotation(v,2) for p,v in zip(tofit,lsqout["x"])])
		resStr="R^2 = "+",".join([scientificNotation(r,2) for r in residuals])
		title=valStr+", "+resStr
		#figFile={"show":"","save":"/".join(listOfFiles[0].split("/")[:-1])+"/"+callingScript+"/pics/"+listOfFiles[0].split("/")[-1]+".png"}[plotting]
		lplot(Xp, Yp, "-", "-", title=title, markers=mkrs, labels=dlbs, filename=figFile(listOfFiles[0],plotting))
		#plt.legend(loc="upper right",frameon=False)
		#plt.show()
	conditionalPrint("ss2","found:"+str(lsqout["x"])+","+str(sigmas)+","+str(residuals))
	return lsqout["x"],[max(residuals),0]

# What happens when you try to simultaneous-fit TDTR+SSTR? y values for a typical TDTR scan analyzing ratio are 1-6 (unitless), whereas y values for a typical SSTR scan can be in the thousands (semi-arbitrary uV pump and uV probe, divide by aux in V). least_squares is not scaling-agnostic (double Fᵢ and Yᵢ both, MSE doubles), which means you've applied a 1000x weighting to the SSTR scan! the "best fit" ends up being "a good fit for SSTR, and a shoddy fit for TDTR". This has greater rammifications than just for fitting: if our uncertainty is calculated (in part) by exploring the range over which a parameter can be set and a good fit can still be found (by tweaking the other fitted params), then this "shoddy fit for TDTR" ends up artificially narrowing your uncertainty: there is a wider range of values where semi-poor fits can be found for both. 
# TODO OH NO, THIS STILL DOESN'T RESOLVE THE ISSUE. WORKS FOR 2 MEASUREMENTS SIMULTANEOUSLY, BUT NOT 3! EG, MIN LSQ FOR 3 RESIDUALS MIGHT BE WHERE ONE RESIDUAL IS ABOVE THRESHOLD, EVEN THOUGH AN ALL-3-BELOW-THRESHOLD SOLUTION EXISTS
# looks like just returning a single max of the residuals works fine (and i think this is "correct": if one measurement has a good fit and another has a bad fit, we don't really care about the good fitting measurement. that bad fit is bad). 
def ss2h_equalweighting(parameterValues,Xs,Ys,sV,listOfTypes):
	#print(Xs,Ys,parameterValues,listOfTypes,sV,traceback.format_stack())
	Ys_m=ss2h(parameterValues,Xs,Ys,sV,listOfTypes,flatten=False)
	#tosum=[]
	#for Y,Ym in zip(Ys,Ys_m):
	#	for y,ym in zip(Y,Ym):
	#		tosum.append( (y-ym)**2 )
	#return np.sqrt(np.sum(tosum))
	residuals=[ error(d,f) for d,f in zip(Ys,Ys_m) ]
	#diffs=[]
	#for i,r1 in enumerate(residuals):
	#	for j,r2 in enumerate(residuals):
	#		if i==j:
	#			continue
	#		diffs.append((r1-r2)*10)
	#residuals=residuals+diffs
	#print(residuals)
	return residuals # TODO when using ss2 for contours1Axis, it's possible for the "best fit" to be "one scan has a perfect fit, the other has a shoddy fit" (which might appear outside the threshold), even when "both scans have just okay fits" exists (within threshold). You can try to address by, for example, returning max(residuals) here, BUT, that leads to local minima in basic solving (fails to converge to the right solution, because no matter which way it perturbs the fitting params, the max goes up. you can see this with basic mfSSTR, 2 frequency, eg, testscripts/DATA/2022_02_15_Fiber/. starting out, if 1kHz is the "worse of the two", then the solver "learns" it's insensitive to G1, then never tries changing G1 later once 10MHz becomes the worse. I don't know what to do about this, aside from just say that measureContours1Axis flat out doesn't work
	#residuals=[ r if r<=2.5 else r*2 for r in residuals ]
	#return np.asarray(residuals) #)**2-1
	#return [max(residuals)]
	#return max(residuals)
	#return np.sum(np.asarray(residuals)**2)**(1/2)
multipliers={}
def ss2h(parameterValues,Xs,Ys,sV,listOfTypes,flatten=True):
	conditionalPrint("ss2h","trying with parameters:"+str(parameterValues))
	global fm,rprobe,rpump

	Ys_m=[]
	for i in range(len(Xs)):
		for var in sV.keys():
			setVar(var,sV[var][i])
	
		#print(parameterValues,tofit)
		if len(parameterValues)==len(tofit): # note: set fitting params *afteR* settableVars is handled. eg, what if we're fitting for spot sizes? those were read from the file headers (possibly) by ss2, passed in through sV, and we don't want those to override the fitted params (same applies for magicModifiers. plausible to put the whole thermal properties matrix in magicModifiers.txt, but we still want to fit for one of the thermal properties. so fitted free param needs to override imported).
			setTofitVals(parameterValues)

		for var in multipliers.keys():
			m=multipliers[var][i]
			setVar(var,m*getVar(var))
		

		fun={"TDTR":TDTRfunc,"SSTR":SSTRfunc,"pSSTR":SSTRfunc,"PWA":PWAfunc,"FDTR":FDTRfunc}[listOfTypes[i]]
		ys=fun(Xs[i])
		Ys_m.append(ys)
	if flatten:
		dys=[]
		for y,ym in zip(Ys,Ys_m):
			y=np.asarray(y) ; ym=np.asarray(ym)
			dys+=list(y-ym)
		return dys
	else:
		return Ys_m
"""

def processMagic(filename,settables={}):
	conditionalPrint("processMagic","found magic file: "+filename) 
	lines=open(filename).readlines()
	for l in lines:					# N lines for N files going into ss2 and friends
		pieces=l.split(";")			# multiple vars (we'll call setVars) per line allowed
		for p in pieces:
			k,v=p.strip().split("=")	# using "var=value" to define them
			if k not in settables.keys():
				settables[k]=[]
			settables[k].append(eval(v))
	conditionalPrint("processMagic","found settables: "+str(settables)) 

# (predictUncert > measureContour1Axis > parallel > mc1aWorker) > solve or ss2 > ss3h
# in versions >= 0.152, predictUncert takes settables dict instead of making dumbass assumptions based on ss2Glos. measureContour1Axis takes solveFunc dict too, which cascades ss2Types (mode, via solveFunc["kwargs"]=dict, passed into ss2's kwarg "listOfTypes") through. the new ss2 also must now manually infer fm,rpu,rpr from the files themselves! (TODO, this may not be good). ss2 then populates dict "settables" which is handled by ss3h
# why the re-write of this entire chain? it makes preductUncert much *much* cleaner when trying your "atypical" stuff like mfSSTR or SSTR+TDTR (predictUncert's "settables" ought to handle it all)
def ss2(listOfFiles,listOfTypes="",plotting="show",settables="",refit=True,hybridMethod="both"):
	conditionalPrint("ss2","received: "+str(listOfFiles)+", "+str(listOfTypes)+", "+str(settables))
	datafile=combinedFilename([ f for f in listOfFiles if "magic" not in f ])
	r,e=readResultFile(datafile)
	if not refit and r is not None:
		#print("REFIT=True")
		return r,e

	if len(settables)==0:
		settables={}

	# handle "magic" files
	for f in listOfFiles:
		if "magic" in f:
			processMagic(f,settables)	# populated "settables", which means mode from magicMods overrides all! 
	listOfFiles=[ f for f in listOfFiles if "magic" not in f ]
	# handle listOfTypes (overrides settables["mode"] if populated)
	if "mode" not in settables.keys():
		settables["mode"]=[""]*len(listOfFiles)
	for i,f in enumerate(listOfFiles):
		m=mode					# mode either comes from: the current mode
		if i<len(listOfTypes):
			m=listOfTypes[i]		# or listOfTypes (if populated)
		if len(settables["mode"][i])==4:
			m=settables["mode"][i]		# or settables["mode"] (if pre-populated)
		settables["mode"][i]=m			# (direct setting, instead of appending, helps avoid settables["mode"] getting out of control
	# read in files:
	Xs=[] ; Ys=[] #; settables={"fm":[],"rpump":[],"rprobe":[],"mode":[]}
	settables["fm"]=[] ; settables["rpump"]=[] ; settables["rprobe"]=[]
	for i,f in enumerate(listOfFiles):
		for k in settables.keys():	# update mode before reading in file. ALSO update things like "fitting=M" 
			if k in ["fm","rpump","rprobe"]:
				continue
			conditionalPrint("ss2","calling setVar("+k+","+str(settables[k][i])+")")
			setVar(k,settables[k][i])
		fun={"TDTR":readTDTR,"SSTR":readSSTR,"PWA":readPWA,"FDTR":readFDTR,"pSSTR":readSSTR}[mode]
		xs,ys=fun(f) ; Xs.append(xs) ; Ys.append(ys)
		settables["fm"].append(fm) ; settables["rpump"].append(rpump) ; settables["rprobe"].append(rprobe) #; settables["mode"].append(mode)
	if not autofm:
		del settables["fm"]
	if not autorpu:
		del settables["rpump"]
	if not autorpr:
		del settables["rprobe"]
	# fitting algorithm
	conditionalPrint("ss2","preparing to run: "+str(listOfFiles)+" , "+str(settables)+" , "+str(tofit))
	#printtp()
	if len(tofit)>0:
		# ITERATIVELY, CALL EACH APPROPRIATE MODEL FUNCTION, MINIMIZING DIFFERENCE
		guesses=getTofitVals() #handle guesses
		bnds=lookupBounds()
		conditionalPrint("ss2","guesses,bounds: "+str(guesses)+","+str(bnds))
		# TWO METHODOLOGIES:
		# 1) construct a 1D composite dataset (use "sum(listOfLists,[])" to "flatten" each list of datasets), and then run curve_fit on that. this, in effect, means we're using the metric: >>> minimize(residual(compositeDataset)) <<< . be careful, because you may inadvertently overweight one dataset (e.g. a 10% deviation between datapoint and function for TDTR (where the nominal ratio or magnitude is 3), will be a dz of 0.3. whereas a mere 1% deviation for SSTR (where the nominal magnitude may be 300) will be a dz of 3. meaning, unscaled, your SSTR will super out-weigh your TDTR). 
		# 2) calculate a residual for each dataset individually, and consider the max residual available. this, in effect, means we're using the  the metric: >>> minimize(max(listOfResiduals)) <<< . be careful, the minimize procedure may have difficulty if your parameter space is not smooth. in other words, whichever dataset has the worst residual is given 100% weighting (for the moment) to push the fit. or visualized, this is different region of the 2D contour surface plot where different surfaces meet up. 
		# TODO problem: method 2 does a better job for TDTR+SSTR hybrid fitting and exploring the uncertainty via contours. method 1 does a better job (and in fact, 2 flat out doesn't work) for calspots (multi-sample SSTR to fit for rpu and gamma). why???
		# ooky spooky 2023-10-31 update: it looks like "minmaxres" hybridMethod actually does work fine for calspots (and actually returns a solution with lower max residual, ie, a "better" solution), it's just more sensitive to the guesses (e.g. start with gamma=1e4 rpu=10 and "curvecombo" works but "minmaxres" fails (nominal values 2.6e4 and 1.41)). "both" also works, because it means curvecombo overcomes our bad guesses and minmaxres finishes the job. And even better, we can play around with scipy.optimize.minimize methods https://docs.scipy.org/doc/scipy/reference/optimize.html#local-multivariate-optimization and see if any are better at dodging bad-guess issues. Nelder-Mead seems good (and this: https://stackoverflow.com/questions/58925576/how-to-choose-proper-method-for-scipy-optimize-minimize says it is suitable for "noisy" functions)
		# 2023-11-01 update: looking at fast contours (which sweep through one parameter and then run solve() or ss2() on the remaining parameters) for a situation which we *know* is unbounded (e.g. two-frequency SSTR but fitting for 3 unknowns), we can see the instability in minmaxres (and see that our "fix" using scipy.optimize.minimize's Nelder-Mead method is imperfect) by noticing discontinuities (and non-zero residuals) in the fast-contour. for this reason, we switch to "both" as our default hybridMethod
		conditionalPrint("ss2","running hybridMethod = "+hybridMethod)
		# METHODOLOGY 1: minimize(residual(compositeDataset))
		if hybridMethod=="curvecombo":
			Xs_flat=sum([ list(x) for x in Xs ],[]) ; 
			ss2Scaling=[ [np.mean(y)]*len(y) for y in Ys ] ; ss2Scaling=sum(ss2Scaling,[])
			def ss3hwrapped(xs,*parameterValues):
				dy=ss3h(parameterValues,Xs,Ys,settables)
				return np.asarray(dy)/ss2Scaling

			solvedParams,parm_cov=curvefit(ss3hwrapped,Xs_flat,np.zeros(len(Xs_flat)),p0=tuple(guesses),bounds=tuple(bnds))
			sigmas=np.sqrt(np.diag(parm_cov))

		# METHODOLOGY 2: minimize(max(listOfResiduals))	
		elif hybridMethod=="minmaxres":
			def ss3hwrapped2(parameterValues):
				Ym=ss3h(parameterValues,Xs,Ys,settables,flatten=False)
				residuals=[ error(y,ym) for y,ym in zip(Ys,Ym) ]
				conditionalPrint("ss3hwrapped","tried: "+str(parameterValues)+", found: "+str(residuals))
				#Xp=[Xs
				#plot([Xs],
				return max(residuals)

			#bnds=[ [ bnds[0][i] , bnds[1][i] ] for i in range(len(tofit)) ]
			#lsqout=minimize(ss3hwrapped2, x0=tuple(guesses), bounds=tuple(bnds) )
			lsqout=minimizee(ss3hwrapped2, tuple(guesses), tuple(bnds) , method="Nelder-Mead")

			solvedParams=lsqout['x']
			sigmas=np.zeros(len(tofit))
		elif hybridMethod=="both":
			ss2(listOfFiles,listOfTypes,"none",settables,refit,hybridMethod="curvecombo")
			return ss2(listOfFiles,listOfTypes,plotting,settables,refit,hybridMethod="minmaxres")
	else:
		solvedParams=[] ; sigmas=[]

	# calculate residuals based on what we found
	Ys_model=ss3h(solvedParams,Xs,Ys,settables,flatten=False)
	residuals=[ error(y,ym) for y,ym in zip(Ys,Ys_model) ]

#	flatYs=[] ; flatYm=[] ; ss2Scaling=[]
#	for y,ym,m in zip(Ys,Ys_model,settables["mode"]):
#		flatYs.append(list(y)) ; flatYm.append(list(ym))
#		ss2Scaling.append([np.mean(y)]*len(y))
#		if m=="TDTR":
#			ss2Scaling[-1]=[ v*10 for v in ss2Scaling[-1] ]
#	flatYs=sum(flatYs,[]) ; flatYm=sum(flatYm,[]) ; ss2Scaling=sum(ss2Scaling,[])
#	flatYs=np.asarray(flatYs) ; flatYm=np.asarray(flatYm) ; ss2Scaling=np.asarray(ss2Scaling)
#
#	flatYs/=ss2Scaling ; flatYm/=ss2Scaling
#	singleresidual=error(flatYs,flatYm)

	conditionalPrint("ss2","found: "+str(solvedParams)+","+str(residuals)+" max:"+str(max(residuals)))

	writeResultFile(datafile,solvedParams,[max(residuals),sigmas])

	if plotting in ["show","save"]:
		dlbs=[ f.split("/")[-1]+","+settables["mode"][i] for i,f in enumerate(listOfFiles) ]
		Xp=Xs+Xs ; Yp=Ys+Ys_model ; mkrs=["o"]*len(listOfFiles)+["-"]*len(listOfFiles)
		maxy=[ max(y) for y in Yp ] ; maxx=[ max(x) for x in Xp ] ; n=int(len(Xp)/2)
		#print(maxy,n)
		if len(set(settables["mode"]))!=1: # if only one type of datafile, don't normalize (mfSSTR, mfTDTR, etc)
			Xp=[ x/maxx[i%n] for i,x in enumerate(Xp) ] ; Yp=[ y/maxy[i%n] for i,y in enumerate(Yp) ]
		#Yp=[ y/maxy for y in Yp ]
		valStr=", ".join([p+"="+scientificNotation(v,2) for p,v in zip(tofit,solvedParams)])
		resStr="R^2 = "+",".join([scientificNotation(r,2) for r in residuals])
		title=valStr+", "+resStr
		#figFile={"show":"","save":"/".join(listOfFiles[0].split("/")[:-1])+"/"+callingScript+"/pics/"+listOfFiles[0].split("/")[-1]+".png"}[plotting]
		fout=figFile(listOfFiles[0],plotting)
		lplot(Xp, Yp, "-", "-", title=title, markers=mkrs, labels=dlbs, filename=fout)



	return solvedParams,[max(residuals),sigmas]
	#return solvedParams,[singleresidual,sigmas]

def ss3h(parameterValues,Xs,Ys,settables,flatten=True):
	Ys_model=[]
	for i in range(len(Xs)):
		for k in settables.keys():
			setVar(k,settables[k][i],warning=False) # rpu/rpr/fm end up in settables for ss2, but we don't need to warn the user
		fun={"TDTR":TDTRfunc,"SSTR":SSTRfunc,"pSSTR":SSTRfunc,"PWA":PWAfunc,"FDTR":FDTRfunc}[mode]
		conditionalPrint("ss3h",str(i)+": "+str(fun)+", "+str([ [k,settables[k][i]] for k in settables.keys() ]))
		ys=fun(Xs[i],*parameterValues)
		Ys_model.append(ys)
	if flatten:
		Ys_model=[ (ym-yd) for ym,yd in zip(Ys_model,Ys) ]
		Ys_model=sum([ list(y) for y in Ys_model],[])
	return Ys_model

"""
# What are these KZF functions? See paper "Measuring Sub-Surface Spatially-Varying Thermal Conductivity of Silicon Implanted with Krypton" by Pfeifer et al, Journal of Applied Physics. We can fit for a function of thermal conductivity vs depth (K(z)) if there is a gradient of properties, as is the case in ion-bombarded materials. 
from scipy.optimize import differential_evolution
kzfGuess=[] ; kzfBnd=[] ; lockedParam=[] ; lockedAs=[]
#@profile # install kernprof (python3-line-profiler), and run "sudo kernprof -lv SiKr.py 3_1Axis"
def solveKZFSimultaneous(listOfFiles,guesses='',bnds='',plotting="show"): #guesses=[var1guess,var2guess,var3guess...], bounds=((var1lower,var2lower,...)(var1upper,...)) #"plotting" options include: show, save, none
	incrementCounter("solveKZFSimultaneous")
	data=[];ts=[];fms=[]
	for fileToRead in listOfFiles: #import all files, reading in varying processing parameters (eg, modulation frequency). TODO: expand for spot size differences scan to scan. 
		ts,dataFromFile=readTDTR(fileToRead)
		fms.append(fm) #readTDTR set the modulation frequency as it read in the data points! so we read it here. 
		data.append(dataFromFile)
	if len(guesses)==0: # why? idk, passing funcs to funcs to funcs is hard. easier to use solveKZFsimultaneous() as a drop-in replacement for solve(), eg, within measureContour1Axis(). so you can either pass guesses/bounds, OR, set globals for them, idc
		guesses=kzfGuess
	if len(bnds)==0:
		bnds=kzfBnd
	#guesses=brute(brutehelper,np.transpose(np.asarray(bnds)),args=(data,ts,fms),Ns=10,workers=-1,finish=None)
	#print("Brute found guesses: ",guesses)
	lsqout=least_squares(solveKZFSimHelper, tuple(guesses), bounds=tuple(bnds),args=(data,ts,fms))
	#bnds=list(zip(*bnds)) # i guess differential_evolution wants [[minA,maxA],[minB,maxB],...] rather than what least_squares wants [[minA,minB,...],[maxA,maxB,...]]
	#print(bnds)
	#lsqout=differential_evolution(solveKZFSimHelper, bounds=tuple(bnds), args=(data,ts,fms))

	plotLabelStrings=[]
	for lsq in lsqout["x"]:
		plotLabelStrings.append(str(scientificNotation(lsq,2)))
	#print(lsqout["x"])
	#
	#
	residuals=solveSimultPlotting(lsqout,ts,data,listOfFiles,plotting,paramString=", ".join(plotLabelStrings))
	#print("counters['decayKZF']",counters)
	return lsqout["x"],residuals


#@profile
def solveKZFSimHelper(parameterValues,listOfDecays,ts,fms):
	results=np.zeros((len(fms),len(ts)))
	# hidden feature: say we'd normally fit for gaussian parameters A,B,C and interface 1 conductance G1. normally, we'll ignore tofit and just infer from parameterValues' length. 
	# now let's say you want to explore fitting of only parameters A,C,G1, and set B. for this, you'll need to add to lockedParam ("B") and lockedAs (locked value), and also ensure 
	# tofit can serve as an index of which entry in parameterValues corresponds to which parameter by name (["A","B","C","G1"])
	#print(lockedParam,lockedAs,tofit)
	for p,v in zip(lockedParam,lockedAs):
		i=tofit.index(p)
		parameterValues[i]=v
	#print("try with",parameterValues)
	global fm
	for i in range(0,len(fms)):
		fm=fms[i]
		result=decayKZF(ts,*parameterValues)
		results[i,:]=result[:]
	#print("SOLVEKZFHELPER:",parameterValues)
	#plotKZF(*parameterValues)
	#plot([ts]*2*len(fms),list(listOfDecays)+list(results))
	dz=listOfDecays-results
	#prettyPrint()
	#print("->",sum(dz.flatten()**2))
	#if counters["resKZFDecay"]>4:
	#	sys.exit()
	#return sum(dz.flatten()**2)
	return dz.flatten()

#@profile
def decayKZF(ts,*args,addNoise=False): #basically just wrap existing TDTRfunc code. only difference: we do our own prep (update tp ourselves) and then don't pass thermal properties to change
	global tp
	incrementCounter("decayKZF")
	conditionalPrint("decayKZF","pre-slice:",pp=True)
	tp_orig=copy.deepcopy(tp)	# save off: now that popPropmatKZF no longer just scoops up all middle layers, we need to make sure we don't leave a junky thermal property matrix when we're done. 
	popPropmatKZF(*args)		# generate sliced matrix
	conditionalPrint("decayKZF","post-slice:",pp=True)
	results=TDTRfunc(ts,addNoise=addNoise)		# generate TDTR decay curve
	#printtp()
	tp=copy.deepcopy(tp_orig)	# restore old matrix
	return results

#@profile
nslices=10 ; sliceWhich=2 # slice layer 2 of N (first layer is 1)
def popPropmatKZF(*args): #given an arbitrary property matrix with an arbitrary number of layers, we keep the first (transducer typically), apply KZF to the middle, and keep the final, slicing the middle into nslices. (thus, we can populate an initial base matrix, and still rehandle a matrix that's already been sliced up). 
	#if nslices==0: #silly hack to default in nslices. can't just do func(nslices=10,*args) because then trying to pass your args, nslices will eat the first one. 
	#	nslices=100 #NOPE. ASSUME GLOBAL NSLICES INSTEAD
	global tp
	d,C,Kz,Kr=[ c+str(sliceWhich) for c in ["d","C","Kz","Kr"] ] # names for params for sliced layer: "d2", "C2" etc
	d=getParam(d) ; C=getParam(C) ; Kz=getParam(Kz) ; Kr=getParam(Kr) # values of original params for sliced layer
	dz=d/nslices
	depths=getCol(2) # list of depths for stack
	#print(depths)
	dmin=sum(depths[:sliceWhich-1]) ; dmax=dmin+d
	centerdepths=np.linspace(dmin,dmax,nslices+1)[:-1]+dz/2 #linspace(0,7,5)-> [0,1.75,3.5,5.25,7], linear spacing ends included. we want n regions though, and the points in the center of them. transducer|-o-|-o-|-o-|base.
	i=(sliceWhich-1)*2 # index of slicable layer
	l0=[]
	for l in tp[:i]:	# save off all layers and interfaces before our slicable one
		l0.append(l)
	ln=[]
	for l in tp[i+2:]: # and all layers and interfaces after our slicable one (note: we sub TBC on the bottom-side of slicable for inf)
		ln.append(l)
	Ks=KZF(centerdepths,*args) # compute all Ks for sublayers
	Cs=CZF(centerdepths,*args) # and Cs for sublayers
	ls=[]
	for K,C in zip(Ks,Cs):	# layer and interface n +
		ls.append([C,K,dz,"Kz"])
		ls.append([np.inf])
	tp=l0+ls+ln
	KZFother(*args)
	conditionalPrint("popPropmatKZF","using parameters:",pp=True)

def KZF(z,*args): #HERE YOU DEFINE YOUR FUNCTION FOR Kz(z)! see testing22 for examples and testing. or set in your calling code and overwrite with sys.modules["TDTR_fitting"].KZF=yourKZF
	KZFWarning("KZF")
	K=tp[-1][1]
	return np.ones(len(z))*K

def CZF(z,*args):
	KZFWarning("CZF")
	C=tp[-1][0]
	return np.ones(len(z))*C

def KZFgaussian(z,*args): #I gave you a gaussian. your calling code should set this to overwrite KZF, eg, sys.modules["TDTR_fitting"].KZF=KZFgaussian
	A,B,C=args[:3] # historically we did this, but it yields an incredibly uneven weighting for final K(z=center). eg, A=linspace(0,1,21) (as results from scipy brute) yields K_center=[120., 17.1, 9.23, 6.32, 4.8, 3.87, 3.24, 2.79, 2.45, 2.18, 1.97, 1.79,...]
	K0=tp[-1][1]
	A=1/A-1/K0 # instead, let's say A is K(z=center). K=1/(1/Ko+A*gauss) -> A=1/K-1/Ko. but this comes with it's own challenge (too easy to be too course. 1e14 dose is definitely a min conductivity of 2-3 W/m/K,  

	# A, B, C are gausian center magnitude, center position, and width (no need to offset center position by transducer)
	R0=1./K0
	Rg=A*np.exp(-1.*(z-B)**2./(2.*C**2.))
	Rt=R0+Rg
	K=1./Rt
	return K

def KZFgaussian2(z,*args): #I gave you a gaussian. your calling code should set this to overwrite KZF, eg, sys.modules["TDTR_fitting"].KZF=KZFgaussian
	K0=tp[-1][1]
	#args 1 2 3 are gausian center magnitude, center position, and width (no need to offset center position by transducer)
	dK=-args[0]*np.exp(-1.*(z-args[1])**2./(2.*args[2]**2.))
	K=K0+dK
	return K
"""

def yemxpb(xs,m,b):
	return m*xs+b
def quadratic(xs,a,b,c):
	return a*xs**2+b*xs+c

"""
def KZFother(*args): #MANUALLY UPDATE ANY OTHER THERMAL PARAMETERS YOU WANT TO FIT FOR HERE. eg, setParam("G1",args[-1]) 
	KZFWarning("KZFother")

def KZFWarning(funcName,warned=[]): #function used to warn users who may be inappropriately calling stock KZF CZF KZFother
	if funcName not in warned:
		warn(funcName,"WARNING: Did you remember to populate TDTR_fitting.py>"+funcName+"? or set remotely? eg:\nFrom TDTR_fitting import *\ndef your"+funcName+":\n\t[your code]\nsys.modules[\"TDTR_fitting\"]."+funcName+"=your"+funcName+" #replaces TDTR_fitting's defined func with yours. beware: order of these three operations (import, define, replace) matters if yours is simply titled "+funcName+"()")
		warned.append(funcName)

def plotKZF(*args,plotting="show",maxdepth=2.5e-6): #USE THIS FOR PLOTTING YOUR GUESSES, MAKE SURE YOUR KZF IS SANE
	#print("plotting with args:",args)
	plotsteps=100
	ds=np.linspace(tp[0][2],maxdepth,plotsteps) # TODO: possible the user massed a maxdepth that is greater than our slicable layer's thickness! if so, we'll lie to you (showing you a larger sliced region than we'd actually use in poppropmatKZF
	ks=KZF(ds,*args)
	plotDs=[ds*1e9,ds*1e9];plotKs=[ds*0,ks]
	if 'referenceMat' in globals(): #check if variable defined: http://code.activestate.com/recipes/59892-testing-if-a-variable-is-defined/
		ds=np.linspace(0,maxdepth,plotsteps)
		dlayers=getCol(2)
		klayers=getCol(1)
		boundaries=[];rollsum=0
		for d in dlayers:
			rollsum=rollsum+d
			boundaries.append(rollsum) #location of nth boundary, is just the sum of previous layer thicknesses
		ks=np.ones(plotsteps)*klayers[0]
		for i in range(0,len(boundaries)-1):
			ks[ds>boundaries[i]]=klayers[i+1] #everywhere above this boundary gets set to the next k (subsuquent boundary overwrites)
		plotDs.append(ds*1e9);plotKs.append(list(ks))
	if len(tp)>10: #TODO: this is some bullshit method to infer whether the sliced properties matrix has been generated yet. do better. 
		ds=np.linspace(0,maxdepth,plotsteps)
		dlayers=getCol(2)
		klayers=getCol(1)
		boundaries=[];rollsum=0
		for d in dlayers:
			rollsum=rollsum+d
			boundaries.append(rollsum) #location of nth boundary, is just the sum of previous layer thicknesses
		ks=np.ones(plotsteps)*klayers[0]
		for i in range(0,len(boundaries)-1):
			ks[ds>boundaries[i]]=klayers[i+1] #everywhere above this boundary gets set to the next k (subsuquent boundary overwrites)
		plotDs.append(ds*1e9);plotKs.append(list(ks))
	figFile=""
	if "save" in plotting:
		figFile="pics/"+"-".join(list(map(str,args)))+plotting[4:]+".png"
	lplot(plotDs, plotKs, "depth (nm)", "Kz(z) (W m^-1 K^-1)", markers=['k:','k-','b-','g-'], labels=[".","Kz(z)","reference layers","sliced"], filename=figFile) #,legendLoc="lower right")
"""

# KNIFE EDGE
# https://en.wikipedia.org/wiki/Gaussian_function
# def Gaussian(xs,A,b,c): # Gaussian function f(x)=A*exp(-(x-b)^2/(2*c^2)). ps: beam radius is 2*c
#	return A*np.exp(-(xs-b)**2/(2*c**2))
# https://www.wolframalpha.com/input/?i=integral+A*exp%28-%28x-b%29%5E2%2F%282*c%5E2%29%29
# def integGaussian(xs,A,b,c,d): # integral of a gaussian function = sqrt(pi/2)*c*erf((b-x)/(sqrt(2)*c)) ps: beam radius is 2*c
#	return -A*np.sqrt(np.pi/2)*c*erf((b-xs)/(np.sqrt(2)*c))+d 

def Gaussian(xs,A,b,radius): # Gaussian function without goofy 2*c crap: f(x)=2/π/r²*np.exp(-2*rs²/r²)
	return A*np.exp(-2*(xs-b)**2/radius**2)
def integGaussian(xs,A,b,radius,d):
	return -A*np.sqrt(2*np.pi)/4*radius*scipy.special.erf((b-xs)*np.sqrt(2)/(radius))+d 

def dx(f,x):
	dx=x[:-1]-x[1:]
	df=f[:-1]-f[1:]
	return df/dx,(x[:-1]+x[1:])/2

def knifeAll(directory): # we'll do knife-edge on all files within the directory, detecting pump vs probe and x vs y, and generate a radii file
	diameters=[] ; centers=[]
	d_sorted={"Probe_X":[],"Probe_Y":[],"Pump_X":[],"Pump_Y":[]} ; c_sorted={"Probe_X":[],"Probe_Y":[],"Pump_X":[],"Pump_Y":[]}
	files=glob.glob(directory+"/*.mat")+glob.glob(directory+"/*.txt")
	files=[ f.replace("\\","/") for f in sorted(files) ] # catch \\ issue on windows: https://stackoverflow.com/questions/60010487/python-glob-path-issue
	files=[ f for f in files if "radii.txt" not in f ] # previous run's "radii.txt" file is not data! ha!
	
	conditionalPrint("knifeAll",str(directory)+" "+str(files))
	for f in files:
		b,r,pupr_xy=knifeEdge(f)
		conditionalPrint("knifeAll","f,pupr_xy,b,c "+str(f)+" "+str(pupr_xy)+" "+str(b)+" "+str(r))
		diameters.append(2*r)
		centers.append(b)
		#print(f.split("_")[-4:-2],b)
		d_sorted[pupr_xy].append(2*r)
		c_sorted[pupr_xy].append(b)

	conditionalPrint("knifeAll","diameters center")
	for k in d_sorted.keys():
		ds=d_sorted[k] ; cs=c_sorted[k]
		conditionalPrint("knifeAll",str(k)+" : "+str(np.mean(ds))+" +/- "+str(np.std(ds))+" ; "+str(np.mean(cs)))

	sizes={k:np.mean(d_sorted[k]) for k in d_sorted.keys()}
	offsets={ "X":np.mean(c_sorted["Pump_X"])-np.mean(c_sorted["Probe_X"]) , "Y":np.mean(c_sorted["Pump_Y"])-np.mean(c_sorted["Probe_Y"]) }

	rpu=( sizes["Pump_X"]+sizes["Pump_Y"] )/2 ; rpr=( sizes["Probe_X"]+sizes["Probe_Y"] )/2
	if not np.isnan(rpu) and not np.isnan(rpr):
		d=f.split("/")[:-1] # trim off trailing filename, and our added "knifeEdge.py_" folder
		d.append("radii.txt")
		d="/".join(d)
		conditionalPrint("knifeAll","both radii found, writing to file: "+str(d)+". copy this into the calibration scan folder and run fiberCals.py to process those, using these radii")
		f=open(d,'w')
		f.write("rpu="+str(np.round(rpu,2))+"e-6/2\n")
		f.write("rpr="+str(np.round(rpr,2))+"e-6/2\n")
		f.close()
	return sizes,offsets

def knifeEdge(f,useLast=True): # you can measure a spot size by passing the edge of a sample (abrupt edge required, e.g. a transparent sample with a masked transducer) through the beam, and measuring the reflected and/or transmitted beam intensity. if the beam is gaussian, your intensity vs position response will be the integral, i.e., an error function
	# STEP 1: read in the data
	if ".mat" in f:
		matlabData=scipy.io.loadmat(f)					# read matlab file
		P=matlabData["V1"][0] ; X=matlabData["vect"][0]			# getting powers, and locations
		#print("knifeEdge dict keys",matlabData.keys())
	else:
		lines=open(f,'r').readlines()
		X=[float(l.split(",")[0]) for l in lines if "#" not in l ] ; P=[float(l.split(",")[1]) for l in lines if "#" not in l ]
		X=np.asarray(X) ; P=np.asarray(P)
	if X[0]>X[-1]:
		X=np.asarray(list(reversed(X))) ; P=np.asarray(list(reversed(P)))
	if P[0]>P[-1]:
		P=np.asarray(list(reversed(P)))
	# STEP 2: preparing the plot
	fname=f.split("/") ; fname.insert(-1,"knifeEdge.py_")		# false matches for Y! 
	fname="/".join(fname) ; fname=fname.replace(".mat",".png").replace(".txt",".png")
	pupr={True:"Probe",False:"Pump"}["Probe" in f]			# detect if pump or probe
	xy={True:"Y",False:"X"}["Y" in f.split("/")[-1]]		# x or y dimension. only use filename, not path! or we get

	col={"Pump_X":'r',"Pump_Y":'c',"Probe_X":'g',"Probe_Y":'b'}[pupr+"_"+xy]
	plotX=[X] ; plotY=[P] ; mkrs=[col+'o']

	# STEP 3: figure out guesses for fitting gaussian (integral)	
	mean=(max(P)+min(P))/2 ; imean=np.argmin(np.abs(P-mean))	# CENTER: mean (highest,lowest point), -> X datapoint for closest P datapoint
	b=X[imean]
	r=5								# WIDTH: fuck it, assume radius of 5um for now (good enough)
	A=(max(P)-min(P))/r/np.sqrt(np.pi/2)				# MAGNITUDE: highest-lowest = 2 * A*sqrt(pi/2)*c
	d=mean								# VERTICAL OFFSET: just the mean
	plotX.append(X) ; plotY.append( integGaussian(X,A,b,r,d) ) ; mkrs.append(col+":")

	#lplot(plotX, plotY, "position (um)", "", markers=mkrs, labels=[""]*len(plotX), useLast=useLast, includeZeroX=False)
	#sys.exit()
	# STEP 4: actually solve it
	solvedParams, parm_cov = curve_fit(integGaussian, X, P, p0=(A,b,r,d)) #curve fit, dR/R (x) to the integral of a gaussian
	A,b,r,d=solvedParams ; r=abs(r)

	# STEP 5: plot the results
	plotX.append(X) ; plotY.append( integGaussian(X,A,b,r,d) ) ; mkrs.append(col+"-")
	dPdX,dX=dx(P,X)	; dPdX2=Gaussian(dX,A,b,r)			# derivative of our data / fit (gaussian)
	dPdX-=min(dPdX) ; dPdX/=max(dPdX) ; dPdX*=max(P)
	dPdX2-=min(dPdX2) ; dPdX2/=max(dPdX2) ; dPdX2*=max(P)
	plotX.append(dX) ; plotY.append(dPdX) ; mkrs.append(col+"o")
	plotX.append(dX) ; plotY.append(dPdX2) ; mkrs.append(col+":")

	# PLOTTING
	#P2=integGaussian(X,A,b,c,d)					# found integral
	
	#Xs=[X,X,dX,dX] ; Ys=[P/max(P),P2/max(P2),dPdX/max(dPdX),dPdX2/max(dPdX2)]
	#print("maxes",[max(y) for y in Ys])
	xlabel="position (um)" ; ylabel="Aux (V) , dA/dx (-)" #; mkrs=[col+'o',col+':',col+'+',col+':']
	title=pupr+" "+xy+" diameter="+str(round(2*r,4))+"um"
	#plot(Xs,Ys,xlabel,ylabel,markers=mkrs,datalabels=[""]*4,title=title,filename=fname,useLast=useLast,includeZeroX=False)
	lplot(plotX, plotY, xlabel, ylabel, markers=mkrs, labels=[""]*len(plotX), title=title, filename=fname, useLast=useLast, includeZeroX=False)
	return b,r,pupr+"_"+xy # diameter=4*c, center position=b

# based off of isKeWrong.py (2022_12_13_Fiber), sweep spot sizes, fit each cal at each spot size
def sweepFiberSpotSizes(fileDirec,calmatDirec,factor=25,n=15,materials=["Al2O3","SiO2","Quartz","Si"]): 
	conditionalPrint("calsForSpots","preparing to run for: "+fileDirec+" , "+calmatDirec)
	aliases={"Quartz":"cSiO2"}
	# save off original parameters
	rpu,rpr=getParam("rpu"),getParam("rpr")
	setVar("autorpu",False) ; setVar("autorpr",False)
	# we're going to run through a sweep of spot sizes, and fit for every cal at each spot size
	results=[] ; factor=np.linspace(1-(factor/100),1+(factor/100),n)
	for s in factor:
		conditionalPrint("calsForSpots","scaling: "+str(s)+"x")
		# scale the default spot sizes
		setParam("rpu",rpu*s) ; setParam("rpr",rpr*s)
		results.append([])
		# cycle through materials
		for mat in materials:
			conditionalPrint("calsForSpots","mat: "+mat)
			matfile=glob.glob(calmatDirec+"/*"+mat+"_cal_matrix.txt")[0]
			importMatrix(matfile)
			files=glob.glob(fileDirec+"/*_"+mat+"_*.txt")
			if mat in aliases.keys():
				files+=glob.glob(fileDirec+"/*_"+aliases[mat]+"_*.txt")
			#files=files[:3]
			conditionalPrint("calsForSpots","files: "+str(files))
			#files = [ f for f in files if "FDTR" not in f ]
			# solve for each file (result,error returned. ge the first result for each)
			rs = [ solve(f,plotting="save")[0][0] for f in files ]
			conditionalPrint("calsForSpots","fitted: "+str(rs))
			results[-1].append(np.mean(rs))

	lplot([factor]*len(materials),np.transpose(results),xlabel="scale radii by (-)",ylabel=tofit[0],title="",xlim=["nonzero","nonzero"], ylim=["nonzero","nonzero"],labels=materials)

# PROCEDURE FOR ANALYZING FIBER SSTR CALS: (this function is based off of superMegaFit.py (2022_12_13_Fiber)
# run file-averaging on each material's set of files (we're going to run super-simultaneous fitting, and don't want to deal with 15x4 files)
# create a magicModifiers file: each row sets the thermal properties matrix (based on calmats files), one row per (averaged) file
# run super-simultaneous (ss2), fitting for rpu+gamma
# THIS IS A GENERALIZED FUNCTION: it should work for other scenarios too (whatever you set mode/fitting/etc to), e.g., HQ TDTR probe down, fitting for spot sizes too. for fiber-specific functionality, check out gui.py > fibercals(), which does the fiber-specific setup, and then calls this
def calsForSpots(fileDirec,calmatDirec,materials=["Al2O3","SiO2","Quartz","Si"],averaging=True):
	conditionalPrint("calsForSpots","preparing to run for: "+fileDirec+" , "+calmatDirec)
	FILES=[] ; aliases={"Quartz":"cSiO2","SiO2":"aSiO2","Al2O3":"Sapphire"} ; matDict={}
	f=open(fileDirec+"/magicModifiers_superMegaFit.txt",'w')
	conditionalPrint("calsForSpots","generating magicMods: "+fileDirec+"/magicModifiers_superMegaFit.txt")
	for mat in ["Al2O3","SiO2","Quartz","Si"]:					# for each cal
		files=glob.glob(fileDirec+"/*_"+mat+"_*.txt")				# n files following this naming pattern
		if mat in aliases.keys():						# also search for files with alias names
			files+=glob.glob(fileDirec+"/*_"+aliases[mat]+"_*.txt")
		conditionalPrint("calsForSpots","mat: "+mat+", found files: "+",".join(files))
		if len(files)==0:							# fileAverager may crash if the empty filelist given
			continue
		if len(files)>1:
			files=[ f for f in files if "AVG" not in f ]			# if an averaged file shows up in the list (e.g. re-running
		matDict[mat]={"files":files}						# calspots from the GUI), skip it (we'll regenerate)
		if averaging:
			ftype={"SSTR":"fSSTR","TDTR":"TDTR"}[mode] ; i0=(mode=="SSTR")
			fo,ig=fileAverager(files,fileType=ftype,ignoreOutliers=i0)
			matDict[mat]["files"]=[ f for f in matDict[mat]["files"] if f not in ig ]
			files=[fo]
			conditionalPrint("calsForSpots","(mat: "+mat+", averaged: "+fo)
		FILES+=files
		matfile=glob.glob(calmatDirec+"/*"+mat+"_cal_matrix.txt")[0]
		matDict[mat]["matfile"]=matfile
		conditionalPrint("calsForSpots","mat: "+mat+", matfile: "+matfile)
		importMatrix(matfile)							# read in the matrix file
		tpstring="tp="+str(getVar("tp"))+"\n" 					# "tp=[[C1,Kz1,d1,..],[...]...]"
		for i in range(len(files)):							# for each datafile, we'll add a line to magicMods
			f.write(tpstring)
	f.close()

	FILES.append(fileDirec+"/magicModifiers_superMegaFit.txt")
	conditionalPrint("calsForSpots","ready to run ss2 with files: "+",".join(FILES))
	r,e=ss2(FILES,[mode]*(len(files)-1))
	conditionalPrint("calsForSpots",str(r)+","+str(e))
	return r,e,matDict

	

def picoAcoustics1(filename,soundspeed=6340,minFitting=5e-12,maxFitting=50e-12,humpWidth=6e-12,plotting="show"):
	stuff=picoAdvanced(filename,soundspeed,minFitting,maxFitting,humpWidth,plotting)
	return stuff[:2]

def picoAdvanced(filename,soundspeed=6340,minFitting=5e-12,maxFitting=50e-12,humpWidth=6e-12,plotting="show"):	
	# import data
	ts,xs,ys,aux=readTDTRdata(filename) ; ms=np.sqrt(xs**2+ys**2)
	ms/=max(ms) ; f=interp1d(ts,ms)
	Xs=[ts] ; Ys=[ms] ; mkrs=['ko'] ; dlbs=["raw"]
	def trim(ts,ms,tmin,tmax):
		mt=ms[ts<=tmax] ; tt=ts[ts<=tmax] ; mt=mt[tt>=tmin] ; tt=tt[tt>=tmin]
		return tt,mt
	d,d2,A1,A2=0,0,0,0
	try:
		ts_trim,ms_trim=trim(ts,ms,minFitting,maxFitting)
		if len(ts_trim)==0 or max(ts_trim)<=15e-12: # if no data between min and max, OR, if this was a normal TDTR scan (no real acoustics data)
			warn("picoAcoustics","this may not be an acoustics scan? (len(ts)==0 or np.amax(ts)<15e-12)")
			return 0,0,0,0
		# find rise start (point of max curvature, d²M/dt²), and peak
		i_rise=np.argmax(np.gradient(np.gradient(ms))) ; i_peak=np.argmax(ms) # i_rise and i_peak are wrt base
		# initial pulse at 3/4ths of the way between rise start and first peak (found by comparing hump rises against acoustic runs with a second peak)
		t0a=ts[i_peak] ; t0b=ts[i_rise] ; weight=.75
		t0=t0a*(1-weight)+t0b*weight ; m0=f(t0)
		Xs.append([t0]) ; Ys.append([m0]) ; mkrs.append('ro') ; dlbs.append("t0")
		# running average --> fit exponential --> subtract
		ms_avg=runningAverage(ms_trim,15)[7:-7] ; ts_avg=ts_trim[7:-7] # TDTR_fitting's runningAverage does wrapping, hence 7:-7
		guesses=(max(ms_avg)-min(ms_avg) , -.05e12 , min(ms_avg) , -ts_avg[0])
		def exponential(ts,A,B,C,D):
			return A*np.exp((ts+D)*B)+C
		expo, parm_cov = curve_fit(exponential, ts_avg, ms_avg,p0=guesses) #solve it
		Xs.append(ts) ; Ys.append(exponential(ts,*expo)) ; mkrs.append('k:') ; dlbs.append("expo")
		# fit a quadratic to the hump
		t_hump=ts_avg[np.argmax(ms_avg-exponential(ts_avg,*expo))] ; m_hump=f(t_hump)
		Xs.append([t_hump]) ; Ys.append([m_hump]) ; mkrs.append('bo') ; dlbs.append("hump")
		hw=humpWidth/4 ; hW=humpWidth/2
		def quadratic(ts,a,h,k): # f(x)=a(x-h)²+k where (h,k) are the local minima/maxima
			return a*(ts-h)**2+k
		h=t_hump ; k=m_hump ; a=(f(t_hump-hw)-k)/((t_hump-hw-h)**2) # h,k (see above). for a, pick another nearby point (-3ps), and solve
		ts_hump,ms_hump=trim(ts,ms,t_hump-hW,t_hump+hW)
		Xs.append(ts_hump) ; Ys.append(ms_hump) ; mkrs.append('go') ; dlbs.append("raw hump")
		guesses=(a,h,k) ;  quad, parm_cov = curve_fit(quadratic, ts_hump, ms_hump,p0=guesses)
		ts_quad,ms_quad=trim(ts,ms,t_hump-hW*2,t_hump+hW*2)
		Xs.append(ts_quad) ; Ys.append(quadratic(ts_quad,*quad)) ; mkrs.append('k-') ; dlbs.append(" ")
		Xs.append(ts_quad) ; Ys.append(quadratic(ts_quad,*quad)) ; mkrs.append('g:') ; dlbs.append("quad")
		# precise hump location found, calculate time between initial pulse, and echo
		tf=quad[1] ; mf=f(tf) ; tsh,msh=trim(ts,ms,tf-hW,tf+hW) ; ih=np.argmax(msh) ; tf=tsh[ih] ; mf=msh[ih] # use quad middle, OR, local max
		Xs.append([tf]) ; Ys.append([mf]) ; mkrs.append('ro') ; dlbs.append(" ")
		d=(tf-t0)*soundspeed/2 # distance=velocity*time, echo makes 2 trips (down and back)
	# for the sake of plotting
	#Xs=[ts,ts,[t_hump],ts_hump,ts_hump,[t0,tf]] ; Ys=[ms,exponential(ts,*expo),[m_hump],ms_hump,quadratic(ts_hump,*quad),[m0,mf]]
	#mkrs=['ko','k:','bo','g:','go','ro']
		title="d="+str(np.round(d*1e9,2))+"nm"
	# if data exist out to where a valley should be, then check for it!
		t2=tf+(tf-t0)
		ms_hump2=ms[ts>=t2-hW] ; ts_hump2=ts[ts>=t2-hW] ; ms_hump2=ms_hump2[ts_hump2<=t2+hW] ; ts_hump2=ts_hump2[ts_hump2<=t2+hW]
		if len(ts_hump2)>5:
			Xs.append(ts_hump2) ; Ys.append(ms_hump2) ; mkrs.append('go') ; dlbs.append(" ")
			print("second hump available!")
			h=t2 ; k=f(t2) ; a=(f(t2-hw)-k)/((t2-hw-h)**2)
			guesses=(a,h,k) ;  quad2, parm_cov = curve_fit(quadratic, ts_hump2, ms_hump2,p0=guesses)
			ts_quad2,ms_quad2=trim(ts,ms,t2-hW*2,t2+hW*2)
			Xs.append(ts_quad2) ; Ys.append(quadratic(ts_quad2,*quad2)) ; mkrs.append("k-") ; dlbs.append(" ")
			Xs.append(ts_quad2) ; Ys.append(quadratic(ts_quad2,*quad2)) ; mkrs.append("g:") ; dlbs.append(" ")
			tf2=quad2[1] ; mf2=f(tf2) ; tsh,msh=trim(ts,ms,tf2-hW,tf2+hW) ; ih=np.argmin(msh) ; tf2=tsh[ih] ; mf2=msh[ih]
			Xs.append([tf2]) ; Ys.append([mf2]) ; mkrs.append("ro") ; dlbs.append(" ")
			d2=(quad2[1]-tf)*soundspeed/2 ; title=title+","+str(np.round(d2*1e9,2))+"nm"
		print(ts_hump2)
		ts_interp=np.linspace(min(ts),max(ts),1000) ; Xs.append(ts_interp) ; Ys.append(f(ts_interp)) ; mkrs.append("r:") ; dlbs.append("interp")
		# plot it all: original, exponential fit, hump datapoint, hump quadratic
		# ANOTHER WAY TO MEASURE HUMP HEIGHT: take the region of the hump plus a bit on either side, and find 3 points with max curvature. two are your outer "begin rise" "end fall", one is your center. third, minus average of the first two, gives height
		ts_L_1a,ms_L_1a=trim(ts,ms,t_hump-3*hW,t_hump-hw) #; print("len(ts_L_1a)",len(ts_L_1a))
		#Xs.append(ts_L_1a) ; Ys.append(ms_L_1a) ; mkrs.append('b+') ; dlbs.append("L_1a")
		ts_L_1c,ms_L_1c=trim(ts,ms,t_hump+hw,t_hump+3*hW) #; print("len(ts_L_1c)",len(ts_L_1c))
		#Xs.append(ts_L_1c) ; Ys.append(ms_L_1c) ; mkrs.append('bx') ; dlbs.append("L_1c")
		i1a=np.argmax(  np.absolute( np.gradient(  np.gradient(ms_L_1a,ts_L_1a),ts_L_1a  ) )  )
		#Xs.append([ts_L_1a[i1a]]) ; Ys.append([ms_L_1a[i1a]]) ; mkrs.append("bo") ; dlbs.append("maxcurve")
		i1c=np.argmax(  np.absolute( np.gradient(  np.gradient(ms_L_1c,ts_L_1c),ts_L_1c  ) )  )
		#Xs.append([ts_L_1c[i1c]]) ; Ys.append([ms_L_1c[i1c]]) ; mkrs.append("bo") ; dlbs.append(" ")
		A1=mf - np.mean([ms_L_1a[i1a],ms_L_1c[i1c]]) ; print("A1",A1)

		t_hump2=ts[ np.argmin( np.absolute(ts-tf2) ) ]
		ts_L_2a,ms_L_2a=trim(ts,ms,t_hump2-3*hW,t_hump2-hw) #; print("len(ts_L_2a)",len(ts_L_2a))
		#Xs.append(ts_L_2a) ; Ys.append(ms_L_2a) ; mkrs.append('b+') ; dlbs.append("L_2a")
		ts_L_2c,ms_L_2c=trim(ts,ms,t_hump+hw,t_hump+3*hW) #; print("len(ts_L_2c)",len(ts_L_2c))
		#Xs.append(ts_L_2c) ; Ys.append(ms_L_2c) ; mkrs.append('bx') ; dlbs.append("L_2c")
		i2a=np.argmax(  np.absolute( np.gradient(  np.gradient(ms_L_2a,ts_L_2a),ts_L_2a  ) )  )
		#Xs.append([ts_L_2a[i2a]]) ; Ys.append([ms_L_2a[i2a]]) ; mkrs.append("bo") ; dlbs.append(" ")
		i2c=np.argmax(  np.absolute( np.gradient(  np.gradient(ms_L_2c,ts_L_2c),ts_L_2c  ) )  )
		#Xs.append([ts_L_2c[i2c]]) ; Ys.append([ms_L_2c[i2c]]) ; mkrs.append("bo") ; dlbs.append(" ")
		A2=mf - np.mean([ms_L_2a[i2a],ms_L_2c[i2c]]) ; print("A2",A2)

	except Exception as e:
		print("TDTR_fitting > picoAdvanced > ERROR ON FILE:",filename)
		print(e)
		traceback.print_exc()
		#return d,0,0
		pass
	dt=ts[1]-ts[0]
	#direc=filename.split("/")[:-1]+[callingScript,"pics",filename.split("/")[-1]+".png"]
	#figFile={"show":"","save":"/".join(direc)}[plotting]
	title=filename.split("/")[-1]+".png\n"+title+", (1 datapoint="+str(np.round(dt*1e12,2))+"ps="+str(np.round(dt*soundspeed/2*1e9,2))+"nm)"
	#dlbs=["","time pts","expo","","quad","hump pts","","","","runav"]
	print(mkrs,dlbs)
	lplot(Xs, Ys, "time (s)", "Mag (-)", markers=mkrs, title=title, filename=figFile(filename,plotting,"accoustics"), labels=dlbs, xlim=[-15e-12,100e-12])
	return d,d2,A1,A2

def picoAcoustics(filename,soundspeed=6340,plotting="show"):
	def p():
		lplot(Xs, Ys, "time (s)", "Mag (-)", markers=mkrs, title=title, labels=dlbs, xlim=[-15e-12,100e-12],ylim=[-0.1*max(ms),1.1*max(ms)])
	def trim(ts,ms,tmin,tmax):
		mt=ms[ts<=tmax] ; tt=ts[ts<=tmax] ; mt=mt[tt>=tmin] ; tt=tt[tt>=tmin]
		return tt,mt
	def exponential(ts,A,B,C,D):
		return A*np.exp((ts+D)*B)+C
	def quadratic(ts,a,h,k): # f(x)=a(x-h)²+k where (h,k) are the local minima/maxima
		return a*(ts-h)**2+k
	timepoints=[]
	# READ IN DATA
	ts,xs,ys,aux=readTDTRdata(filename) ; ms=np.sqrt(xs**2+ys**2)
	ms/=max(ms) ; f=interp1d(ts,ms)
	Xs=[ts] ; Ys=[ms] ; mkrs=['ko'] ; dlbs=["raw"] ; title="raw" ; p()
	# FIND THE RISE (MAX CURVATURE d²M/dt²) AND THE MAX
	i_rise=np.argmax(np.gradient(np.gradient(ms))) ; i_peak=np.argmax(ms)
	# t=0 IS CONSIDERED AS 3/4th BETWEEN RISE START AND MAX:
	t0a=ts[i_rise] ; t0b=ts[i_peak] ; weight=.25 ; m0a=ms[i_rise] ; m0b=ms[i_peak]
	t0=t0a+(t0b-t0a)*weight ; m0=f(t0) ; timepoints.append(t0)
	Xs.append([t0,t0a,t0b]) ; Ys.append([m0,m0a,m0b]) ; mkrs.append('ro') ; dlbs.append("t0") ;  p()
	# FIT EXPO TO ALL DATA *except* 15 < t < 40
	ts_a,ms_a=trim(ts,ms,3e-12,15e-12) ; ts_b,ms_b=trim(ts,ms,40e-12,100e-12)
	ts_exc=np.concatenate((ts_a,ts_b)) ; ms_exc=np.concatenate((ms_a,ms_b))
	Xs.append(ts_exc) ; Ys.append(ms_exc) ; mkrs.append('bo') ; dlbs.append("exclude") ;  p()
	# ACTUAL EXPO FIT
	guesses=(max(ms_exc)-min(ms_exc) , -.05e12 , min(ms_exc) , -ts_exc[0])
	ms_guesses=exponential(ts,*guesses) ; Xs.append(ts) ; Ys.append(ms_guesses) ; mkrs.append('b:') ; dlbs.append("guesses") ;  p()
	expo, parm_cov = curve_fit(exponential, ts_exc, ms_exc,p0=guesses) #solve it
	ms_expo=exponential(ts,*expo) ; Xs.append(ts) ; Ys.append(ms_expo) ; mkrs.append('g,--') ; dlbs.append("expo") ;  p()
	for i in range(2):
		# SUBTRACT, SELECT PEAK, APPLY QUAD FIT
		if i==0:		# FIRST HUMP, USE SUBTRACTED DATASET'S ABSMAX
			ms_sub=ms-ms_expo ; ts_sub,ms_sub=trim(ts,ms_sub,5e-12,50e-12)
			Xs.append(ts_sub) ; Ys.append(ms_sub) ; mkrs.append('b.') ; dlbs.append("sub") ;  p()
			i_hump=np.argmax(np.absolute(ms_sub)) # INDEX RELATIVE TO TO TRIMMED SUBTRACTED
			h=ts_sub[i_hump]
		else:			# SUBSEQUENT HUMPS, WE GUESS OFFSET FROM THE PREVIOUS HUMP
			h=timepoints[-1]+(timepoints[1]-timepoints[0])
		# y=a(x-h)²+k, given two points (xₕ,yₕ) for the hump, and (x₃,y₃) for 3ps off, h=xₕ, k=yₕ, a=(y₃-k)/(x₃-h)²
		k=f(h) ; x3=h-3e-12 ; y3=f(x3) ; a=(y3-k)/(x3-h)**2 ; guesses=(a,h,k)
		ms_guesses=quadratic(ts,*guesses) ; Xs.append(ts) ; Ys.append(ms_guesses) ; mkrs.append('b:') ; dlbs.append("guesses") ;  p()
		ts_quad,ms_quad=trim(ts,ms,h-6e-12,h+6e-12)
		quad, parm_cov = curve_fit(quadratic, ts_quad, ms_quad,p0=guesses) #solve it
		t_hump=quad[1] ; m_hump=f(t_hump) ; timepoints.append(t_hump)
		# CALCULATE THICKNESS FROM FIRST HUMP BEFORE PLOTTING
		thickness=(timepoints[-1]-timepoints[-2])*soundspeed/2*1e9 ; thickness=str(np.round(thickness,2))
		if i==0:
			title="thickness="+thickness+"nm"
		else:
			title=title+","+thickness+"nm"
		ms_quad=quadratic(ts,*quad) ; Xs.append(ts) ; Ys.append(ms_quad) ; mkrs.append('g,--') ; dlbs.append("quad") ;  p()


def printtp(): # print the thermal properties matrix
	printwidth=17
	Clab="     C (J/m³/K)";lenClab=10;KzGlab="Kz/G (W/m⁽²⁾/K)";lenKzGlab=15;dlab="d (m)";lendlab=5;Krlab="Kr (W/m/K)";lenKrlab=10 #len(string) and ljust fail (https://bugs.python.org/issue3446, https://stackoverflow.com/questions/29109944/python-returns-length-of-2-for-single-unicode-character-string) on mixed unicode strings depending on python environment, so we manually define the length so we can pad them correctly
	print(Clab+"".ljust(printwidth-lenClab)+KzGlab+"".ljust(printwidth-lenKzGlab)+dlab+"".ljust(printwidth-lendlab)+Krlab+"".ljust(printwidth-lenKrlab))
		
	for i,r in enumerate(tp):
		printStr={0:"L "+str(int(i/2+1)),1:"I "+str(int((i-1)/2+1))}[i%2]+": "
		if len(r)==1:
			printStr=printStr.ljust(printwidth)
		for c in r:
			c=str(c) #.0000000123456789123456789 -> "1.234567891234568e-08"
			if "e" in c:
				c=c.split('e') #"1.234567891234568e-08" -> ["1.234567891234568","-08"]
				c=c[0][:printwidth-1-1-len(c[1])]+'e'+c[1] #trim trailing decims to length, printwidth minus 1, and space for e and exp
			c=c[:printwidth-1]
			printStr=printStr+str(c).ljust(printwidth)
		print(printStr)

def main():
	for f in filesToRead:
		results,[residuals,sigmas]=solve(f,plotting="show") #solve() handles importing, guesses, solving, plotting, and displaying. (tbh there's really no point for main() anymore)
### END SETUP AND SOLVING ###

doPhaseCorrect=True
### VARIOUS OTHER HELPERS ###

def readTDTR(filename,plotPhase=False):
	autos(filename)
	ts,xs,ys,aux=readTDTRdata(filename)
	conditionalPrint("readTDTR","raw read:"+str(ts)+", "+str(xs)+", "+str(ys))
	#conditionalPrint("readTDTR","raw ts,xs: "+str(ts)+str(xs))
	#recalculate other global values
	#wo=math.sqrt(.5*rpump**2.+.5*rprobe**2.) #w₀=√(½(w₁²+w₂²)) #Jiang eq 2.18+
	omegaM=fm*2.*pi
	#then do phase correction, trimming, normalization
	if doPhaseCorrect:
		xs,ys=phaseCorrect(xs,ys,ts,plotting=plotPhase) 	#phase correction: ΔY across t=0 should be zero, so check, and rotate entire system about origin: radius=√ X²+Y², simply changing angle
		#if plotPhase:
		#	return xs,ys
	conditionalPrint("readTDTR","phased:"+str(ts)+", "+str(xs)+", "+str(ys))
	xs=xs[ts>=minimum_fitting_time] ; ys=ys[ts>=minimum_fitting_time]
	aux=aux[ts>=minimum_fitting_time] ; ts=ts[ts>=minimum_fitting_time] 	#trim to times above cutoff for fitting:
	if maximum_fitting_time is not None:
		xs=xs[ts<=maximum_fitting_time] ; ys=ys[ts<=maximum_fitting_time]
		aux=aux[ts<=maximum_fitting_time] ; ts=ts[ts<=maximum_fitting_time] 	#trim to times above cutoff for fitting:
	data={"R":-xs/ys,"M":(xs**2.+ys**2.)**.5,"X":xs,"Y":ys}[fitting] 	#convert to type requested (ratio, magnitude, etc), and normalize if appropriate
	#data*=1e6 # converts V to uV????
	if fitting in "MXY":
		data=normalize(ts,data,auxes=aux[:,0])
	#print("ts,data",ts,data)
	conditionalPrint("readTDTR","processed ts,data: "+str(ts)+str(data),pp=True)
	#data-=np.amin(data)/20
	if "expA" in tofit:
		global expA,expB,expC
		expA=max(data)-data[-1] ; expC=data[-1]
	return ts,data

# HOW DO WE HANDLE PULSED VS CW SSTR? sneakily. if the file selected has "TDTR" in the name, we read it in using the TDTR code (readpSSTR > readTDTRdata), set a global for time delay (sstrDelay), and set mode="pSSTR" (readSSTR). we then use pulsed code for the SSTR function (SSTRfunc)
def readSSTR(filename):	
	autos(filename)
	lines=open(filename,'r').readlines()
	if "pSSTR" in filename[-10:]:
		conditionalPrint("readSSTR","calling pulsedSSTR code")
		global mode ; mode="pSSTR"
		P,M=readpSSTR(filename)
	elif "Pump Mag" in lines[2]:
		conditionalPrint("readSSTR","calling HQ code")
		P,M=readHQSSTR(filename)
	else:
		conditionalPrint("readSSTR","calling fiber code")
		P,M=readFiberSSTR(filename)
	conditionalPrint("readSSTR","found P,M: "+str(P)+","+str(M))
	#P=P[1:] ; M=M[1:] # experimental: first SSTR datapoint at pump power = 0 will just be all noise, so it may be reasonable to omit
	solvedParams, parm_cov = curve_fit(yemxpb, P, M) # y=m*x+b
	M-=solvedParams[1]
	#print((M[-1]-M[0])/(P[-1]-P[0]),solvedParams[0])
	#print(curve_fit(yemxpb, P, M))
	#print(solvedParams,P,M-solvedParams[1])
	return P,M

# HOW DO WE HANDLE PULSED VS CW SSTR? sneakily. if the file selected has "TDTR" in the name, we read it in using the TDTR code (readpSSTR > readTDTRdata), set a global for time delay (sstrDelay), and set mode="pSSTR" (readSSTR). we then use pulsed code for the SSTR function (SSTRfunc)
def readpSSTR(filename):
	ts,xs,ys,aux=readTDTRdata(filename)
	aux=np.asarray(aux)
	P=aux[:,1]/1000 ; M=(xs**2+ys**2)**(1/2)*1000/aux[:,0] # P (aux) is in Volts, M (signal) is in uV. if you don't scale, you're looking at gamma ~ 1e7 and the fitting code hates huge numbers
	global sstrDelay ; sstrDelay=ts[0]
	return P,M

def readFiberSSTR(filename):
	lines=open(filename,'r').readlines()
	# columns are: Pu X, Pu X std, Pu Y, Pu Y std, Pr X, Pr X std, Pr Y, Pr Y std, aux 1, aux 2
	PuX,PuY,PrX,PrY,Aux=[],[],[],[],[]
	for l in lines[2:]:
		#print(l)
		if len(l)==0 or l[0]=="#":
			continue
		#print(l)
		l=[ float(v) for v in l.split() ]
		#print(l)
		PuX.append(l[0]) ; PuY.append(l[2]) ; PrX.append(l[4]) ; PrY.append(l[6]) ; Aux.append(l[8])
	PuX=np.asarray(PuX) ; PuY=np.asarray(PuY) ; PrX=np.asarray(PrX) ; PrY=np.asarray(PrY) ; Aux=np.asarray(Aux)
	conditionalPrint("readSSTR","PuX:"+str(PuX)+", PuY:"+str(PuY)+", PrX:"+str(PrX)+", PrY:"+str(PrY)+", Aux:"+str(Aux))
	P=np.sqrt(PuX**2+PuY**2) ; M=np.sqrt(PrX**2+PrY**2)/Aux
	return P,M

def readHQSSTR(filename):
	lines=open(filename,'r').readlines()
	# columns are: Pu M, Pu M std,Pr X, Pr Y, Pr X std, Pr Y std, V0, V1
	PuM,PrX,PrY,Aux=[],[],[],[]
	for l in lines[3:]:
		l=[ float(v) for v in l.split() ]
		PuM.append(l[0]) ; PrX.append(l[2]) ; PrY.append(l[3]) ; Aux.append(l[6])
	P=np.asarray(PuM) ; PrX=np.asarray(PrX) ; PrY=np.asarray(PrY) ; Aux=np.asarray(Aux)
	M=np.sqrt(PrX**2+PrY**2)/Aux
	return P,M

def readMap(filename): # read data from SSTR fiber map files
	matlabData=scipy.io.loadmat(filename)				# read matlab data file
	#print(matlabData.keys())
	posXs=matlabData['xmat'] ; posYs=matlabData['ymat']		# x,y position of each SSTR measurement
	pumpXs=matlabData['pump_x'] ; pumpYs=matlabData['pump_y']	# laser X and Y signal -> magnitude
	probeXs=matlabData['probe_x'] ; probeYs=matlabData['probe_y']
	aux1=matlabData['V0'] ; aux2=matlabData['V1'] # Aux 1 is from the sample, aux 2 is the references
	pumpMs=np.sqrt(pumpXs**2+pumpYs**2)				# magnitude, from in and out of phase signal
	probeMs=np.sqrt(probeXs**2+probeYs**2)
	probe={"R":probeMs,"M":probeMs,"X":probeXs,"Y":probeYs}[fitting]
	return posXs,posYs,pumpMs,probe,aux1 # WHY do we return pump and probe mag? maybe the calling user wants to do some filtering of crap data based on one or the other. TODO might consider importing aux data too?

sampleTemp=0 ; pumpPower=0 ; probePower=0 ; repRate=0 # i don't like it, but if additional info is in the header file, we can at least update these globals for the calling code to use
def readTDTRdata(filename):
	skip=2
	if ".dat" in filename: # these are GA tech's TDTR files
		data=np.loadtxt(filename,skiprows=17,encoding='ISO-8859-1')
		fms,ts,xs,ys=np.transpose(data)[:4,:]
		lines=open(filename,'r',encoding='ISO-8859-1').readlines()
		spotlines,preblock,postblock=lines[7],lines[12],lines[15]
		dpu,dpr=spotlines.split()[2:4] # pump, probe, power, diameter
		xi,yi=preblock.split()[:2] # Vin, Vout, with pump blocked, and probe blocked
		xf,yf=postblock.split()[:2]
		xs-=np.mean([float(xi),float(xf)]) ; ys-=np.mean([float(yi),float(yf)])
		global fm,rprobe,rpump
		if autofm:
			fm=np.mean(fms)*1e6
		if autorpr:
			rprobe=float(dpr)/2*1e-6
		if autorpu:
			rpump=float(dpu)/2*1e-6
		a=np.zeros(len(ts))+1
		return ts*1e-12,xs,ys,a
	

	data=np.loadtxt(filename,skiprows=2)
	#print(data,np.shape(data))
	npts,ncols=np.shape(data)
	if ncols==3: # files from TDTRfunc(save!=False)
		ts,xs,ys=np.transpose(data) ; aux=np.ones((len(ts),2))
		return ts,xs,ys,aux
	else:
		npts,nsets=np.shape(data)
		ps,ts,xs,ys,rs,phis=np.transpose(data)[:6,:] ; aux1=np.ones(len(ts)) ; aux2=np.ones(len(ts))
		if nsets>=8:
			aux1,aux2=np.transpose(data)[6:8,:]
		if nsets==10:
			xp,yp=np.transpose(data)[8:10,:]
			mp=np.sqrt(xp**2+yp**2)
			xs/=mp ; ys/=mp
		return ts*1e-12,xs,ys,np.asarray(list(zip(aux1,aux2)))
	
	"""
	with open(filename) as f: #TODO: betterify with csv import and list hacking?
  		rows = f.readlines()

	# handle files from TDTRfunc(save!=False)
	if len(rows[-1].split())==3: 
		ts,xs,ys=[],[],[]
		for r in rows:
			if "#" in r:
				continue
#				r=r.split(":")
#				if "fm" in r[0]:
#					fm=float(r[1])
#				if "rpu" in r[0]:
#					rpu=float(r[1])
#				if "rpr" in r[0]:
#					rpr=float(r[1])
#			else:
			r=r.split() ; t,x,y=r
			ts.append(float(t)) ; xs.append(float(x)) ; ys.append(float(y))
		ts=np.asarray(ts) ; xs=np.asarray(xs) ; ys=np.asarray(ys) ; aux=np.ones(len(ts))
		return ts,xs,ys,aux#,rpu,rpr,fm

	# or data files from the system, which has more columns
	ts=[];xs=[];ys=[];aux=[]
	for line in rows[2:]: #no data in first two lines of file
		line=line.split('\t')
		t,x,y=line[1:4] ; a="1"
		if "Aux" in rows[1]:
			a=(float(line[6]),float(line[7]))
		t=float(t) ; x=float(x) ; y=float(y) #; a=float(a)
		ts.append(t*1e-12) ; xs.append(x) ; ys.append(y) ; aux.append(a)

	ts=np.asarray(ts) ; xs=np.asarray(xs) ; ys=np.asarray(ys) ; aux=np.asarray(aux)
	return ts,xs,ys,aux#,rpu,rpr,fm
	"""

# ONE function for all filetypes, which reads TDTR headers or preductUncert (PU) data, reads HQ SSTR headers, or Fiber headers/radii.txt/filename. gathering up radii, and fmod. 
# file headers are either 1 line to the tune of "2/16/2022	11:42 AM	Pump: x =17.8m y = 15.6um   Probe: x = 9.5um y = 9.2um 	Freq (Hz): 8800000.000000"
# OR or a series of lines with "#" before various info, eg "#  rpump: 1e-05". You may be interested in testing/unitTest/ for validation of this. 
wavelength=0
def autos(filename):
	if ".dat" in filename: # TODO TDTR datafiles from GA tech are .dats, and i haven't add auto rpu/rpr/fm for those (yet)
		return
	if not autorpu and not autorpr and not autofm:
		return
	global autoFailed ; autoFailed=False ; vals={"rpu":[0,"",""],"rpr":[0,"",""],"fm":[0,"",""]} # "valname":(value,whereItCameFrom,set/failed/ignored)

	glos=["rpump","rprobe","fm"] ; keys=["rpu","rpr","fm"] ; autoglos=[autorpu,autorpr,autofm]

	lines=open(filename).readlines()
	if lines[0][0]=="#": # second case above (multi-line header, each line starts with a "#")
		for l in lines:
			if len(l)==0 or l[0]!="#":
				break
			for glo,key in zip( glos , keys ):
				if glo in l:
					vals[key]=[ float(l.split(":")[1]) , "header" , "" ]
	else:	# first case above, single-line header
		vardict=readDataHeader("\n".join(lines[:2])) # Why are we reading the first TWO lines? HQ SSTR has a 2-line header! and it doesn't hurt for TDTR to read 2 lines
		if vardict["freq"]!=0:
			vals["fm"] = [ vardict["freq"] , "header" , "" ]
		if vardict["pumpx"]!=0 and vardict["pumpy"]!=0:
			vals["rpu"] = [ (vardict["pumpx"]+vardict["pumpy"])/4 , "header" , "" ]
		if vardict["probex"]!=0 and vardict["probey"]!=0:
			vals["rpr"] = [ (vardict["probex"]+vardict["probey"])/4 , "header" , "" ]
		if vardict["Wave"]!=0: # SPECIAL CASE FOR DATA FROM VARIABLE-WAVELENGTH SYSTEMS, WAVELENGTH STORED IN FILE HEADER
			global wavelength ; wavelength=vardict["Wave"]
	conditionalPrint("autos","header results:"+str(vals))
	# also check for a radii file (used for SSTR)
	if mode in ["SSTR","FDTR"]: # TODO should we allow radii files to be used by TDTR? seems convenient! but also, if you have TDTR+SSTR data in the same folder, and want TDTR to use the header and SSTR to use radii.txt, you'd be SOL. should radii.txt be ignored if header was successful, or might we have successfully-read SSTR headers which we want to be ignored? 
		radfile=filename.split("/")[:-1] ; radfile.append("radii.txt") ; radfile="/".join(radfile)
		if os.path.exists(radfile):
			lines=open(radfile).readlines()
			for l in lines:
				for key in keys:
					if key in l:
						vals[key] = [ float(eval(l.split("=")[-1])) , "radii.txt" , "" ]
						
	conditionalPrint("autos","checked for radfile. results:"+str(vals))
	# Great, we've looked everywhere, and loaded what we've found into vals, including where the values came from (header vs radii.txt file). now we need to go through each measurement type and decide what's acceptable (eg, fiber SSTR should likely not get radii from the header, since they're usually wrong?)
	for glo,key,auto in zip( glos , keys , autoglos ):
		if auto and vals[key][0]==0:
			autoFailed=True
			vals[key][2]="failed"
		elif auto:
			setParam(glo,vals[key][0],warning=False) # setParam warns if the user manually sets rpu/rpr/fm. 
			vals[key][2]="set"
		else:
			vals[key][2]="ignored"

	conditionalPrint("autos",str(vals))


def readDataHeader(firstline): # TODO this needs a serious unit test...
	#read file header:  eg "2/10/2021 11:56 AM Pump: x =55um y = 55um 45 mW  Probe: x = 15um y = 15um  3 mW RR = 1MHz T=274.2K Freq (Hz): 450.996249" or "4mm 20x Power (mW): 35 Freq (Hz): 1000" or "pump:3.11 um, probe:3.64um 20x Power (mW): 28 Freq (Hz): 1000"
	firstline=firstline.replace(":"," ").replace("="," ").replace("("," ").replace(")"," ").replace("/",".").replace(","," ")
	chunks=firstline.split()
	vardict={ "pumpx":[0,["m"],-1] , "pumpy":[0,["m"],-1] , "probex":[0,["m"],-1] , "probey":[0,["m"],-1] , "temp":[0,["K","C","F"],-1] , "freq":[0,["Hz"],-1] , "reprate":[0,["Hz"],-1] , "pumpP":[0,["W"],-1] , "probeP":[0,["W"],-1] , "Wave":[0,["um"],-1] } # value's name,(value,[list,of,allowed,units],whichElementOfChunks)
	aliases={"temp":"t","reprate":"rr"} # [values name] might also be called [something else]
	unitPrefixes={"u":1e-6,"m":1e-3,"k":1e3,"M":1e6}
	def powerIn(searchString,searchList): # "if searchstring in list", but checks if searchstring is *in* each element, not just "equal to"
		truth=[searchString.lower() in element.lower() for element in searchList] # check "searchstring in e
		if True in truth:
			return truth.index(True)
		return -1
	# Only requirement is that the value name comes first in a set of 3 or more
	for key in vardict.keys():
		where=powerIn(key[:-1],chunks) # trim off "x" from "pumpx" and look for it
		if where==-1 and key in aliases.keys(): # if not present in any of the chunks, we'll try its alias
			alias=aliases[key]
			where=powerIn(alias,chunks)
		vardict[key][2]=where # whether we found it or not, log where in chunks it is, and move on.
	#print(vardict)
	for key in vardict.keys():
		conditionalPrint("readDataHeader",key)
		i=vardict[key][2] # index where this chunk starts
		if i<0:
			continue
		# index where all other chunks start, and index where this chunk ends (where the next chunk starts)
		js=np.asarray([vardict[k][2] for k in vardict.keys()]) ; js=js[js>i]
		j=len(chunks)
		if len(js)>0:
			j=min(js)
		#print(js,j)
		pieces=chunks[i:j]
		conditionalPrint("readDataHeader",str(pieces))
		#print(pieces)
		sub,unit,val,scale='','',0,1 # we're now ready to peruse for subname (eg, pumpx), units, and the value itself, scaling factor (mW-->1e-3)
		for p in pieces: # eg ['probe', 'x', '11.1um', 'y', '10.5um', '5.5mW']
			if "x" in p:			# "subnames", eg, "x" in, "Pump: x =55um y = 55um 45 mW"
				sub="x"
			if "y" in p:
				sub="y"
			for u in vardict[key][1]:	# eg "W" in "mW" TODO BUG: "m" for radius picks up on the m in mW for power! oof! 
				if u in p:
					unit=u ; scale=1
					i=p.index(u) # can't just do "for k, if k in p" because standalone unit m does not want milli prefix! 
					if i!=0:
						pref=p[i-1]
						for k in unitPrefixes.keys():
							if k == pref:
								scale=unitPrefixes[k]
								break
			
			num = "".join( [ c for c in p if c in "-0123456789.e" ] )
			# reject empty. reject "..". reject "ee". reject "e4". require at least one number
			if len(num)>0 and num!="e" and num.count(".")<2 and num.count("e")<2 and num[0]!='e' and sum([num.count(c) for c in "123456789"])>0:
				val=float(num)
			if (len(unit)>0 or len(vardict[key][1])==0) and val>0: # if this is a full set (number+unit)
				#print(key,sub,"=",val,"*",scale,unit)
				if vardict[key][0]==0 and (("x" not in key and "y" not in key) or sub in key):
				#	print(key,sub,"=",val,"*",scale,unit)
					vardict[key][0]=val*scale
				sub,unit,val,scale='','',0,1
			#print(num)
	vardict={key:vardict[key][0] for key in vardict.keys()}
	conditionalPrint("readDataHeader",str(vardict))	
	return vardict

def isNum(string):
	for c in string:
		if c not in "-0123456789.e":
			return False
	if len(string.replace(".",""))<len(string)-1:
		return False
	return True

def importMatrix(filename,overrides=''):
	conditionalPrint("importMatrix","importing thermal property matrix from file "+filename)
	f=open(filename,encoding="utf8")	# encoding flag needed on windows for some reason, to handle the special characters in the matrix file
	lines=f.readlines()			
	global tp
	tp=[]
	if "# thermal properties matrix and other parameters" in lines[0]: # THIS ISN'T A MATRIX FILE, IT'S A SYNTHETIC DATA FILE
		def parse(l):
			return [ float(v) for v in l.split(": ")[-1].replace("[","").replace("]","").split(",") ]
			
		for l in lines:
			if "Cs:" in l:
				Cs=parse(l)
			elif "Kzs:" in l:
				Kzs=parse(l)
			elif "ds:" in l:
				ds=parse(l)
			elif "Krs:" in l:
				Krs=parse(l)
			elif "Gs:" in l:
				Gs=parse(l)
		for i in range(len(Cs)):
			C,Kz,d,Kr=Cs[i],Kzs[i],ds[i],Krs[i]
			tp.append([C,Kz,d,Kr])
			if i!=len(Gs):
				G=Gs[i] ; R=1/G
				if useTBR:
					tp.append([R])
				else:
					tp.append([G])
		return

	for line in lines:			# for each line in the file
		if "C" in line and "Kz" in line and "Kr" in line: # skip first line "C, Kz, d, Kr..."
			continue
		if "#" in line:
			line=line.split("#")[0]
		line=line.strip()

		if len(line)<5:
			continue
		#comment=""
		#if "#" in line:
		#	line,comment=line.split("#")[:2]# ; comment=comment.strip() ; comment="# "+comment
		if ":" in line:
			line=line.split(":")[1]		# trim off leading "layer 1:" or "interface 2:"
			line=line.strip()
		if "," in line:
			line=line.split(",")
		else:
			line=line.split()		# split tab-delimited lines
		conditionalPrint("importMatrix","line: "+" ".join(line))
		if len(tp)%2==0 and len(line)!=4: # tp has an even number of lines, expecting another layer (4 entries)
			continue
		if len(tp)%2==1 and len(line)!=1: # tp has an odd number of lines, expecting an interface (1 entry)
			continue
		vals=[]
		for v in line:			# for each piece in the line, we'll eval() it (allowing matrix files to have variable names, eg, C_Al).
			conditionalPrint("importMatrix","found value:"+str(v)+","+str(type(v)))
			if "Kz" in v:
				vals.append(v)	# "Kz" can be passed for isotropic material props (Kr=Kz)
			else:
				vals.append(eval(v)) #TODO is this a security vulnerability? users can put arbitrary code in their input files!
		if len(tp)%2==1: # interface
			if (useTBR and vals[0]>1) or ((not useTBR) and vals[0]<1): # we want a TBR, but read in a TBC, or vice versa
				vals[0]=1/vals[0] #*1e9
		#print(line,"-->",vals)
		conditionalPrint("importMatrix","found row of vals:"+str(vals))
		tp.append(vals)
	if len(overrides)>0:
		for k in overrides.keys():
			setParam(k,overrides[k])
	conditionalPrint("importMatrix","found thermal property matrix:",pp=True)

def importRadii(filename):
	lines=open(filename).readlines()
	for l in lines:
		if "rpu" in l:
			rpu=float(eval(l.split("=")[-1]))
		if "rpr" in l:
			rpr=float(eval(l.split("=")[-1]))
		if "gamma" in l:
			global gamma
			gamma=float(eval(l.split("=")[-1]))
	if autorpu:
		setParam("rpu",rpu)
	if autorpr:
		setParam("rpr",rpr)
	warn("importRadii",{True:"using",False:"ignoring"}[autorpu]+" found rpu: "+str(rpu))
	warn("importRadii",{True:"using",False:"ignoring"}[autorpr]+" found rpr: "+str(rpr))

shiftPrezero=False
def normalize(ts,values,constantNorm=False,auxes=1): #normalize based on the point closest to time_normalize (beware, this means if your minimum time is, say, 1000ps, and normalizing set to 500ps, we norm to 1000ps)
	conditionalPrint("normalize",str(ts)+","+str(values)+" ("+str(time_normalize)+")")
	if shiftPrezero and min(ts)<shiftPrezero:
		values-=np.mean(values[ts<=shiftPrezero])
	#print("postshift",values)
	#return valuess
	if time_normalize=="" or time_normalize==-1 or time_normalize=="-1": #user is allowed to kill normalization
		return values
	if time_normalize=="aux":
		values/=auxes ; return values
	if time_normalize=="peak":
		values/=np.amax(values) ; return values
	# TODO for noisy data, the deviation of your normalization point will affect your results! 
	inorm=np.argmin(np.abs(ts-float(time_normalize))) #find closest point
	#print(inorm)
	normFact=values[inorm] #scale all by value there
	if constantNorm: #user may also choose to set a constant normalization factor
		normFact=constantNorm
	return values/normFact

# Jiang et al. Eq. 2.21 & 2.22. 
def phaseCorrect(xs,ys,ts,plotting=False,n=2): #apply phase correction: get pre-zero average, measure zero-crossing height in x and y, measure angle from these heights, and apply as an offset (change in Y signal across t=0 should be zero)
	conditionalPrint("phaseCorrect","x_orig,y_orig:"+str(xs)+","+str(ys))
	#return phaseCorrectBrute(xs,ys,ts)
	xtrim,ytrim,ttrim=xs[n:],ys[n:],ts[n:] # sometimes, the delay stage doesn't *quite* make it where it to where it should. sudden-drop in magnitude between 1st and 2nd point is then registered as max dM/dt, resulting in an indefined dphi, and a dataset full of nans
	if min(ts)>10e-12:
		return xs,ys
	# 1) attempt to detect rise start, ie, just before point of max slope
	mtrim=np.sqrt(xtrim**2+ytrim**2) # magnitude
	dMdt=np.gradient(mtrim,ttrim) # dM/dt is slope. we'll look for a max slope, and that's our initial rise
	tcut=ttrim[np.argmax(np.absolute(dMdt))]-2e-12 # 2ps prior to the point of our initial rise
	# 2) finding pre-zero average
	avgX=np.mean(xtrim[ttrim<=tcut]) ; avgY=np.mean(ytrim[ttrim<=tcut])
	# 3) finding humps (lcation and height) in X and Y
	imax=np.argmax(mtrim[ttrim<=tcut+20e-12]) # hump is point of max magnitude (whether we're mostly-good on phase (max in x) or not (max in y), and regardless of sign)
	maxX=xtrim[imax] ; maxY=ytrim[imax]
	# 4) measure height of the peak, calculate angle (since ΔY across t=0 should be zero), and apply as an offset
	dx=maxX-avgX;dy=maxY-avgY
	dphi=-np.arctan2(dy,dx)#-0.003*np.pi*2 # phase offset is the angle (height of hump in x vs y)
	conditionalPrint("phaseCorrect","dphi="+str(dphi))
	if np.isnan(dphi):
		dphi=0
	#xfixed=xs*np.cos(dphi)-ys*np.sin(dphi) # Braun "The role of compositional..." Eq 3.68
	#yfixed=ys*np.cos(dphi)+xs*np.sin(dphi)
	zfixed=(xs+1j*ys)*np.exp(1j*dphi) ; xfixed=zfixed.real ; yfixed=zfixed.imag # value*eⁱᶱ is the same as rotating by θ
	if ("phaseCorrect" in verbose) or plotting:
		t_pre=ts[ts<=tcut] ; x_pre=np.ones(len(t_pre))*avgX ; y_pre=np.ones(len(t_pre))*avgY
		t_max=ts[imax]
		Xs=[ts[ts<50e-12],ts[ts<50e-12],ts[ts<50e-12]    ,ts[ts<50e-12]    ,t_pre,t_pre,[t_max,t_max]]
		Ys=[xs[ts<50e-12],ys[ts<50e-12],xfixed[ts<50e-12],yfixed[ts<50e-12],x_pre,y_pre,[maxX,maxY]  ]
		lplot(Xs, Ys, "t (s)","R/M/X/Y",title="", labels=["X_o","Y_o","X_c","Y_c"," "," "," "], markers=['k-','g-','b-','r-','r.','r.','ro'])#,filename=altFnames)
	#xfixed,yfixed=phaseCorrect(xfixed,yfixed,ts)
	conditionalPrint("phaseCorrect","xfixed,yfixed:"+str(xfixed)+","+str(yfixed))
	return xfixed,yfixed #,ts

def phaseCorrectBrute(xs,ys,ts): # same as above, but we basically just brute force, check all phase angles
	def swingPhi(dphi):
		xfixed=xs*np.cos(dphi)-ys*np.sin(dphi) # Braun "The role of compositional..." Eq 3.68
		yfixed=ys*np.cos(dphi)+xs*np.sin(dphi)
		return xfixed,yfixed
	best=0 ; bestMSE=np.inf
	for i,dphi in enumerate(np.linspace(-np.pi/2,np.pi/2,500)):
		xf,yf=swingPhi(dphi)
		t_plot=ts[ts<=20e-12] ; x_plot=xs[ts<=20e-12] ; y_plot=ys[ts<=20e-12]
		xf=xf[ts<=20e-12] ;  yf=yf[ts<=20e-12]
		mse=MSE(yf,np.ones(len(yf))*np.mean(yf)) # a "smooth" yfixed should have the lowest MSE to its mean
		lplot( [t_plot,t_plot,t_plot,t_plot], [x_plot,y_plot,xf,yf], labels=["X_o","Y_o","X_c","Y_c"], markers=['-']*4, title=str(dphi)+","+str(mse), filename="pics/"+str(i)+".png" )
		if mse<bestMSE:
			bestMSE=mse
			best=dphi
	#print(best,bestMSE)
	#print(solvedParams)
	#xfixed,yfixed=swingPhi(solvedParams[0])
	xfixed,yfixed=swingPhi(best)
	lplot([ts,ts,ts,ts], [xs,ys,xfixed,yfixed], labels=["X_o","Y_o","X_c","Y_c"], markers=['-']*4)
	return xfixed,yfixed

def getRowCol(paramName): #given a parameter name ("Kz2" for example), return the row and column for it in thermal properties matrix, tp
	translateDict={"C":0,"KZ":1,"G":0,"R":0,"D":2,"KR":3}
	param=paramName[:-1].upper()
	if param not in translateDict:
		return 100,100
	col=translateDict[param]
	layer=paramName[-1]
	if layer not in ["1","2","3","4","5","6","7","8","9"]:
		return 100,100
	layer=int(layer)
	row=(layer-1)*2
	if param=="G" or param=="R":
		row=row+1
	if row>=len(tp) or col>=len(tp[0]):
		return 100,100
	return row,col

def isTPparam(paramName): # detect params that edit "tp" global, e.g. "d1", "Kz2", "R1", but not "rpu", "rpr", "gamma"
	for i,c in enumerate(paramName): 					# start by finding index of first number
		if c in "123456789":
			param,laynum=paramName[:i],paramName[i:]		# split by that index, "Kz2" --> "Kz","2"
			if param not in ["Kz","Kr","C","G","R","d"]:		# "JUNK1" is not a valid paramName
				return False
			if False in [ c in "0123456789" for c in laynum ]:	# "Kz1wow" is not a valid paramName
				return False
			return True						# if both pieces are legit "Kz","2" etc, then return True
	return False # OR, if you cycled through all characters, didn't find a number, it's not a tp

# TODO I think we have found some sanity on setVar vs setParam and getVar vs getParam: setParam does it all now, and setVar is just a wrapper. we could get rid of it, but we'll keep it around in case old code stull uses it. TODO NEEDS SUPER MEGA THOROUGH TESTING THOUGH
# (historically, setParam was meant for internal-use only, for setting things that might be set during fitting, e.g. "Kz2" which would update the "tp" thermal properties global. setVar was meant for external-use only, for setting global variables which would be easily-enough set internally via "global {gloName}" etc. And it was up to the user to keep track of which things are params vs glos. in reality though, it was common practice externally to, for example, "importMatrix(matfile); setParam('d1',d1)" to customize the thermal properties matrix that was imported. conversely, it was common practice internally to, for example "setVar(varname)" to take advantage of the indirection (where we have the global's name as a string). 
paramAliases={"rpu":"rpump", "rpr":"rprobe", "fm":"fm", "fp":"fp", "da":"depositAt", "ma":"measureAt", 
"sphase":"slopedPhaseOffset","phase":"variablePhaseOffset"}
def setParam(paramName,value,warning=True): # setParam: during fitting, we update things (by name), expect the relevant globals to be updated, and then the model is regenerated (e.g. TDTRfunc generating the TDTR curve), iteratively, until a good fit is achieved. This should handle thermal properties by name (e.g. "Kz2" should update the thermal property matrix ("tp" global) 3rd row 2nd column. "rpu" on the other hand will update the "rpump" global). 
	# Step 1: 
	if isTPparam(paramName):	
		r,c=getRowCol(paramName)  #to fit "Kz1" -> lookup list "Kzs", and set index 0 (first layer's)
		tp[r][c]=value
		return
	# Step 2: if we haven't returned, we must have a param that doesn't go in "tp" global. check aliases, and set the globals
	if paramName in paramAliases.keys():
		paramName=paramAliases[paramName]
	#if isinstance(value,str) and "{" in value:
	#	value=dict(value)
	# Step 3: set globals
	globals()[paramName]=value
	# Step 4: also issue warnings if the user is trying to write to a parameter that *may* be read in via autos()
	for pname,autoparam,flagname in zip(["rpump","rprobe","fm"],[autorpr,autorpu,autofm],["autorpr","autorpu","autofm"]):
		if warning and paramName==pname and autoparam:
			warn("setParam","WARNING: you set "+pname+", but did not turn off automatic reading from the file (try setParam(\""+flagname+"\",False))")
			# print(traceback.format_stack()) # USE THIS TO FIND WHERE FALSE-POSITIVE WARNINGS COME FROM
			
	#if paramName=="rpump" and autorpu:
def getParam(paramName):
	if isTPparam(paramName):
		r,c=getRowCol(paramName)
		val = tp[r][c] #tofit's "Kz1" -> lookup list "Kzs", and read index 0 (first layer's) 
		if val == "Kz" and paramName in tofit:		  # SPECIAL CASE: USER IS FITTING FOR (OR RUNNING SENSITIVITY FOR) Kr,
			r,c=getRowCol(paramName.replace("r","z")) # BUT DOESN'T HAVE A SPECIFIC VALUE SET. THEY PROBABLY WANT TO INHERIT
			val=tp[r][c]				  # THE ISOTROPIC VALUE. (e.g. see contributions of z vs r to iso sens)
		return val	# "gotcha" to beware of: does this double count? doesn't look like it.check sensitivity for Kr1,Kz1, or Kz1,R1, with 
	if paramName in paramAliases.keys():
		paramName=paramAliases[paramName]
	return globals()[paramName]

def setVar(var,val,warning=True):
	setParam(var,val,warning)
def getVar(var):
	return getParam(var)
def setVars(paramValPairs):
	for k in paramValPairs.keys():
		setVar(k,paramValPairs[k])

def setTofitVals(values): #given a list of values corrosponding to the variable names in tofit, set them in thermal properties matrix, tp
	setParams(tofit,values)
def setParams(params,values):
	for p,v in zip(params,values):
		setParam(p,v)


def getScaleUnits(param):
	unitsDict={'rpum':'um','rprob':'um','Kz':'W m^-1 K^-1','Kr':'W m^-1 K^-1','G':'MW m^-2 K^-1','C':'MJ m^-3 K^-1','d':'nm','R':'m^2 K GW^-1',"gamm":'(-)',"alph":"m","exp":"-"}
	factDict={'rpum':1e6,'rprob':1e6,'Kz':1,'Kr':1,'G':1e-6,'C':1e-6,'d':1e9,'R':1e9,'gamm':1e-3,'phas':1,'alph':1,'exp':1}

	if param in paramAliases.keys():
		param=paramAliases[param]
	name=param[:-1]
	return factDict.get(name,1),unitsDict.get(name,"-") # default to scalefactor of 1, and unitless, if unrecognized param??




"""
def setParams(paramNames,values): # TODO i don't like this. should pass a dict, not two lists. two lists is dumb and gross.
	for p,v in zip(paramNames,values):
		conditionalPrint("setParams","setting "+p+" = "+str(v))
		setParam(p,v)

nonTPparams={"rpu":"rpump", "rpr":"rprobe", "fm":"fm", "fp":"fp", "da":"depositAt", "ma":"measureAt", 
"gamma":"gamma", "tshift":"tshift", "chopwidth":"chopwidth","yshiftPWA":"yshiftPWA",
"sphase":"slopedPhaseOffset","phase":"variablePhaseOffset","alpha":"alpha",
"expA":"expA","expB":"expB","expC":"expC"}
def setParam1(paramName,value): #given a param name, set it in the thermal properties matrix (or appropriate global for radii). note: meant for internal calls only! and really just for those values found in tofit ("K--","G-","C-","rpu","rpr") TODO should setting rpu or rpr automatically set autorpu or autorpr to False? TODO should we have one variable "r" which, which set, sets rpu=rpr to fit for both together?
	if paramName in nonTPparams.keys():
		varname=nonTPparams[paramName]
		#globals()[varname]=float(value)
		globals()[varname]=value
	elif paramName[:-1] in ["Kz","Kr","C","G","R","d"]:
		r,c=getRowCol(paramName)  #to fit "Kz1" -> lookup list "Kzs", and set index 0 (first layer's)
		tp[r][c]=value
	else:
		warn("setParam","WARNING: unrecognized paramName "+paramName+" (skipping)")
		return
	if paramName=="rpu" and autorpu:
		warn("setParam","WARNING: you set pump radius, but did not turn off automatic reading from the file (try setVar(\"autorpu\",False))")
	if paramName=="rpr" and autorpr:
		warn("setParam","WARNING: you set probe radius, but did not turn off automatic reading from the file (try setVar(\"autorpr\",False))")
	if paramName=="fm" and autofm:
		warn("setParam","WARNING: you set modulation frequency, but did not turn off automatic reading from the file (try setVar(\"autofm\",False))")


def getParam1(paramName): #reverse of setParam
	if paramName in nonTPparams.keys():
		varname=nonTPparams[paramName]
		return globals()[varname]
	else:
		r,c=getRowCol(paramName)
		val = tp[r][c] #tofit's "Kz1" -> lookup list "Kzs", and read index 0 (first layer's) 
		if val == "Kz" and paramName in tofit:		  # SPECIAL CASE: USER IS FITTING FOR (OR RUNNING SENSITIVITY FOR) Kr,
			r,c=getRowCol(paramName.replace("r","z")) # BUT DOESN'T HAVE A SPECIFIC VALUE SET. THEY PROBABLY WANT TO INHERIT
			val=tp[r][c]				  # THE ISOTROPIC VALUE. (e.g. see contributions of z vs r to iso sens)
		return val	# "gotcha" to beware of: does this double count? doesn't look like it.check sensitivity for Kr1,Kz1, or Kz1,R1, with Kr1="Kz". run this with verbose=["SSTRfunc"] to make sure both are set in iso case, and only one in aniso case)

def setVars(names,values): # TODO i don't like this. should pass a dict, not two lists. two lists is dumb and gross.
	for n,v in zip(names,values):
		setVar(n,v)
def setVar(name,value): #meant for external calls only! literally just setting variables. TODO should have some means of checking if a variable exists or not (did the caller intend to call setParam instead?). note: checking globals() doesn't appear to work 100%
	globals()[name] = copy.deepcopy(value)
	conditionalPrint("setVar","setting "+name+"="+str(value))
	#deep copy prevents fitting code from setting a calling function's thermal property matrix! consider the following:
	#>>>a=[[1,2,3,4][2,3,4,5],[3,4,5,6]]
	#>>>def setElement22(l,v):
	#>>>	l[2][2]=v
	#>>>setElement22(a,7)
	#>>>print(a # --> [[1,2,3,4][2,3,4,5],[3,4,7,6]])
	#other common strategies for copying, like passing in a[:] or list(a) or copy.copy(a) aren't deep enough! since the elements of "a" are themselves objects

def getVar(name): #useful for external calling: your calling script's "fm" won't necessarily match module TDTR_fitting's "fm". (eg, after readTDTR set it for you)
	return globals()[name]
"""

def getTofitVals():#read the values corrosponding to the variable names in tofit, from thermal properties matrix
	values=[]
	for name in tofit:
		values.append(getParam(name))
	return values

customBounds={} #; customBounds['Kz2']=[32.,34.]
#customBounds={"Kz2":[32,34]}
def lookupBounds():
	bnds=[[],[]] #set up bounds (done based on fitting parameter type (Ks use different bounds from Cs)
	if len(customBounds.keys())>0:
		conditionalPrint("lookupBounds","custom bounds: "+str(customBounds)) # +" (keys: "+str(customBounds.keys())+")")
	for param in tofit:
		conditionalPrint("lookupBounds","checking parameter "+param+".")
		if isTPparam(param):
			for c in "1234567890":
				param=param.replace(c,"")
		if param in paramAliases.keys():
			param=paramAliases[param]
		lb=lbs[param];ub=ubs[param]
		if param in customBounds.keys():
			lb,ub=customBounds[param]
			conditionalPrint("lookupBounds","using custom bounds:"+str(lb)+"<"+param+"<"+str(ub))
		bnds[0].append(lb) ; bnds[1].append(ub)
	return bnds
"""
def lookupBounds():
	bnds=[[],[]] #set up bounds (done based on fitting parameter type (Ks use different bounds from Cs)
	for param in tofit:
		#print(tofit,param)
		p=param
		if param not in nonTPparams.keys():
			p=p[:-1]
		#else:
		#	p=nonTPparams[p]
		#lb=g*0.05;ub=g*20.
		lb=lbs[p];ub=ubs[p]
		#lb=g*.125 ; ub=g*2.0
		if param in customBounds.keys():
			lb,ub=customBounds[param]
			conditionalPrint("lookupBounds","using custom bounds:"+str(lb)+"<"+param+"<"+str(ub))
		bnds[0].append(lb);bnds[1].append(ub) #guesses=[var1guess,var2guess,var3guess...], bounds=((var1lower,var2lower,...)(var1upper,...))
		
		if "K" in param:
			zr=param[1] ; n=param[2:] ; other={"z":"r","r":"z"}[zr] ; other="K"+other+n
			#print(other)
			if other not in tofit and not isinstance(getParam(other),str): # fit both? or fitting Kz, and Kr=="Kz"? normal bounds
				conditionalPrint("lookupBounds","anisotropic bounds enacted: "+str(param))
				anisobounds=list(sorted( [ getParam(other)*v for v in LBUB["anisotropy"] ] ))
				#print(bnds,anisobounds)
				bnds[0][-1]=max([bnds[0][-1],anisobounds[0]]) # lower bound from lbs, vs lower bound based on anisotropy
				bnds[1][-1]=min([bnds[1][-1],anisobounds[1]])
	conditionalPrint("lookupBounds","found bounds: "+str(bnds))
	return bnds
"""

def prettyPrint(pop=True): # TODO: "print("a = %6.2f +/- %4.2f" % (a_opt, Da))" -> "a =   2.02 +/- 0.06" we should take advantage of this
	if pop:
		popGlos() #populates Cs, Kzs, etcetera. it may have already been done, but it's cheap to redo. TODO?
	print("  Cs: "+str(Cs)+"\n  Kzs: "+str(Kzs)+"\n  ds: "+str(ds)+"\n  Krs: "+str(Krs)+
	   "\n  Gs: "+str(Gs)+"\n  (Rs: "+str(Rs)+")\n  f mod,pulse: "+str(fm)+", "+str(fp)+
	   "\n  r pump,probe: "+str(rpump)+", "+str(rprobe)+" ("+pumpShape+")"+
	   "\n  tm,tn: "+str(minimum_fitting_time)+", "+str(time_normalize)+
	   "\n  nmax: "+str(nmax)+
	   "\n  da,ma,A1,gamma,phase: "+str(depositAt)+", "+str(measureAt)+", "+str(A1)+", "+scientificNotation(gamma)+#", "+scientificNotation(slopedPhaseOffset)+
	   "\n tofit: "+str(tofit)+"\n fitting: "+str(fitting)+
	   "\n chopwidth: "+str(chopwidth)+"\n tshift: "+str(tshift)+"\n yshiftPWA: "+str(yshiftPWA))
	print("ids:",id(Cs),id(Kzs),id(ds),id(Krs),id(Gs))

"""
def solveSimultPlotting(lsqout,ts,data,listOfFiles,plotting="show",paramString=""): #"plotting" options include: show, save, none. (paramString is shown instead of "tofit[0]=val,..."
	#compute decay curves (extract from lsqout, lsqout["fun"] is already dz=data-func)
	funcPts=np.asarray(data)-lsqout["fun"].reshape((len(data),len(ts)))
	#compute R² fit values
	residuals=[];resList=""
	for f,d in zip(funcPts,data):
		residual=error(d,f)
		residuals.append(residual)
		#print(resList,type(resList),scientificNotation(residual),type(scientificNotation(residual)))
		resList=resList+scientificNotation(residual)+" "
	if plotting=="none": #if no plotting, return residuals, else, construct plot
		return residuals
	#assemble list of values we found (for the plot)
	if len(paramString)>0:
		calcLabel=paramString
	else:
		calcLabel=""
		for i in range(0,len(tofit)): #for each fitting parameter
			calcLabel=calcLabel+tofit[i]+"="+str(scientificNotation(lsqout['x'][i]))+" "
		#val=lsqout['x'][i] #the value we found
		#expo=int(np.floor(np.log(val)/np.log(10.))) #express as scientific notation; extract the exponential
		#val=str(round(val/10.**expo,2))+"e"+str(expo)
		#calcLabel=calcLabel+tofit[i]+"="+str(val)+" "
	#assemble markers (dots for datapoints, lines for generated curve)
	mkrs=['.']*len(data)+['-']*len(data)
	#assemble labels for legend
	labels=[]
	for f in listOfFiles:
		fname=f.split("/")[-1]
		if len(fname)>40:
			fname=fname[:40-len(fname.split('_')[-1])]+"..."+fname.split('_')[-1]
		labels.append(fname)
	for f in listOfFiles:
		fmod=".".join(f.split("_")[-1].split(".")[:-1]).replace("MHz","e6") #11122020_MURI42_1_152133_8.80MHz.txt -> 8.80MHz  in PLSB or 12172019_Si-Kr_500kV-1e12_2_163315_8400000.txt -> 8400000 in HQ
		#fmod=scientificNotation(int(fmod),2)
		labels.append(calcLabel+", "+fmod)
	title=listOfFiles[0].split("/")[-1] #media/Media/Class stuff/U Virginia/Research/DATA/Pfeifer_Thomas/2019_12_17_HQ/12172019_Si-Kr_500kV-1e12_2_163315_8400000.txt -> 12172019_Si-Kr_500kV-1e12_2_163315_8400000
	title="_".join(title.split('_')[1:4]) #12172019_Si-Kr_500kV-1e12_2_163315_8400000 -> Si-Kr_500kV-1e12_2
	figFile=""
	if plotting=="save":
		figFile="/".join(listOfFiles[0].split("/")[:-1])+"/"+callingScript+"/pics/"+title+".png"
		#print(figFile)
	#if len(altFnames)>0:
	#	figFile=altFnames+[figFile]
	#other plotting business
	ylabel={"R":"Ratio (-X/Y)","M":"Mag (uV)","X":"X (uV)","Y":"Y (uV)"}[fitting]
	xdata=[ts]*(len(data)*2) ; ydata=list(data)+list(funcPts)
	lplot(xdata, ydata, "time delay (s)", ylabel, markers=mkrs, labels=labels, filename=figFile, title=title+", R^2="+resList)
	return residuals
"""

def runningAverage(Ys,N):
	if N<2:
		return Ys
	A=np.zeros((len(Ys),N))
	for i in range(N):
		A[:,i]=np.roll(Ys,i-int(N/2))
	return np.mean(A,axis=1)

def scientificNotation(val,decimals=2):
	if val==0.0:
		return "0.0"
	expo=int(np.floor(np.log(abs(val))/np.log(10.))) #express as scientific notation; extract the exponential
	return str(round(val/10.**expo,decimals))+"e"+str(expo)
def sigFigs(val,SF=3):
	val=scientificNotation(val,SF-1) # as a string, with sig fig rounding: 123456.789101987 -> "1.23e6"
	val=float(val)
	val=str(val)
	return val
def lcs(strings): #Longest Common Substring: see TDTR_fitting/testing26
	subs=[]
	a=strings[0]
	for i in range(len(a)): #for each starting point in one string
		for l in range(len(a)-i): #for each substring starting at that point
			for b in strings[1:]: #for each other string in set
				if a[i:i+l+1] not in b: #if sub not in it, quit now
					break
			else: #if we made it through without quitting (then substring exists in all other strings in set)
				subs.append(a[i:i+l+1]) #add to the list
	for i in reversed(range(len(subs))): #filter out substring which are entirely numbers (eg, ignore 8400000 if longest shared substring)
		for c in subs[i]:
			if c not in "1234567890":
				break
		else:
			del subs[i]
	return max(subs,key=len) #return substring with longest length

def setFigDPI(dpi):
	#sys.path.insert(1,"../niceplot")
	from niceplot import setPlotRC ; from nicecontour import setContRC
	#setDPI(dpi)
	setPlotRC('figure',{'dpi':dpi})
	setContRC('figure',{'dpi':dpi})
#def setFigNames(names):
#	figNames(names)
#def pltclf():
#	 killFigAx()
#def pltclf():
#	clf()

import matplotlib
import matplotlib.pyplot as plt #https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.html
fig=plt.figure() ; matplotlib.rcParams['figure.dpi'] = 200 ; fig.set_size_inches(8.0, 6.0)
def setFigPx(px,py):
	if py==0:
		py=py=int(px*6/8)
	global fig ; fig.set_size_inches(px/75, py/75)

### END VARIOUS OTHER HELPERS ###

### OTHER TOOLS ###
from multiprocessing.sharedctypes import Array
counterNames=["resDecay","resKZFDecay","brutehelper","solveSimultHelper","TDTRfunc","decayKZF","solve","PWAfunc","solveKZFSimultaneous"]
counters=Array('i',[0,0,0,0,0,0,0,0,0],lock=True)

def resetCounters():
	global counters
	with counters.get_lock():
		for i in range(len(counterNames)):
			counters[i]=0

def incrementCounter(funcName): # lock-protected writing to shared memory objects across workers: https://docs.python.org/3/library/multiprocessing.html#multiprocessing.sharedctypes.Array
	global counters
	i=counterNames.index(funcName)
	with counters.get_lock():
		counters[i]+=1
		conditionalPrint("c_"+funcName,"incrementCounter: "+funcName+"+=1 > "+str(counters[i]))

def conditionalPrint(funcName,message,pp=False):
	if funcName in verbose:
		print("TDTR_fitting > "+funcName+": "+message)
		if pp and funcName!="popGlos": # prettyPrint calls popGlos, so let's not cause infinite recursion. 
			prettyPrint()
def warn(funcName,message):
	if not quietWarnings:
		print("TDTR_fitting > "+funcName+": "+message)

def resDecay(parameterValues,ts,data):
	incrementCounter("resDecay")
	res=error(data,func(ts,*parameterValues))
	#print(parameterValues,"-->",res)
	return res

def resKZFDecay(parameterValues,ts,data):
		global tp
		tp_o=copy.deepcopy(tp)
		#print("*parameterValues",*parameterValues)
		with open("resKZFdecay.log",'a') as f:
			f.write(str(parameterValues)+"\n")
		#print("PRE DECAY") ; prettyPrint()
		res=error(data,decayKZF(ts,*parameterValues))
		incrementCounter("resKZFDecay")
		#print("POST DECAY") ; prettyPrint()
		tp=copy.deepcopy(tp_o)
		#print("POST RESTORE") ; prettyPrint()
		#print(parameterValues,"-->",res)
		return res

def brutehelper(parameterValues,listOfDecays,ts,fms):
	incrementCounter("brutehelper")
	return sum(solveKZFSimHelper(parameterValues,listOfDecays,ts,fms)**2)

# a shared function used by genContour2D (either traditional 2D contours if len(tofit)==2, or collapsed ND contours where we test permutations of tofit[:2] and fit for the remaining), and used by genContour3D (tranditional 3D contours testing all permutations where len(tofit)==3)
def contourDefaults(fileIn,fileOut,paramRanges,paramResolutions,solveFunc):
	#DEFAULTING IN OR PARAM RANGES AND RESOLUTIONS, AND OUTPUT FILE NAME(s), 2D AND 3D
	nParams=len(tofit)
	#if nParams>3:
	#	warn("generateHeatmap","ERROR: heatmaps only allowed with 2 or 3 fitting parameters!")
	#	return ""
	if len(fileOut)==0:
		fo=fileIn
		if type(fileIn) == list:
			fo=combinedFilename(fileIn)
		dirOut=fo.split('/')[:-1]+[callingScript,"contourfiles"]
		fileOut="/".join(dirOut)+"/"+fo.split('/')[-1]+"_"+"_v_".join(tofit)+"_"+fitting+".txt"
		warn("generateHeatmap","no fileOut specified: defaulting to "+fileOut)
	if len(paramRanges)==0:
		warn("generateHeatmap","no paramRanges specified: solving...")
		r=solveFunc["func"](fileIn,plotting="none",**solveFunc["kwargs"])[0]
		ps="no paramRanges specified: found "
		for name,val in zip(tofit,r):
			ps=ps+name+"="+str(val)+", "
		ps=ps+", setting paramRanges to +/- 50%"
		warn("generateHeatmap",ps)
		paramRanges=np.zeros((nParams,2))
		paramRanges[:,0]=r*.5 ; paramRanges[:,1]=r*1.5
		#paramRanges[:,0]=r*.01 ; paramRanges[:,1]=r*10
	if isinstance(paramResolutions,(int,float)):
		paramResolutions=[int(paramResolutions)]*nParams
	if len(paramResolutions)==0:
		if nParams==2: # S={3:20,2:40}[nParams]
			S=40
		else:
			S=20
		paramResolutions=[S]*nParams
		warn("generateHeatmap","no paramResolutions specified: defaulting to "+"x".join(list(map(str,paramResolutions))))
	return fileOut,paramRanges,paramResolutions

# fka generateHeatmap, which used to do 2D or 3D. we have since split this apart into genContour2D and genContour3D. similarly, displayHeatmap has been split into displayContour2D and displayContour3D. We also incorporated "Wafer Bonding/isSSTRneeded.py > flat3DContour" into genContour2D, where we can basically "flatten" an N-dimensional parameter space into 2D. instead of "for each combination of 2 fitted parameters, check the residual" it is "for each combination of 2 (or 2 or more) fitted parameters, run fitting (on the remaining parameters) and check the residual". This is similar in spirit to measureContour1Axis, which is "for a range of values for 1 fitted parameter, run fitting on the remaining". This is all tested via testing61.py
def genContour3D(fileIn,fileOut='',paramRanges='',paramResolutions='',overwrite=False,ND="N"): #[[parameter1lowerbound, parameter1upperbound], [parameter2lowerbound, parameter2upperbound]]
	# FIRST we allow passing in a list of files (and out a list of files), eg, if you did simultaneous fitting (ss2) on a list of files, you might also like to generate contours for the same list of files
	if type(fileIn)==list:
		filesOut=[]
		# process vars n magicModifiers file
		settables={}
		for f in fileIn:
			if "magic" in f:
				processMagic(f,settables)
		fileIn=[ f for f in fileIn if "magic" not in f ]
		
		for i,f in enumerate(fileIn):
			for var in settables.keys():
				setVar(var,settables[var][i])
			conditionalPrint("generateHeatmap","we received a list. handling individual file #"+str(i+1)+" : "+f,pp=True)
			fo=genContour3D(f,'',paramRanges,paramResolutions,overwrite,ND)
			conditionalPrint("generateHeatmap","found output file: "+str(fo))
			filesOut.append(fo)
		return filesOut

	fileOut,paramRanges,paramResolutions=contourDefaults(fileIn,fileOut,paramRanges,paramResolutions,{"func":solve,"kwargs":{}})
	if not overwrite and os.path.exists(fileOut+"_0"):
		warn("generateHeatmap","overwrite=False and fileOut exists, skipping generations")
		return fileOut
	

	#BOUNDS AND RANGES IN X, Y, Z
	xlb=paramRanges[0][0];xub=paramRanges[0][1];xr=paramResolutions[0]
	ylb=paramRanges[1][0];yub=paramRanges[1][1];yr=paramResolutions[1]
	zlb=paramRanges[2][0];zub=paramRanges[2][1];zr=paramResolutions[2]

	#IMPORT DATA
	ts,data=readFile(fileIn)

	#ITERATE ACROSS ALL COMBINATIONS OF VALUES: USE SCIPY.BRUTE IF 3D OR IF nX==nY, OTHERWISE, FOR-LOOPS
	x0, fval, grid, jout = brute(resDecay, paramRanges, args=(ts,data), Ns=paramResolutions[0], full_output=True, workers=4, finish=None) #You can use scipy.optimize.brute IF paramResolutions are the same. FYI: workers=1 does NOT run as single-threaded. if that's what you're after (because you're doing ridiculous global-setting shenanigans), you'll actually want to force it with paramResolutions
	residuals=jout

	#SAVING IT OFF. HEADER(s) CONTAIN PARAM RANGES AND RESOLUTIONS
	os.makedirs("/".join(fileOut.split('/')[:-1]),exist_ok=True)
	header=[]
	header=tofit[0]+"="+str(xlb)+":"+str(xr)+":"+str(xub)+" , "+tofit[1]+"="+str(ylb)+":"+str(yr)+":"+str(yub)
	#print(residuals)
	header=header+" , "+tofit[2]+"="+str(zlb)+":"+str(zr)+":"+str(zub)
	for n in range(paramResolutions[0]):
		np.savetxt(fileOut+"_"+str(n), residuals[:,:,n], delimiter=",",header=header)
	return fileOut

# TODO TODO TODO how should genContour2D receive a list of types when passing a list of files? currently, ss2 receives a list of types when you give it the multiple files. predictUncert receives them in a dict too (for what file types to create). previously, genContour2D assumed based on ss2Types and shit, and/or magicMods global. but that shit's nasty. 
# fka generateHeatmap, which used to do 2D or 3D. we have since split this apart into genContour2D and genContour3D. similarly, displayHeatmap has been split into displayContour2D and displayContour3D. We also incorporated "Wafer Bonding/isSSTRneeded.py > flat3DContour" into genContour2D, where we can basically "flatten" an N-dimensional parameter space into 2D. instead of "for each combination of 2 fitted parameters, check the residual" it is "for each combination of 2 (or 2 or more) fitted parameters, run fitting (on the remaining parameters) and check the residual". This is similar in spirit to measureContour1Axis, which is "for a range of values for 1 fitted parameter, run fitting on the remaining". This is all tested via testing61.py
def genContour2D(fileIn,fileOut='',paramRanges='',paramResolutions='',overwrite=False,solveFunc=None,settables={}): #[[parameter1lowerbound, parameter1upperbound], [parameter2lowerbound, parameter2upperbound]]
	tofit=getVar("tofit") ; tf=getVar("tofit")
	nParams=len(tofit)

	# Something to consider: overlapped 2D contours may NOT represent the actual uncertainty for multi-technique fitting, in the flattened 3D case. 
	# Why? consider SSTR+TDTR, both 3D contours will generate volumes, and we care about the boolean intersection of these volumes. 
	# flattening each will "project" the volume onto the axis plane, and to overlap there will over-estimate uncertainty. 
	# stated differently, consider the 2D flattened contour for SSTR: set 2 parameters, fit for the 3rd, you will *always* find a good fit,
	# meaning the flattened contour bounds for SSTR are "all values". overlap "all values" with "whatever TDTR finds" would (wrongly) suggest
	# that SSTR+TDTR is no better than TDTR alone. 
	if solveFunc is None: # measureContour1Axis is the rare exception to *just* using mode: eg, we still want to be able to pass in solveKZF shit
		#solveFunc={"TDTR":solveTDTR,"SSTR":solveSSTR,"FDTR":solveFDTR,"PWA":solvePWA,"FD-TDTR":solveSimultaneous}[mode]
		solveFunc={"func":solve,"kwargs":{}}

	if isinstance(fileIn,list) and solveFunc["func"]==solve:
		filesOut=[]

		#settables={}
		for f in fileIn:
			if "magic" in f:
				processMagic(f,settables)
		fileIn=[ f for f in fileIn if "magic" not in f ]

		conditionalPrint("generateHeatmap","processed magicMods: "+str(settables))
		for i,f in enumerate(fileIn):
			for var in settables.keys():
				setVar(var,settables[var][i])
			conditionalPrint("generateHeatmap","we received a list. handling individual file #"+str(i+1)+" : "+f,pp=True)
			fo=genContour2D(f,'',paramRanges,paramResolutions,overwrite,solveFunc)
			conditionalPrint("generateHeatmap","found output file: "+str(fo))
			filesOut.append(fo)
		return filesOut

	#file_s=fileIn[:]
	#if isinstance(fileIn,list):
	#	fileIn=fileIn[0]

	fileOut,paramRanges,paramResolutions=contourDefaults(fileIn,fileOut,paramRanges,paramResolutions,solveFunc)
	if not overwrite and os.path.exists(fileOut):
		warn("generateHeatmap","overwrite=False and fileOut exists, skipping generations")
		return fileOut

	conditionalPrint("genContour2D",fileOut+","+str(paramRanges)+","+str(paramResolutions))

	#BOUNDS AND RANGES IN X, Y (and Z)
	xlb=paramRanges[0][0];xub=paramRanges[0][1];xr=paramResolutions[0]
	ylb=paramRanges[1][0];yub=paramRanges[1][1];yr=paramResolutions[1]

	#xlb=0 ; xub=15e-9 ; ylb=0 ; yub=500

	global solveResidualOnly

	def solveResidualOnly(parameterValues,fileIn):
		setParams(tofit[:2],parameterValues)
		r,e=solveFunc["func"](fileIn,plotting="none",**solveFunc["kwargs"])
		return e[0]

	setVar("tofit",tf[2:]) # at each x,y location, we'll fit for z only

	#print(paramRanges[:2],paramResolutions[0],getVar("tofit"),fileOut)

	#ITERATE ACROSS ALL COMBINATIONS OF VALUES: USE SCIPY.BRUTE IF 3D OR IF nX==nY, OTHERWISE, FOR-LOOPS
	x0, fval, grid, jout = brute(solveResidualOnly, paramRanges[:2], args=[fileIn], Ns=paramResolutions[0], full_output=True, workers=-1, finish=None)
	residuals=jout

	#CLEANUP,
	setVar("tofit",tf)
	
	#SAVING IT OFF. HEADER(s) CONTAIN PARAM RANGES AND RESOLUTIONS
	os.makedirs("/".join(fileOut.split('/')[:-1]),exist_ok=True)
	header=[]
	header=tofit[0]+"="+str(xlb)+":"+str(xr)+":"+str(xub)+" , "+tofit[1]+"="+str(ylb)+":"+str(yr)+":"+str(yub)
	np.savetxt(fileOut, residuals, delimiter=",",header=header)
	return fileOut

# fka displayHeatmap, which handled 2D or 3D. we have since split this into displayContour2D and displayContour3D. similarly, generateHeatmap has been split into genContour2D and genContour3D. We also incorporated "Wafer Bonding/isSSTRneeded.py > flat3DContour" into genContour2D, where we can basically "flatten" an N-dimensional parameter space into 2D. instead of "for each combination of 2 fitted parameters, check the residual" it is "for each combination of 2 (or 2 or more) fitted parameters, run fitting (on the remaining parameters) and check the residual". This is similar in spirit to measureContour1Axis, which is "for a range of values for 1 fitted parameter, run fitting on the remaining". This is all tested via testing61.py
def displayContour2D(csvFile='', plotting="show", residuals='', ranges='', labels='', threshold=0.025, title='', bonusCurveFiles='', useLast=False, fileSuffix=".png",bonusXY='',interpolate=True):
	# POSSIBLE TO PASS LIST OF csvFiles TO GENERATE OVERLAID PLOTS
	if type(csvFile)==list:
		conditionalPrint("displayHeatmap","(calling displayHeatmap for each file)")
		# FIRST, cycle through all, useLast=False, so we generate independent plots
		for i,f in enumerate(csvFile):
			displayContour2D(f, plotting, residuals, ranges, labels, threshold, title, bonusCurveFiles,useLast=False,fileSuffix=fileSuffix)
		# THEN, cycle through all, useLast=True, fileSuffix="combined", so we generate a second set of plots, the last of which will have all overlapped?
		for i,f in enumerate(csvFile):
			displayContour2D(f, plotting, residuals, ranges, labels, threshold, title, bonusCurveFiles,useLast=(i!=0),fileSuffix="_combined"+fileSuffix)
		return

	
	conditionalPrint("displayHeatmap",str(csvFile)+","+str(plotting)+","+str(threshold)+","+str(useLast))

	# SET THRESHOLDS FOR EACH CONTOUR LINE ON CURVE
	levels=[.001,.005,.01,.025,.05,.1,.2,.3,.4,.5,.6] ; lines=["dashed"]*len(levels) ; alphas=[.75]*len(levels)
	if threshold not in levels:
		i=np.where(np.asarray(levels)>threshold)[0][0]
		levels.insert(i,threshold)
		lines.insert(i,"solid")
		alphas.insert(i,1)
	i=levels.index(threshold) ; lines[i]="solid" ; alphas[i]=1	
	ls=[ {"solid":"-","dashed":"--"}[l] for l in lines ]
	conditionalPrint("displayHeatmap",str(levels)+","+str(lines)+","+str(alphas))
	figDir=csvFile.split('/')[:-1] ; figDir="/".join(figDir)
	filename={"show":"","save":figDir+"/"+csvFile.split('/')[-1].replace(".txt",fileSuffix)}[plotting]
	conditionalPrint("displayHeatmap","saving to: '"+filename+"'")
	
	# READ IN DATA
	if len(residuals)>1: # RESIDUALS PASSED: (along with ranges, labels)
		xlabel,ylabel=labels ; xlb,xub=ranges[0] ; ylb,yub=ranges[1]
		nx,ny=np.shape(residuals)
		x=np.linspace(xlb,xub,nx) ; y=np.linspace(ylb,yub,ny)
	else:
		#residuals,[x,y,z],[xlabel,ylabel,zlabel]=importHeatmapFile(csvFile)
		[x,y],[xlabel,ylabel]=contourFileHeader(csvFile)
		residuals = np.genfromtxt(csvFile, delimiter=',')
		xlb,xub=min(x),max(x) ; ylb,yub=min(y),max(y)

	if interpolate: # PERFORMANCE CONSIDERATIONS: interp is slow, but it sure as heck beats generating an actual 200x200 contour! and it smooths out the contour plot mighty nicely. 300x300 is significantly slower for no gains. 100x100 still had ragged edges. 
		interp=scipy.interpolate.RegularGridInterpolator((x,y),residuals,method='cubic')
		xs=np.linspace(xlb,xub,200) ; ys=np.linspace(ylb,yub,200)
		xm,ym=np.meshgrid(xs,ys)
		resinterp=interp((xm,ym)).T # idk how, but somehow through interpolation, our indices get switched. 
		x,y,residuals=xs,ys,resinterp


	kwargs={"filename":filename, "heatOrContour":"contour", "xlabel":xlabel, "ylabel":ylabel, "title":title, "linecolor":"inferno", "linestyle":ls, "inline":True, "levels":levels, "useLast":useLast}

	if len(bonusXY)>0:
		kwargs["overplot"]=[{"xs":bonusXY[0],"ys":bonusXY[1],"kind":"scatter"}]
		if len(bonusXY)>2:
			kwargs["overplot"][0]["c"]=bonusXY[2]

	#Zvals=np.transpose(residuals) ; Xvals=x ; Yvals=y
	#print(np.shape(Zvals),np.shape(Xvals),np.shape(Yvals))
	lcontour(residuals.T, x, y, **kwargs) # updated lcontour > nicecontour expects z[y,x], but the genContour2D files read in as z[x,y], so we need to transpose it here

# fka displayHeatmap, which handled 2D or 3D. we have since split this into displayContour2D and displayContour3D. similarly, generateHeatmap has been split into genContour2D and genContour3D. We also incorporated "Wafer Bonding/isSSTRneeded.py > flat3DContour" into genContour2D, where we can basically "flatten" an N-dimensional parameter space into 2D. instead of "for each combination of 2 fitted parameters, check the residual" it is "for each combination of 2 (or 2 or more) fitted parameters, run fitting (on the remaining parameters) and check the residual". This is similar in spirit to measureContour1Axis, which is "for a range of values for 1 fitted parameter, run fitting on the remaining". This is all tested via testing61.py
def displayContour3D(csvFile='', plotting="show", residuals='', ranges='', labels='', threshold=0.025, title='', elevAzi=[36,-60],projected=True,bonusCurveFiles='',useLast=False,fileSuffix=".png",booleanCombine="intersection",levels=[.001,.005,.01,.025,.05,.1,.2,.3,.4,.5,.6],cmap="inferno",alpha=.25): # lifehack: 3D contours look like shit. use custom levels to select which 3D surfaces you want plotted, use elevAzi and custom filename ("plotting=filename") to make gifs of your 3D contours rotating and stuff! if your papers look bad, at least your presentations can look fire. 
	conditionalPrint("displayHeatmap","received file(s) : "+str(csvFile))

	if type(csvFile)==list:
		stack=[]
		for f in csvFile:
			residuals,[x,y,z],[xlabel,ylabel,zlabel]=importHeatmapFile(f)
			stack.append(residuals)
		if booleanCombine=="intersection":
			residuals=np.max(stack,axis=0)
		else:
			residuals=np.min(stack,axis=0)

	else:
		residuals,[x,y,z],[xlabel,ylabel,zlabel]=importHeatmapFile(csvFile)	

	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib.path import Path
	import matplotlib.patches as patches

	
	fig = plt.figure()
	#ax = fig.gca(projection='3d')
	ax = fig.add_subplot(projection = '3d')

	levels=np.asarray(levels)

	#PLOT CONTOUR PLANES
	zfact=(max(z)-min(z))/100 #normal strategy is to "flatten" levels by divide by, say, 100, if residuals are between .1 and 5%, we're faking the countours as a 3D surface but only with a height that changes by .001-.05. BUT, if our z scale is already tiny (eg, nanometers or something), this'll be a problem, so we need to further reduce it.
	for i in range(0,len(z)):
		if np.amin(residuals[:,:,i])>=max(levels):
			continue
		Z=np.transpose(residuals[:,:,i]*zfact+z[i])
		lv=levels*zfact+z[i]
		CS=plt.contour(x, y, Z, levels=lv,alpha=alpha,cmap=cmap) # a series of 2D contour slices
		i=np.where(levels==threshold)[0][0]
		if i<len(CS.collections):
			CS.collections[i].set_color('red')
			CS.collections[i].set_alpha(min(6*alpha,1))

	ax.view_init(elev=elevAzi[0], azim=elevAzi[1])

	if projected:
		levels=np.asarray([threshold]) 		# only draw 2.5% contour on the axis planes...
		lxyz=[x,y,z] ; cxyz="xyz"
		for transAx in [2,0,1]:									# transpo,meshgrid,plt.cont,zdir 
			projected=np.transpose(np.amin(residuals,axis=transAx))				# 2	x,y	X,Y,projZ  "z"			
			xy=[lxyz[i] for i in range(3) if i!=transAx ]					# 0	y,z	projX,Y,Z  "x"
			XY=np.meshgrid(*xy)								# 1	x,z	X,projY,Z  "y"
			XY.insert(transAx,projected)							#
			#XYZ=[ [XY[0],Y,projected] , [projected,X,Y]
			#print(np.shape(XY))
			#print(transAx,cxyz[transAx],XY)
			plt.contour(*XY, levels=levels, colors=["red"], zdir=cxyz[transAx],alpha=0.25)
			zAxis=[x,y,z][transAx] ; zoff=max(zAxis) ; XY[transAx]+=zoff
			plt.contour(*XY, levels=levels+zoff, colors=["red"], zdir=cxyz[transAx],alpha=0.25)
	
	ax.set_xlabel(xlabel,labelpad=8)
	ax.set_ylabel(ylabel,labelpad=8)
	ax.set_zlabel(zlabel,labelpad=8)


	if plotting=="show":
		plt.show()
	elif len(fignames)>0:
		for f in fignames:
			plt.savefig(f.replace(".csv",fileSuffix))
	else:
		if plotting=="save":
			fout=csvFile.replace(".txt",fileSuffix)
		else:
			fout=plotting
		plt.savefig(fout)
	return fig,ax

def contourFileHeader(csvFile): # e.g "# Kz2=13.366038035902228:20:40.098114107706685 , R1=2.4173314242876275e-09:20:7.2519942728628825e-09"
	header=open(csvFile).readlines()[0].replace("#","").replace("\n","").replace(" ","")
	header=re.split('=|:|,',header)
	labels=header[::4]
	v0s=[ float(v) for v in header[1::4] ] ; ns=[ int(v) for v in header[2::4] ] ; vfs=[ float(v) for v in header[3::4] ]
	values=[ np.linspace(v0,vf,n) for v0,vf,n in zip(v0s,vfs,ns) ]
	#print(unitsDict.keys(),factDict.keys())
	values=[ v*getScaleUnits(l)[0] for l,v in zip(labels,values) ]
	labels=[ l+" ("+getScaleUnits(l)[1]+")" for l in labels ]
	return values,labels

def importHeatmapFile(csvFile):
	nParams=len(tofit)
	header=open({2:csvFile,3:csvFile+"_0"}[nParams],'r').readlines()[0] #eg: "# Kz2=1:20:30 , d2=5e-08:20:5e-07"
	header=re.split('=|:|,',header)
	#print(header)
	#set up titles and scaling factors, handle bounds
	xlabel=tofit[0] #; xunit="-" ; xfact=1
	#if xlabel[:-1] in unitsDict.keys(): #conceivably, (eg, KZFheatmap), the values in the header aren't fittable thermal properties. so default to unitless and no factor
	xfact,xunit=getScaleUnits(xlabel)
	xlb=float(header[1]) ; nx=int(header[2]) ; xub=float(header[3]) ; xlb=xlb*xfact ; xub=xub*xfact ; xlabel=xlabel+" ("+xunit+")"

	ylabel=tofit[1] #; yunit="-" ; yfact=1
	#if ylabel[:-1] in unitsDict.keys():
	yfact,yunit=getScaleUnits(ylabel)
	ylb=float(header[5]) ; ny=int(header[6]) ; yub=float(header[7]) ; ylb=ylb*yfact ; yub=yub*yfact ; ylabel=ylabel+" ("+yunit+")"
	#ylabel=tofit[1] ; yunit=unitsDict[ylabel[:-1]] ; yfact=factDict[ylabel[:-1]] ; ylabel=ylabel+" ("+yunit+")"
	#ylb=float(header[5]) ; ny=int(header[6]) ; yub=float(header[7]) ; ylb=ylb*yfact ; yub=yub*yfact
	
	zlb=0 ; zub=0 ; nz=0 ; zlabel=0
	if nParams==3:
		zlabel=tofit[2] #; zunit="-" ; zfact=1
		#if zlabel[:-1] in unitsDict.keys():
		zfact,zunit=getScaleUnits(zlabel)
		zlb=float(header[9]) ; nz=int(header[10]) ; zub=float(header[11]) ; zlb=zlb*zfact ; zub=zub*zfact ; zlabel=zlabel+" ("+zunit+")"
		#zlabel=tofit[2] ; zunit=unitsDict[zlabel[:-1]] ; zfact=factDict[zlabel[:-1]] ; zlabel=zlabel+" ("+zunit+")"
		#zlb=float(header[9]) ; nz=int(header[10]) ; zub=float(header[11]) ; zlb=zlb*zfact ; zub=zub*zfact
	
	#import R²s
	if nParams==2:
		residuals = np.genfromtxt(csvFile, delimiter=',')
	else:
		residuals = np.zeros((ny,nx,nz))
		for n in range(nz):
			residuals[:,:,n]=np.genfromtxt(csvFile+"_"+str(n), delimiter=',')[:,:]

	#set up lists of values
	xs=np.linspace(xlb,xub,nx)
	ys=np.linspace(ylb,yub,ny)
	zs=np.linspace(zlb,zub,nz)

	return residuals,[xs,ys,zs],[xlabel,ylabel,zlabel]

# Tz() returns index ordering: T[d,r],depths[d],radii[r]
def showTrzs(Ts,depths,radii,realAspect=False,plotting="show",plotTitle="",includeTPD=False,savefile="",bonusContours="",cbounds="auto"): #given a 2D matrix of temperatures at radii and depths (output of Tz()), we'll plot it for you
	depths=depths*1e6 ; radii=radii*1e6 #convert to μm
	#calculate Temp at T=1/e, and depth (thermal penetration depth)
	tpdFact=np.exp(1)**2
	#tpdFact=2
	if includeTPD and type(includeTPD)!=bool:
		tpdFact=includeTPD
	T_tpd=np.amax(Ts)/tpdFact ; tpd=0

	if np.amin(Ts[:,0])<T_tpd: #if datapoints "down the center" (0th r value) does have values less than
		tpd=depths[np.where(Ts[:,0]<T_tpd)[0][0]]
	#tpd=100e-6
	if plotting=="none":
		return tpd/1e6
	levels=list( np.linspace(np.amin(Ts),np.amax(Ts),10) )
	if cbounds!="auto":
		levels=list( np.linspace(cbounds[0],cbounds[1],30) )
	labels=[ str(np.round(v,3)) for v in levels ]
	lines=["--"]*len(levels)
	#add thermal penetration depth marker contour line
	if includeTPD and np.amin(Ts)<T_tpd:
		i=np.where(np.array(levels)>T_tpd)[0][0]
		levels.insert(i,T_tpd) ; labels.insert(i," ") ; lines.insert(i,"-")
	for bc in bonusContours:
		if bc>=max(levels) or bc<=min(levels):
			continue
		i=np.where(np.array(levels)>bc)[0][0]
		levels.insert(i,bc) ; labels.insert(i,"") ; lines.insert(i,"-")
	filename={True:plotting.replace("save",""),False:""}["save" in plotting]
	if len(filename)>0:
		direc="/".join(filename.split('/')[:-1])
		if len(direc)>0:
			os.makedirs(direc,exist_ok=True) #create directory if it doesn't already exist
	#print("showTrzs: saving to:",filename)
	#from plotter import plotHeatmapContour
	#from nicecontour import contour
	if len(savefile)>3:
		header="# radii="+str(list(radii))+";depths="+str(list(depths))
		np.savetxt(savefile,Ts,header=header,delimiter=",")

	lcontour(Ts,radii,-1*depths,filename=filename,heatOrContour="both",xlabel="radius (um)",ylabel="depth (um)",title="T(r,z)",zlabel="",linestyle=lines) # Tz() returns T[d,r],depths[d],radii[r], but contours expects Z[y,x],xs[x],ys[y]
	#plotHeatmapContour(Ts, radii, -1*depths, "radius (um)", "depth (um)", "T(r,z)", colorBounds=cbounds, levels=levels, styles=lines,filename=filename)

	return 0 

def Ttzr(mindepth=0,maxdepth=1500e-9,dsteps=50,rsteps=50,maxradius=0,tsteps=50): # we'll just call Tz(), but with full=True, for the appropriate frequencies (fm, plus low for steady state heating. OR, fourier series if non-sine waveform). then we'll pass back the stack T(t,z,r). this code used to be in gui.py, but i moved it here because it's pretty generally applicable
	p=1/fm
	conditionalPrint("Ttzr",str(time.time())+" generating waveform")
	waveform=waveformPWA # copy the variable just so we can swap in "sine" for unsupported waveforms
	#if waveform not in ["sine","square","triangle"]:
	#	print("WARNING: waveform "+waveform+" not yet supported by T(t,z,r)")
	#	waveform="sine"
	if waveform=="sine":
		fft=np.asarray([1,1]) ; freq=np.asarray([fm,.01])
	else:
		# note, for sine, you could use f(x)=4/π Σ 1/n sin(nπx/L) https://mathworld.wolfram.com/FourierSeriesSquareWave.html, ie, freq=ns*fm, fft=4/np.pi*1/ns, to get frequencies and fft factors, but this doesn't work. fft-ing the time series yields a different result. i think it's a sign difference in how we ifft it vs how we'd sum this up. (probably could still be used with some complex pre-factor)
		ts=np.linspace(0,p,1000,endpoint=False) ; dt=p/1000
		args=[ts,fm]
		if waveformPWA=="square-gauss":
			args.append(chopwidth)
		H=pumpWaveform(*args)
		H=H/np.mean(H)*dt/p
		fft=np.fft.fft(H) ; freq=np.fft.fftfreq(n=len(ts),d=dt) # fft() -> Aₙ, fftfreq -> ωₙ === Σₙ Aₙ*sin(ωₙ)
	omegas=freq*np.pi*2
	#print("Ttzr",omegas)
	# Now we have the frequencies, pass them into T(z,r) (instead of ΔT(ω) for PWAfunc). oh, and instead of doing each frequency one at a time, we edited Tz() to accept a list of omegas (and return a stack: one T(z,r) for each frequency)
	conditionalPrint("Ttzr",str(time.time())+" generating TZs")
	#TRZs=[]
	#for f,a in zip(freq,fft):
	#	setParam("fm",f)
	#	T,d,r,Ts=Tz(rsteps=rsteps,dsteps=dsteps,maxdepth=maxdepth,maxradius=maxradius,full=True)
	#	TRZs.append(Ts*a)
	T,d,r,TZRs=Tz(rsteps=rsteps,dsteps=dsteps,mindepth=mindepth,maxdepth=maxdepth,maxradius=maxradius,full=True,omegas=omegas)
	TZRs*=fft[:,None,None]

	# time to unfourier: since fourier series is sum of sines and cosines, temperature at a given point in time is the sum of each Aₙ*cos(ωₙ*t)-Bₙ*sin(ωₙ*t)
	ts=np.linspace(0,p,tsteps,endpoint=False) #; print(ts,tsteps)
	def zt(ts): # inside PWAfunc, we call delTomega(ω) which just returns a 1D list ΔT(ω). here, we have a 1D list of [ω], but TRZs is ΔT(ω,z,r) is a 3D matrix. 
		# METHOD 1: a for loop for summing over ω. goes easy on ram, but it's slow!
		s=list(np.shape(TZRs)) ; s[0]=len(ts) # ΔT(ω,z,r) --> ΔT(t,z,r)
		T=np.zeros(s)
		for Tzr,o in zip(TZRs,omegas):	
			T+=(Tzr[None,:,:]*np.exp(1j*o*ts[:,None,None])).real
		return T,ts,d,r
		# METHOD 2: vectorized, NOT ACTUALLY FASTER?
		#T=TZRs[None,:,:,:]*np.exp(1j*omegas[None,:,None,None]*ts[:,None,None,None]) # t,ω,z,r
		#T=np.sum(T.real,axis=1)
		#return T,ts,d,r

	# step 5: generate our time-dependant signal, and shift for zero-crossing at t=0, then re-generate
	conditionalPrint("Ttzr",str(time.time())+" stacking TZs")
	z=zt(ts)
	conditionalPrint("Ttzr",str(time.time())+" stack finished")
	return z

def melt(T,Tmelt,Hmelt,Cmat): # given a temperature map (T(r,z), and melting params (melt temp (K), latent heat (J/m3), and heat capacity (J/m3/K), simply apply a temperature offset based on effective temperature H/C, ie "how much would the temperature have risen if melting had not occurred"
	dTmelt=Hmelt/Cmat # J/m³ / J/m³/K = K
	maskA=np.zeros(np.shape(T)) ; maskB=np.zeros(np.shape(T))
	maskA[T>=Tmelt+dTmelt]=1 ; maskB[T>=Tmelt]=1 ; maskB[T>=Tmelt+dTmelt]=0
	T[maskA==1]-=dTmelt ; T[maskB==1]=Tmelt
	return T,dTmelt

def integrateRadial(I,r): # works if I is 1D (list of radii), or if I is 2D (n timesteps x n radii)
	#print("integrateRadial",np.shape(I),np.shape(r),np.shape(np.trapz(I*r,x=r)*np.pi*2))
	return np.trapz(I*r,x=r)*np.pi*2
	# Try it yourself:
	xs=np.linspace(-max(r),max(r),1000) ; ys=np.linspace(-max(r),max(r),1000)
	rads=np.sqrt(xs[:,None]**2+ys[None,:]**2)
	f=interp1d(r,I)
	I_cartesian=np.zeros(np.shape(rads))
	I_cartesian[rads<max(r)]=f(rads[rads<max(r)])
	conditionalPrint("integrateRadial","cartesian: "+str(np.sum(I_cartesian)*(xs[1]-xs[0])**2))
	conditionalPrint("integrateRadial","radial: "+str(np.trapz(I*r,x=r)*np.pi*2))
	return np.trapz(I*r,x=r)*np.pi*2

# TODO: "if asymmetricUncertainty: add negatives (see testing74.py)"
def allParams(): # This is used by perturbUncertainty() and monteCarlo() : if no perturbParams were passed, we'll supply them TODO i hate parallel lists. we should use a dict instead. (also means updatePTPPB() and perturbUncertainty() would both need to be updated)
	#IF NO PERTURB PARAMS WERE PASSED, DEFAULT TO ALL (CONSTRUCT BASED ON SIZE OF TP), DEFAULT TO PERTURBING BY THE SAME AMOUNT
	paramsToPerturb=["rpu","rpr"]
	for l in range(int((len(tp)+1)/2)): #for each layer present
		paramsToPerturb=paramsToPerturb+['C'+str(l+1),'Kz'+str(l+1),'d'+str(l+1)]
		if "Kz" not in tp[l*2]: #TODO: really ought to detect if ANY layers have Kr=Kz
			paramsToPerturb=paramsToPerturb+['Kr'+str(l+1)]
	for i in range(int((len(tp)-1)/2)):
		if useTBR:
			paramsToPerturb.append('R'+str(i+1))
		else:
			paramsToPerturb.append('G'+str(i+1))
	perturbBy=[ {"K":10,"C":2,"r":10,"d":5,"G":15,"R":15,"g":5}[P[0]] for P in paramsToPerturb ]
	
	return paramsToPerturb,perturbBy

# allows perturbUncertainty() kwargs paramsToPerturb and perturbBy to be:
# 1. empty (we default in "all", with sensible perturb amounts, using allParams())
# 2. list of parameters and a single value for perturb amount (perturb all by the same amount)
# 3. parallel lists: ["d1","Kz1"] and [5,20] perturbs d1 by 5%, Kz1 by 20%
# 4. partial parallel lists: if paramsToPerturb is longer, we append to perturbBy with the defaults (from allParams)
def updatePTPPB(paramsToPerturb,perturbBy):
	paramsToPerturb=list(paramsToPerturb) ; perturbBy=list(perturbBy)
	ptp,pb=allParams()
	if len(paramsToPerturb)==0:
		paramsToPerturb=ptp
	if type(perturbBy) in [int,float]:
		perturbBy=[perturbBy]*len(paramsToPerturb)
	else:
		for i in range(len(perturbBy),len(paramsToPerturb)):
			perturbBy.append(pb[ptp.index(paramsToPerturb[i])])
	return paramsToPerturb,perturbBy

#perturb each assumed parameter, and check how each affects the result for the solved-for parameters. change d1 by 5%, observe Kz2 changes by 3%. change C2 by 5%, observe Kz2 changes by 5%, then combine, error=√(Σ[(x-xₚ)²]). "plotting" options include: none, show, save, showall
# TODO: "if asymmetricUncertainty: sort into positives and negatives (see testing74.py)"
def perturbUncertainty(fileToRead,paramsToPerturb='',perturbBy='',plotting="none",reprocess=True,solveFunc=None): 
	#conditionalPrint("perturbUncertainty",str(fileToRead)+","+str(paramsToPerturb)
	if solveFunc is None: # measureContour1Axis and perturbUncertainty are both the rare exception to *just* using mode: eg, we still want to be able to pass in solveKZF shit
		solveFunc={"func":solve,"kwargs":{"plotting":plotting}} # solveFunc should hold the func and a dict of kwargs

	# filename for saving val,residual,remainingparamsfittedvals
	if isinstance(fileToRead,list):
		fout=figFile(lcs(filesToRead),"save",subfolder="perturbUncert").replace(".png",".uncert")
	else:
		fout=figFile(fileToRead,"save",subfolder="perturbUncert").replace(".png",".uncert")

	if not reprocess and os.path.exists(fout):
		print("READING IN PERTURB UNCERTAINTY RESULTS FROM FILE:",fout)
		lines=open(fout,'r').readlines() # lines read: "tofit = Kz2,R1" and so on. so first split by "=", then split by ","
		tf			=	lines[0].split("=")[-1].strip().split(",")
		resultUnperturbed	=	lines[1].split("=")[-1].strip().split(",")
		uncertainty		=	lines[2].split("=")[-1].strip().split(",")
		paramsToPerturb		=	lines[3].split("=")[-1].strip().split(",")
		perturbBy		=	lines[4].split("=")[-1].strip().split(",")
		resultUnperturbed = [ float(v) for v in resultUnperturbed ] # some need to be cast to floats
		uncertainty 	  = [ float(v) for v in uncertainty	  ]
		perturbBy 	  = [ float(v) for v in perturbBy	  ]
		delResults=np.loadtxt(fout,skiprows=5)
		if tf == tofit: 
			return resultUnperturbed,uncertainty,list(zip(paramsToPerturb,perturbBy,delResults))

	#COPY OFF PRE-EXISTING SETTINGS
	global tp,autorpu,autorpr,autofm
	tp_original,autorpu_original,autorpr_original,autofm_original=copy.deepcopy(tp),autorpu,autorpr,autofm #save settings
	
	# no paramsToPerturb passed, OR, perturb names and values don't match, then we'll do defaulting:
	paramsToPerturb,perturbBy=updatePTPPB(paramsToPerturb,perturbBy)

	#SOLVE AT INITIAL PARAMETERS
	conditionalPrint("perturbUncertainty","running for file: "+str(fileToRead)+", solving first at initial parameters:",pp=True)
	resultUnperturbed=[];RESo=100.
	resultUnperturbed,[RESo,sigo]=solveFunc["func"](fileToRead,**solveFunc["kwargs"]) #tuple of results (K, G...), corresponding to tofit. plotting: pass through none, show, or save
	conditionalPrint("perturbUncertainty","--> "+str(resultUnperturbed))
	tp_solved=copy.deepcopy(tp) #harvest solved-for TP, to use as our guesses for perturbed cases

	#FOR EACH PARAMETER, PERTURB IT, AND RESOLVE
	plotting={"none":"none", "show":"none", "save":"none", "showall":"show"}[plotting] #translate plotting for perturbations (only "showall" shows perturbations)
	delResults=[] #we'll hold dK et al here: K original - K perturbed (for each parameter we're perturbing)
	for P,dP in zip(paramsToPerturb,perturbBy): #for each parameter and perturbation pair, re-solve. store off dK, dG, dEtc
		if P in tofit: #ask us to perturb a parameter we're fitting for? skip it
			conditionalPrint("perturbUncertainty","(skipping: "+P+" is being fitted for)")
			delResults.append([0]*len(tofit))
			continue

		 #solve>readTDTR can read pump, probe radii, and modulation frequency. turn them off after we've read them one, so our perturbations of those values works. 
		if P=="rpu":
			autorpu=False
		if P=="rpr":
			autorpr=False
		if P=="fm":
			autofm=False
		conditionalPrint("perturbUncertainty","perturbing "+P+"("+str(getParam(P))+","+str(type(getParam(P)))+") by "+str(dP)+"%")
		
		tp=copy.deepcopy(tp_solved)
		setParam(P,getParam(P)*(1.+dP/100.),warning=False)

		conditionalPrint("perturbUncertainty","",pp=True)

		resultPerturbed=[];RESp=100.
		resultPerturbed,[RESp,sigp]=solveFunc["func"](fileToRead,**solveFunc["kwargs"])
		dRes=[]
		#print(RESp,RESo)
		if RESp<RESo: # TODO perturb up and find a lower residual? what about perturb down? we should check for that and warn for it too?
			warn("perturbUncertainty","warning: perturbed "+P+" by "+str(dP)+"% yielded lower R^2: "+str(RESp)+" vs "+str(RESo)+". This may be an indication that this parameter is set incorrectly in your thermal properties matrix")
		for i in range(0,len(resultUnperturbed)):
			dRes.append(resultPerturbed[i]-resultUnperturbed[i])
			conditionalPrint("perturbUncertainty",tofit[i]+"p="+str(resultPerturbed[i])+" vs "+tofit[i]+"o="+str(resultUnperturbed[i])+", d"+tofit[i]+"/d"+P+"="+str(dRes[-1]))
		delResults.append(dRes)
	
		 #solve>readTDTR can read pump, probe radii, and modulation frequency. turn them off after we've read them one, so our perturbations of those values works. 
		if P=="rpu":
			autorpu=autorpu_original
		if P=="rpr":
			autorpr=autorpr_original
		if P=="fm":
			autofm=autofm_original


	#solving complete, restore old settings
	tp=copy.deepcopy(tp_original) #restore
	# REFIT USING STOCK PARAMETERS. (if you don't, solve > writeResultFile() will mean we're left with a perturbed result-file, which will mess up any subsequent solve(refit=False) runs. so *just in case*, we should refit
	resultUnperturbed,[RESo,sigo]=solveFunc["func"](fileToRead,**solveFunc["kwargs"])
	#compute sqrt(dKdA^2+dKdB^2+dKdC^2+...dKdN^2), sqrt(dGdA^2+dGdB^2+dGdC^2+...dGdN^2), sqrt(dEtcdE^2+dEtcdB^2+dEtcdC^2+...dKdN^2). "for each column in Dres ([[dKdA,dGdA,...],[dKdB,dGdB,...]...]), grab each row, square, sum, root.". nice of numpy to do this for us (squaring is done element-by-element. summing is elementwise as well (each element added to the next. here, "each element" will be each row. collapsing all rows into one. pow.5 is elementwise again as well, leaving the resulting list of uncertainties. noice.
	print(delResults)
	delResults=np.asarray(delResults)
	uncertainty=sum(delResults**2.)**.5

	header =	"tofit = "+",".join( tofit )+"\n"
	header = header+"resultUnperturbed = "+",".join( [ str(v) for v in resultUnperturbed ] )+"\n"
	header = header+"uncertainty = "+",".join( [ str(v) for v in uncertainty ] )+"\n"
	header = header+"paramsToPerturb = "+",".join( paramsToPerturb )+"\n"
	header = header+"perturbBy = "+",".join( [ str(v) for v in perturbBy ] )
	np.savetxt(fout,delResults,header=header)

	return resultUnperturbed,uncertainty,list(zip(paramsToPerturb,perturbBy,delResults))

def monteCarlo(fileToRead,paramsToPerturb='',perturbBy='',N=10,returnFull=False):	# each uncertain assumed param IRL has a gaussian possibility distribution, so
	print("WHY ARE YOU USING MONTE CARLO? IT GIVES THE SAME RESULTS (USUALLY) AS PERTURBUNCERTAINTY(), AND IS MUCH (MUCH) SLOWER. IF YOU HAVE FOUND A SITUATION WHERE IT YIELDS SUBSTANTIALLY DIFFERENT RESULTS, PLEASE EMAIL ME. TWP4FG@VIRGINIA.EDU")

	# no paramsToPerturb passed, OR, perturb names and values don't match, then we'll do defaulting:
	paramsToPerturb,perturbBy=updatePTPPB(paramsToPerturb,perturbBy)

	# gaussian distribution for each uncertain assumed, use perturbed values (combinations) for re-solving
	perturbs=np.random.normal(0,perturbBy,(N,len(perturbBy))) # [ nth, which param ], each column is a a normal distribution between +/-pb%, each row is a set of params we should try

	vals=np.asarray( [ getParam(p) for p in paramsToPerturb ] ) # original values of each
	vals=vals[None,:]+vals[None,:]*perturbs/100 # combinations of perturbed values
	#print(paramsToPerturb,vals)
	# TODO WARNING: IF AUTORPU IS NOT SET TO FALSE, WE WON'T MONTE OVER RPU (ditto with rpr). THIS IS DIFFERENT BEHAVIOR FROM perturbUncertainty()
	# TODO also need to auto-populate paramsToPerturb if none passed	
	if isinstance(fileToRead,str):
		args=[(vs,paramsToPerturb,solve,fileToRead,getVar("tp"),i) for i,vs in enumerate(vals)]
	else: # TODO WATCH OUT! currently we don't pass any sort of ss2Types information (denotes what kind of data the files are). you MUST specify 
		args=[(vs,paramsToPerturb,ss2,fileToRead,getVar("tp"),i) for i,vs in enumerate(vals)] # types inside a magicMods file: "mode='TDTR'" and so on

	results=parallel(monteWorker,args)
	if returnFull:
		return results
	residuals=[r[1][0] for r in results]
	solved=[r[0] for r in results] # solve() returns result,error
	
	solved=np.asarray(solved) # [ which test, which param ]
	r=np.mean(solved,axis=0) ; e=np.std(solved,axis=0)
	return r,e # same format that perturbUncertainty() returns. 
	#return solved,residuals

def monteWorker(args):
	setVar("quietWarnings",True)
	vals,params,solveFunc,filename,tp_old,i=args # multiprocessing pool.map only allows 1 arg, so we must cram into a list
	for p,v in zip(params,vals):
		setParam(p,v)
	conditionalPrint("monteWorker","solving with params:",pp=True)
	r,e=solveFunc(filename,plotting="none") # solve it. anything but none plotting crashes the system
	tp=copy.deepcopy(tp_old) # restore tp after solve, so it's prepped for the next val (else, we may get trapped in local minima)
	return r,e # in theory we've standardized all solve functions to return "[param1Result,param2result,...],[residual,stdev]", even solveSimult, which passes the max of the N simultaneous TDTR scans' residuals. 

def parallel(func,args,nworkers=7):		# if you have an arbitrary function, which you want run over and over for every element in an 
	if os.name=='posix':			# arbitrary list of arguments, use this. if we're on a posix system (linux, macOS), we'll use
		from multiprocessing import Pool# a multiprocessing pool to go run them all. if on windows, we'll settle for a list
		p=Pool(processes=nworkers)	# comprehension, since multiprocessing can't do fork, just spawn (and it's kind of a mess), and
		with p as pool:			# list comprehensions are at least faster than for loops. this is used by monteCarlo (for a list
			results=pool.map(func,args)	# of perturbed assumed parameters, refit), and measureContour1Axis (for a list of vals for the
	else:					# parameter you're fitting, refit and check the residual). 
		results=[ func(a) for a in args]
	return results


# This will eventually serve as a drop-in replacement (but much more compact) version of measureContour above. generate a 2D or 3D contour, read the segments used for drawing the threshold contour, and take the extremes of it
def measureContours(fileIn,fileOut='',paramRanges='',paramResolutions='',overwrite=False,threshold=.025,plotting="save"):

	# STEP 1, OPTIONAL SOLVING TO FIND CENTER OF CONTOUR REGION
	if len(paramRanges)==0:
		if type(fileIn)!=list:
			r,e=solve(fileIn,plotting="save")
		else:
			r,e=ss2(fileIn,plotting="save")
		paramRanges=[ [.5*v,1.5*v] for v in r ]

	# STEP 2, GENERATE CONTOURS
	fileOut=genContour2D(fileIn,paramRanges=paramRanges)

	# STEP 3, INSTEAD OF TRYING TO INFER RESIDUAL BOUNDS FROM THE CONTOUR PLOT (reading/parsing matplotlib objects), USE SCIPY.INTERPOLATE
	[x,y],[xlabel,ylabel]=contourFileHeader(fileOut)
	residuals = np.genfromtxt(fileOut, delimiter=',')
	xlb,xub=min(x),max(x) ; ylb,yub=min(y),max(y)
	#print("interpolating")
	interp=scipy.interpolate.RegularGridInterpolator((x,y),residuals,method='cubic')
	xs=np.linspace(xlb,xub,300) ; ys=np.linspace(ylb,yub,300)
	xm,ym=np.meshgrid(xs,ys)
	resinterp=interp((xm,ym)).T # check displayContour2D: we want residuals as R[x,y]
	#lcontour(residuals,x,y)
	#lcontour(resinterp.T,xs,ys,xlabel=xlabel,ylabel=ylabel)
	#print("rastering")
	# STEP 4: RASTER ALONG (finer) MESH HORIZONTALLY AND VERTICALLY
	overplot=[{"xs":[],"ys":[],"kind":"scatter","color":"red","s":1},{"xs":[],"ys":[],"kind":"scatter","color":"blue","s":1}]
	for j,y in enumerate(ys):
		where=xs[resinterp[:,j]<=threshold]
		if len(where)>0:
			overplot[0]["xs"].append(min(where)) ; overplot[0]["ys"].append(y)
			overplot[0]["xs"].append(max(where)) ; overplot[0]["ys"].append(y)

	for i,x in enumerate(xs):
		where=ys[resinterp[i,:]<=threshold]
		if len(where)>0:
			overplot[1]["ys"].append(min(where)) ; overplot[1]["xs"].append(x)
			overplot[1]["ys"].append(max(where)) ; overplot[1]["xs"].append(x)
	#print("raster finished")
	figf=figFile(fileIn,"save",subfolder="measureContour")
	lcontour(resinterp.T,xs,ys,xlabel=xlabel,ylabel=ylabel,overplot=overplot,filename=figf)

	extremes=np.zeros((2,2))
	i,j=np.unravel_index(np.argmin(resinterp, axis=None), (300,300))	
	bestfit=[xs[i],ys[j]]
	if len(overplot[0]["xs"])>0:
		extremes[0,:]=min(overplot[0]["xs"]),max(overplot[0]["xs"])
		extremes[1,:]=min(overplot[1]["ys"]),max(overplot[1]["ys"])
	fx,ux=getScaleUnits(tofit[0])
	fy,uy=getScaleUnits(tofit[1])
	extremes[0]/=fx ; bestfit[0]/=fx
	extremes[1]/=fy ; bestfit[1]/=fy
	#print(fx,ux)

	conditionalPrint("measureContour","found bestfit,extremes: "+str(bestfit)+","+str(extremes))

	return bestfit,extremes

	"""
	# STEP 3, PLOT THE RESULTS (fileIn may be a list (overlapped contours), so pad fileOut as a list too)
	if type(fileOut)!=list:
		fileOut=[fileOut]
	for i,f in enumerate(fileOut):
		ul=(i!=0)
		displayContour2D(f, plotting="save", threshold=threshold,useLast=ul)

	# STEP 4, READ IN RESIDUALS MAP(S) (stack them, if we have multiple)
	for i,fo in enumerate(fileOut):
		residuals,[x,y,z],[xlabel,ylabel,zlabel]=importHeatmapFile(fo)
		if i==0:
			residualStack=np.zeros((len(fileIn),*np.shape(residuals)))
		residualStack[i]=residuals

	residuals=np.max(residualStack,axis=0)
	if len(np.shape(residuals))==2:
		residuals=residuals[:,:,None]

	# STEP 5, GENERATE CONTOUR CURVES, AND MEASURE THEIR SIZE
	extremes=[ [np.inf,-np.inf] for tf in tofit ]

	# https://stackoverflow.com/questions/30376897/calling-contour-without-plotting-it-python-pylab-inline oops, matplotlib._cntr no longer around
	#https://stackoverflow.com/questions/18304722/python-find-contour-lines-from-matplotlib-pyplot-contour
	for k in range(np.shape(residuals)[2]): # *NOT* "for layer in residuals"; residuals from read are [x,y,z]=val
		layer=residuals[:,:,k]
		if np.all(layer>threshold) or np.all(layer<threshold):
			continue
		#print(layer)
		C = plt.contour(x, y, np.transpose(layer), [threshold], colors='k') # contour object
		segs=C.allsegs[0] # list of segments, each segment is a list of x,y points
		for n in range(len(tofit)):
			if n==2:
				extremes[n] = [ min(extremes[n][0],z[k]) , max(extremes[n][1],z[k]) ]
			else:
				mn=min([ min(seg[:,n]) for seg in segs ]) ; mx=max([ max(seg[:,n]) for seg in segs ])
				extremes[n] = [ min(extremes[n][0],mn) , max(extremes[n][1],mx) ]
	
	# STEP 6, SAVE OVERLAPPED REGION ON OLD IMAGE
	#if plotting!="none":
	#	figDir=fileOut[-1].split('/')[:-1] ; figDir="/".join(figDir)
	#	filename={"show":"","save":figDir+"/"+fileOut[-1].split('/')[-1].replace(".txt",".png")}[plotting]
	#	plt.savefig(filename)
	filename=figFile(fileIn,"save",subfolder="measureContour")
	direc="/".join(filename.split("/")[:-1])
	os.makedirs(direc,exist_ok=True)
	plt.savefig(filename)

	for i,p in enumerate(tofit):
		fact=getScaleUnits(p)[0] # eg, G is plotted in MW/m²/K, so 1e-6 was used before
		extremes[i][0]/=fact
		extremes[i][1]/=fact
	print("measureContours",extremes)
	return extremes
	"""
			
# consider the case of a thin film, where at best, we can fit for G1,Kz2,G2, and then calculate a net resistivity. you can find the bounds on this by doing 3D contours (this) or 1D fast contours (next function) and then "for each test value, calculate resistivity"
def netResistivityFromContour(datafile,threshold): 
	fileOut=generateHeatmap(datafile,overwrite=False)				# first step, generate the ND contour grid
	#displayHeatmap(fileOut)
	residuals,[xs,ys,zs],[xlabel,ylabel,zlabel]=importHeatmapFile(fileOut)		# then read in the result
	minR=np.inf ; maxR=-np.inf							# we're going to track and highest/lowest resistivity we find
	where=np.where(residuals<=threshold)						# select all elements of the grid with "good fits"
	for ijk in zip(*where):								# if 2D, ijk contains x,y coordinates. if 3D, x,y,z coords
		R=0
		for n,param in enumerate(tofit):					# consider tofit=Kz2,G1.	n=0	n=1	
			vals=[xs,ys,zs][n] ; i=ijk[n]					# K or G comes from xs or ys or zs. (which one? the nth)
			val=vals[i]/getScaleUnits(param)[0]
			if param[0]=="G":						# R=...+1/G+...				+1/G
				R+=1/val
			elif param[0]=="K":
				l=param.replace("K","").replace("z","").replace("r","")	# which layer number?
				d=getParam("d"+l)					# get the thickness of that layer
				R+=d/val						# R=...+L/K+...			+L/K
		minR=min(minR,R) ; maxR=max(maxR,R)
	print(minR,maxR)

def netResistivityFrom1Axis(datafile,threshold,paramOfInterest=tofit[0]):
	valRange,fileOut=measureContour1Axis(datafile, paramOfInterest,threshold=threshold,overwrite=False,plotting="savefinal") # first step, generate the 1Axis contour
	lines=open(fileOut,'r').readlines()						# then read in the result
	minR=np.inf ; maxR=-np.inf							# we're going to track and highest/lowest resistivity we find
	print(lines[30].split(),tofit)
	for l in lines:									# simply traverse 1Axis list, looking for where fits were "good"
		R=0
		if len(l)<5 or l[0]=="#":						# eg # G2	residual	G1	Kz2
			if "residual" in l:
				colNames=l[1:].split()
			continue
		xryz=[ float(v) for v in l.split() ]					# eg 7956522.86	0.011906	32646212	169.23
		if xryz[1] > threshold:
			continue
		#xryz.pop(1)								# was x,r,y,z, remove 1nth element, now x,y,z
		for i,param in enumerate(colNames):
			if param[0]=="G":
				R+=1/xryz[i]						# R=...+1/G+...
			elif param[0]=="K":
				l=param.replace("K","").replace("z","").replace("r","")	# which layer number?
				d=getParam("d"+l)					# get the thickness of that layer
				R+=d/xryz[i]						# R=...+L/K+...
		minR=min(minR,R) ; maxR=max(maxR,R)
	print(minR,maxR)

# TODO: how about instead of passing solveFunc (into this, and "perturbUncertainty()". and needing the dict inside gui.py for the solve button), we just have a global for solveFunc, and set the global based on mode? seems much simpler

# TODO: if rpu is the paramOfInterest, we need to suppress auto (autorpu=False)
# sensitivity to fitted parameters is found by exploring the entire N-dimensional parameter space for the number of parameters being fitted. This is represented graphically in 2D for 2 parameters via a contour plot (generateHeatmap() / displayHeatmap) with the two parameters on the x and y axes. To generate these, we will have generated a grid and at each combination, we generate the TDTR curve and compare it to the data, where the resulting residual is the color or contour region on the plot. The size of a given contour (whatever threshold you like, for whatever residual you would accept as a "good fit") represents the uncertainty of the fitted result due to sensitivity of the fitted parameters. 
# We may measure the size of the contour by generating the contours (generate a N dimensional grid, try all combinations). This is done via measureContours(). OR, if there is one parameter we are interested in, we may generate a 1D "grid" of values for that parameter, and then use solve() to solve for the remaining parameters, and then we can check for a satisfactory residual value. This is done via measureContour1Axis(). 
# The former strategy allows the one-time generation of contours, meaning a changing threshold is "free", as is visualization. however, it is likely to be more expensive and less accurate. N-1 -D solve() is likely to require less "tries" of parameters to find the best value (in testing, a 2D solve() needed to compute the TDTR decay curve 29 times, whereas a 20x20x20 3D grid would be 8000 times. This means the latter strategy can achieve a 13x higher resolution. all while more or less guaranteeing that the best combination of the other two parameters will be found). One drawback with the latter solution however is that multi-cpus are not taken advantage of. least_squares does not use multiple cpus, whereas brute does (which is how we populate our grid). 
# (predictUncert) > measureContour1Axis > parallel > mc1aWorker > solve or ss2 > ss3h
# in versions >= 0.152, predictUncert takes settables dict instead of making dumbass assumptions based on ss2Glos. measureContour1Axis takes solveFunc dict too, which cascades ss2Types (mode, via solveFunc["kwargs"]=dict, passed into ss2's kwarg "listOfTypes") through. the new ss2 also must now manually infer fm,rpu,rpr from the files themselves! (TODO, this may not be good). ss2 then populates dict "settables" which is handled by ss3h
# why the re-write of this entire chain? it makes preductUncert much *much* cleaner when trying your "atypical" stuff like mfSSTR or SSTR+TDTR (predictUncert's "settables" ought to handle it all)
# TODO usage, for ratio/magnitude TDTR for example: 
# measureContour1Axis([f,f], "R1" , threshold=0.025, plotting="savefinal", solveFunc={"func":ss2,"kwargs":{"settables":{"fitting":["M","R"]}}} )
# TODO this shit's absolutely unhinged (a three-deep dict? ffs. is there a way to make this more sane? 
def measureContour1Axis(fs, paramOfInterest, ranges='', resolution=100, threshold=0.025, plotting="none", solveFunc=None, customParamControl='', nworkers=7, title='',extend=True,overwrite=False):
	#print("fs",fs)
	if solveFunc is None: # measureContour1Axis is the rare exception to *just* using mode: eg, we still want to be able to pass in solveKZF shit
		#solveFunc={"TDTR":solveTDTR,"SSTR":solveSSTR,"FDTR":solveFDTR,"PWA":solvePWA,"FD-TDTR":solveSimultaneous}[mode]
		solveFunc={"func":solve,"kwargs":{}} # solveFunc should hold the func and a dict of kwargs
	# filename for saving val,residual,remainingparamsfittedvals
	fout=fs
	if type(fs) == list:
		fout=lcs(fs)+"*.txt" # longest common substring will contain file path, i hope
		fout=fs[0]+"etal.txt" # TODO: lcs fails if your multifiles are in different folders. later on, we snatch filename, but end up with a leading slash
	fout=fout.replace("\\","/")
	pathpieces=fout.split("/")[:-1]+[callingScript,"contours"]
	fout="/".join(pathpieces) + "/" + fout.split("/")[-1] # path portion, contour files folder, filename portion
	fout=fout.replace(".txt","_"+paramOfInterest+".txt")
	#print("fs",fs)
	# if file already exists, import it (don't regen)
	if os.path.exists(fout) and not overwrite:
		warn("measureContour1Axis","output file found, no need to reprocess [\""+fout+"\"]") 
		vals,residuals,params,paramVals=read1AxisContour(fout)
		#print("vals,residuals,fout",vals,residuals,fout)
	else:
		global tofit,tp
		# step 1, if range of vals for param are not populated, solve (using initial tofit set of params), and use +/-50%
		ranges=list(ranges)
		if len(ranges)==0:
			p={True:"show",False:"none"}["init" in plotting]
			r,e=solveFunc["func"](fs,plotting=p,**solveFunc["kwargs"])
			solved=r[tofit.index(paramOfInterest)] # "G1,Kz2,G2" -> 100e6,25,200e6 -> index of Kz2 if paramOfInterst -> 25
			ranges=[solved*.5,solved*1.5]
			#ranges=[solved*.25,solved*7]
		# make a copy of fitting params, sans paramOfInterst
		tofit_old=tofit
		tofit=[ p for p in tofit if p != paramOfInterest ]
		tp_old=copy.deepcopy(tp)
		# for all plausible values for paramOfInterest, we'll set it, then solve for the rest in tofit, and record the residual
		vals=np.linspace(ranges[0],ranges[1],resolution) ; residuals=np.zeros(resolution)
		# use multiprocessing to solve N times (use all nodes on Rivanna)
		args=[(v,paramOfInterest,customParamControl,plotting,solveFunc,fs,tp_old,i) for i,v in enumerate(vals)]
		#print(args)
		results=parallel(mc1aWorker,args,nworkers)

		vals=list(vals)
		residuals=[r[1][0] for r in results] # mc1aWorker returns solve's r,e --> e. r contains residual and some other shit
		paramvals=[r[0] for r in results]
		#vals=np.asarray([r[2] for r in results])
		
		# basic check: we're using this for uncertainty, so what if the ENTIRE +/-50% range is still below threshold?
		dv=vals[1]-vals[0] ; i=0
		while max(residuals[:int(len(residuals)/2)])<threshold: # left half of residual vs property plot, if all are below threshold, trot left
			v=vals[0]-dv ; i-=1
			if v<0 or extend==False:
				break
			args=(v,paramOfInterest,customParamControl,plotting,solveFunc,fs,tp_old,i)
			r,e=mc1aWorker(args)
			vals.insert(0, v)
			residuals.insert(0, e[0])
			paramvals.insert(0,r)
		i=resolution
		while max(residuals[int(len(residuals)/2):])<threshold:# and i<200: # repeat for right half: if all are below threshold, trot right
			v=vals[-1]+dv ; i+=1
			if i>200 or extend==False:
				break
			args=(v,paramOfInterest,customParamControl,plotting,solveFunc,fs,tp_old,i)
			r,e=mc1aWorker(args)
			vals.append(v)
			residuals.append(e[0])
			paramvals.append(r)
		#print(paramvals)
		vals=np.asarray(vals)
		residuals=np.asarray(residuals)

		# save off the results
		header="# thermal properties matrix:\n"
		if "Cs" not in globals():
			header=header+"# [ unavailable ]\n"
		else:
			header=header+"#  Cs: "+str(Cs)+"\n#  Kzs: "+str(Kzs)+"\n#  ds: "+str(ds)+"\n#  Krs: "+str(Krs)+"\n#  Gs: "+str(Gs)+"\n"
			header=header+"# other parameters:\n#  fm: "+str(fm)+"\n#  fp: "+str(fp)+"\n#  rpump: "+str(rpump)+"\n#  rprobe: "+str(rprobe)+"\n#  tm: "+str(minimum_fitting_time)+"\n#  tn: "+str(time_normalize)+"\n#  nmax: "+str(nmax)+"\n"
		#print(fout)
		direc="/".join(fout.split('/')[:-1])
		#print(direc,fout)
		os.makedirs(direc,exist_ok=True) #create directory if it doesn't already exist
		with open(fout,'w') as f:
			f.write(header)
			f.write("# "+paramOfInterest+"\tresidual\t"+"\t".join(tofit)+"\n")
			for v,e,r in zip(vals,residuals,paramvals):
				s=str(v)+"\t"+str(e)+"\t"+"\t".join(list(map(str,r)))+"\n"
				f.write(s)
		# finally, restore all the other shit we fudged with
		tofit=tofit_old
		tp=copy.deepcopy(tp_old)

	# and plot the resulting series of vals and residuals
	if "final" in plotting:
		#u="-" ; f=1
		#if paramOfInterest[:-1] in unitsDict.keys():
		f,u=getScaleUnits(paramOfInterest)
		plotvals=vals*f # plotvals copied, since we don't want to scale our returnables
		gv=plotvals[residuals<=threshold] ; gr=residuals[residuals<=threshold]  
		figFile=""
		if "save" in plotting:
			figFile=fout[:-4]+".png"
		#print("SAVING CONTOUR")
		ranges="nan <= "+paramOfInterest+" <= nan"
		if len(gv)>0:
			mn=min(gv) ; mx=max(gv) ; r=mx-mn ; m=(mx+mn)/2 ; p=(r/2)/m*100 ; p=np.round(p,1)
			ranges=str(scientificNotation(mn,2))+" <= "+paramOfInterest+" <= "+str(scientificNotation(mx,2))+" ("+str(p)+"%)"
		title={True:title,False:ranges}[len(title)>0] # TODO this line may crash if no bounds exist (no value for fitting param was found within the bounds)
		#setVar("scaleY","log")
		#if len(altFnames)>0:
		#	figFile=altFnames+[figFile]
		lplot([plotvals,[min(plotvals),max(plotvals)],gv], [residuals,[threshold,threshold],gr], paramOfInterest+" ("+u+")", "lowest residual (-)", markers=['b-','k:','g-'], title=title, filename=figFile, yscale="linear",labels=[""]*3,ylim=[0,None],xlim=[min(plotvals),max(plotvals)])
		#setVar("scaleY","lin")
	# and return good vals
	goodVals=vals[residuals<=threshold]
	if len(goodVals)==0:
		return [0,0],fout
	valRange=[min(goodVals),max(goodVals)]
	#print("valrange,fout",valRange,fout)
	return valRange,fout


	# TODO: implement RMXY ("for fitting in rmxy"), save off residual v vals curve for easy re-thresholding, test with non-simultaneous

def mc1aWorker(args):
	setVar("quietWarnings",True)
	conditionalPrint("mc1aWorker","running with args:"+str(args))
	v,paramOfInterest,customParamControl,plotting,solveFunc,fs,tp_old,i=args # multiprocessing pool.map only allows 1 arg, so we must cram into a list
	#global tp ; tp=copy.deepcopy(tp_old)
	setParam(paramOfInterest,v) # set the parameter value
	if type(customParamControl)!=str: # allow the running of custom code here (used for KZF, with nonstandard params)
		customParamControl(v)
	p={True:"show",False:"none"}["iter" in plotting]
	#global altFname ; altFname="contPics/"+str(i)+".png"
	#r,e=solveFunc(fs,plotting="none") # solve it. anything but none plotting crashes the system
	#print("mc1aWorker","ss2",tp,solveFunc["kwargs"])
	try: # possible to get fitting errors like "max iterations encountered" and such, so just ignore 'em if so. 
		r,e=solveFunc["func"](fs,plotting="none",**solveFunc["kwargs"]) # solve it. anything but none plotting crashes the system
	except Exception as exc:
		r=np.zeros((len(tofit))) ; e=[1,1]
		print("mc1aWorker encountered an error:",exc)
	#print(i,v,r,e)
	#tp=copy.deepcopy(tp_old) # restore tp after solve, so it's prepped for the next val (else, we may get trapped in local minima)
	#print("mc1aworker","ss2",r,e)
	return r,e # in theory we've standardized all solve functions to return "[param1Result,param2result,...],[residual,stdev]", even solveSimult, which passes the max of the N simultaneous TDTR scans' residuals. 

def read1AxisContour(fout):
	rows = open(fout).readlines()
	vals,residuals,params,paramVals=[],[],[],[]
	# find first line containing data
	for i,r in enumerate(rows):
		if r[0]!="#":
			break
	params=rows[i-1].split()[3:] # columns header, eg "# Kz2	residual	G1	d2	G2" -> ["G1","d2","G2"]
	for row in rows[i:]:
		row=row.split()
		vals.append(float(row[0])) ; residuals.append(float(row[1])) ; paramVals.append( [ float(v) for v in row[2:] ] )
	vals=np.asarray(vals) ; residuals=np.asarray(residuals) ; paramVals=np.asarray(paramVals)
	return vals,residuals,params,paramVals

# This used to just be called by predictUncert (generate a fake data file, run contour uncertainty on it), but I found myself writing more and more testing scripts which used this (e.g. testing74.py) and thought it a good idea to break it off into its own function. really all we do is call TDTRfunc / SSTRfunc / PWAfunc / FDTRfunc, but this at least provides a standardized set of default x axis spacing values (time delays / power values / time points for square wave / frequencies)
synetheticMaxTDTR=5.5e-9 ; syntheticMaxFDTR=7
def makeSyntheticDataset(fout,addedNoise=0):
	# cycle through (potentially multiple) files to generate
	x_dict={"TDTR":np.logspace(np.log(300e-12)/np.log(10),np.log(synetheticMaxTDTR)/np.log(10),30) ,
		"SSTR":np.linspace(getVar("Pow")/100,getVar("Pow"),30) , 
		"FDTR":np.logspace(2,syntheticMaxFDTR,100) , 
		"PWA":np.linspace(0,1/fm,1024,endpoint=False) }
	xs=x_dict[mode]
	conditionalPrint("makeSyntheticDataset","saving to file: "+fout)
	ys=func(xs,store=fout,addNoise=addedNoise)
	return ys

# settables must be a dict with the types of files to generate and test contours against. e.g. {"mode":["TDTR"]} will simply run contours on a single generated TDTR file, using all the currently-set parameters (tp, fm, rpu, etc). {"mode":["TDTR","SSTR"],"fm":[8.4e6,1000]} will generate one TDTR and one SSTR file at the respective frequencies
# predictUncert > measureContour1Axis > parallel > mc1aWorker > solve or ss2 > ss3h
# in versions >= 0.152, predictUncert takes settables dict instead of making dumbass assumptions based on ss2Glos. measureContour1Axis takes solveFunc dict too, which cascades ss2Types (mode, via solveFunc["kwargs"]=dict, passed into ss2's kwarg "listOfTypes") through. the new ss2 also must now manually infer fm,rpu,rpr from the files themselves! (TODO, this may not be good). ss2 then populates dict "settables" which is handled by ss3h
# why the re-write of this entire chain? it makes preductUncert much *much* cleaner when trying your "atypical" stuff like mfSSTR or SSTR+TDTR (predictUncert's "settables" ought to handle it all)
def predictUncert(settables="",addedNoise=0.0,regen=True,threshold=.025,nworkers=3,subdir="tdtrcache",npts=100,ranges=""):
	settables=dict(settables) # DICT IN ARG DEFAULTS MEANS DICT CONTENTS ARE KEPT BETWEEN RUNS, AND YOU CAN'T RERUN FOR DIFFERENT MODES!!!
	# default to current mode if no settables passed
	global mode
	if "mode" not in settables.keys():
		settables["mode"]=[mode]
	
	fnames=[subdir+"/predictUncert_"+m+"_"+str(i)+".txt" for i,m in enumerate(settables["mode"])]
	for i,f in enumerate(fnames):
		for k in settables.keys():
			setVar(k,settables[k][i])
		conditionalPrint("predictUncert","generating "+mode+" data with params:",pp=True)
		ys=makeSyntheticDataset(f)

	# run contour uncertainty on file(s) generated
	if len(fnames)==1:
		solver={"func":solve,"kwargs":{}}
		fnames=fnames[0]
	else:
		kwargs={ k:settables[k] for k in settables.keys() if k!="mode" }
		#for k in settables.keys():
		kwargs={"listOfTypes":settables["mode"],"settables":kwargs}

		solver={"func":ss2,"kwargs":kwargs}

	#if not regen:
	#	solver["kwargs"]["refit"]=False
	p=tofit[0] ; v=getParam(p)
	#solver["func"](fnames,plotting="save",**solver["kwargs"],refit=regen) # WHY TF SHOULD IT FIT THE DATA WE JUST MADE? WE JUST MADE IT!
	if len(ranges)==0:
		ranges=[v*.25,v*1.75]

	valRange,fout=measureContour1Axis(fnames, p, plotting="savefinal", resolution=npts, ranges=ranges, extend=False, threshold=threshold, nworkers=nworkers, solveFunc=solver,overwrite=regen)	# mC1A > solve > various depending on mode
	return valRange

# loosely based on testing68.py. we revamped preductUncert to accept a dict of "settables" which we use to generate our fake data file(s), and then run solve or ss2 accordingly. ss2
def whichTechniqueShouldIUse(threshold): # TODO use caution when calling and re-calling this function. gui.py is allowed to, since all relevant globals will be reset on the next run, BUT, as it stands, some settables will faff up others
	settables={	"rTDTR":{"mode":["TDTR"],"fitting":["R"]},
			"mTDTR":{"mode":["TDTR"],"fitting":["M"]},
			"umTDTR":{"mode":["TDTR"],"fitting":["M"],"time_normalize":[""]},
			"SSTR":{"mode":["SSTR"]},
			"mfSSTR":{"mode":["SSTR"]*2,"fm":[1e3,1e7]},
			"mfTDTR":{"mode":["TDTR"]*4,"fm":[8.4e6,4.2e6,2.1e6,1.1e6],"fitting":["R"]*4},
			"mrTDTR":{"mode":["TDTR"]*2,"fitting":["R","M"]},
			"SSTDTR":{"mode":["TDTR","SSTR"],"fm":[8.4e6,1000],"rprobe":[5e-6,1.5e-6],"rpump":[10e-6,1.5e-6]} }
	ranges={} ; global tp ; tp_old=copy.deepcopy(tp)
	for k in settables.keys():			# For each measureent technique
		oldParams={ k2:getVar(k2) for k2 in settables[k].keys() } 	# copy off each parameter predictUncert will be updating
		tp=copy.deepcopy(tp_old) # must deep copy *both ways!* or else you're overwriting your backup when predictUncert updates tp
		vr=predictUncert(threshold=threshold,subdir="tdtrcache/"+k,settables=settables[k])
		for k2 in oldParams.keys():		# restore each parameter back
			setVar(k2,oldParams[k2])
		ranges[k]=vr
	#print(ranges)
	#tp=tp_old
	return ranges

"""
def predictUncert_v01(addedNoise=.003,regen=True,threshold=.025,nworkers=3,filesOnly=False,subdir="tdtrcache",allParams=False): # runs measureContour1Axis for tofit parameters
	global mode
	# Step 1, generate the fake data file(s) on which we'll run contour uncertainty
	# 1.a: for simultaneous solving, use ss2Types, else, use mode: put it in a list for easy iteration
	if len(ss2Types)>0:
		conditionalPrint("predictUncert","types: "+str(ss2Types)+" fms: "+str(ss2fms)+" rpus: "+str(ss2rpus)+" rprs: "+str(ss2rprs)+" magic: "+str(magicMods))
	modes={True:ss2Types,False:[mode]}[len(ss2Types)>0]
	# 1.b: for each mode, figure out the filename (where we'll save the generates fake data)
	fnames=[subdir+"/predictUncert_"+m+"_"+str(i)+".txt" for i,m in enumerate(modes)]
	conditionalPrint("predictUncert","generating fake data into file(s):"+str(fnames))
	if regen and os.path.exists(subdir+"/"):	# remove previous run's saved contour files
		shutil.rmtree(subdir+"/",ignore_errors=True)
	# 1.c for each mode, generate x axis datapoints, and pass them into the function which generates the data
	for i,(f,mode) in enumerate(zip(fnames,modes)):
		if len(ss2Types)>0:
			setParam("fm",ss2fms[i]) ; setParam("rpr",ss2rprs[i]) ; setParam("rpu",ss2rpus[i])
			for var in magicMods.keys():
				setVar(var,magicMods[var][i])
		xs={    "TDTR":np.logspace(np.log(300e-12)/np.log(10),np.log(5.5e-9)/np.log(10),30) ,
			"SSTR":np.linspace(getVar("Pow")/100,getVar("Pow"),30) , 
			"FDTR":np.logspace(2,7,100) , 
			"PWA":np.linspace(0,1/fm,1024,endpoint=False) , # TODO use of dict here is weird. each element executes even if it's not selected, which is wasteful, and more importantly, crashes if you, say, are running hypothetical contours on fdtr and pass a dummy fm of zero
			"FD-TDTR": np.linspace(minimum_fitting_time,5500e-12,40) }[mode]
		conditionalPrint("predictUncert","generating "+mode+" data with params:",pp=True)
		ys=func(xs,store=f,addNoise=addedNoise)
	if filesOnly:
		return fnames
	# Step 2, run contour uncertainty!
	# 2.a solver function and what we pass to it is either ss2, or solve(), and a list, or singular file, depending on if simultaneous or not
	solver={True:ss2,False:solve}[len(ss2Types)>0] ; fname={True:fnames,False:fnames[0]}[len(ss2Types)>0]
	conditionalPrint("predictUncert","using function "+str(solver)+" for solving")
	# 2.b prepare for measureContour1Axis with any special actions, eg, TDTR needs doPhasecorrect=False in order to read fake files correctly
	#setVar("doPhaseCorrect",False) # update, no longer needed? added a min(ts) check to phaseCorrect()
	# 2.c do it!  
	#for p in tofit: # do we want to run contours for all tofit params? or just the first one?
	p=tofit[0] ; v=getParam(p)
	valRange,fout=measureContour1Axis(fname, p, plotting="savefinal", resolution=120, ranges=[v*.25,v*1.75], extend=False, threshold=threshold, nworkers=nworkers, solveFunc=solver)	# mC1A > solve > various depending on mode
	if allParams:
		for p in tofit[1:]:
			v=getParam(p)
			vR,fo=measureContour1Axis(fname, p, plotting="savefinal", resolution=120, ranges=[v*.25,v*1.75], extend=False, threshold=threshold, nworkers=nworkers, solveFunc=solver)	# mC1A > solve > various depending on mod
	return valRange
"""


tsize='' ; lastProg=-1
def progBar(n,N):
	global tsize
	if type(tsize)==str: # first call, we look up terminal window width
		try:
			tsize=os.get_terminal_size()[0]
		except:
			tsize=-1
			return
	global lastProg
	p=n/N
	if tsize<0:
		#print(n,"/",N)
		return
	p=int(round(p*tsize))
	if p>lastProg:
		print(".",end='',flush=True)
		lastProg=p
	if p<lastProg:
		lastProg=p

#Given a set of files for the same sample, compute a single value for the result and uncertainty, with uncertainty found by combining 3 possible methods. Note: conceivable multple measurements at multiple modulation frequencies would be done, so a nested list for fileSet is passable (for modes 2 and 3). We'll then simultaneous-solve each subset.
# there are 4 potential sources of uncertainty:
# 1. variation within a sample: natural variation spot-to-spot or sample-to-sample will yield some spread
#	of fitted results. This is quantified by taking multiple measurements, solving for each (yielding
#	results "xₙ"), and taking the mean (x̄=Σxₙ/N) and standard deviation (σ=√(Σ[(xₙ-x̄)²]/N)). error due
#	to sample-to-sample variation is thus e₁: ±σ
# 2. propagated uncertainty in assumed parameters: any uncertainty in assumed parameters (eg, d1, spot size,
#	etc) will propagate to your fitted values. For example, imagine you aren't sure if your transducer
#	thickness is 78nm or 82nm, but fitting with either changes your result by 10%. your result is 
#	heavily dependant on the assumption for d1, so you can't claim to know your fitted result absolutely
#	without knowing all assumptions absolutely. This is quantified by simply solving for each
#	measurement (yielding result "x"), then perturbing each assumed parameter (p) and re-solving
#	(yielding affected result "xₚ"). error is thus: e₂: ±√(Σ[(x-xₚ)²]).
# 3. systematic error: imagine there is some (hypoethetical) undetected issue within the measurement system
#	which simply increases the slope of your TDTR decay curve. This means you will fit for a higher 
#	thermal conductivity than the actual sample's value. This could be non-hypothetical, due to a
#	 misalligned delay stage, issues with the spot overlap, etc, too. We would hope to detect this by
#	running daily calibration scans, and making sure the found values are within some bounds. If you
#	Sapphire conductivity is 34-36, you keep the system as-is, but if it's 37, you go back and check for
#	misallignment etc. This means the difference between the K=34 and K=36 curves represent the level of
#	systematic error that we allow in our system; we catch systematic issues that yield a calibration's
#	tdtr decay curve outside of these, but accept anything else. This means we should "accept" any
#	combination of fitted parameters that yield a residual ("how far the data is off from the curve")
#	similar to the difference between our accepted calibration values. This uses the Feser & Cahill
#	residual for contouring, but with a less arbitrary threshold. to measure uncertainty due to
#	systematic error, we can either a) generate "every combination" for our N fitted parameters (ND grid
#	of test values) and check the residual, or b) test "every value" for a single parameter of interest,
#	setting it, and fitting for the others to find the best fit and lowest residual. the bounds of
#	values for your fitted parameters that yield a "good" fit (within threshold) are your error.
# 4. uncertainty due to noise: pick an arbitrary function, generate a "noisy" dataset which should follow
#	the function, and then fit the function to the dataset. You may not necessarily come up with the
#	exact parameters that created that fake dataset, dependant on the function's sensitivity to the
#	fitted parameters (e.g. the case where you can change either of the two parameters to find the same
#	curve), or dependant on the level of noise in the data. This is separate from (3), as here, we can
#	accurately quantify the level of noise in the system, whereas below a certain threshold, we can not
#	detect systematic error. to quantify uncertainty due to noise, we can either a) repeatedly generate 
#	noisy fake datasets with our found parameters, and test the spread of results upon re-fitting, or
#	we can b) take the diagonal from the variance-covariance matrix as output by scipy's curve_fit, as
#	these two methods should yield similar results. TODO need to add this to compUncert
#	https://en.wikipedia.org/wiki/Least_squares#Uncertainty_quantification

# Combining: 1 and 2 can be combined for N measurements: x = Σxₙ/N ± √( e₁² + (Σ[eₙ₂]/N)² )
# 3 can also be added by simply adding:                                     + (Σ[eₙ₃]/N)²
#
# all discussed so far is found for one single sample. if multiple samples of the same type exist, their repeat measurements can be lumped into (2).
#"plotting" options include: first cell:solve():  show, save, showsens, savesens, none. second cell:perturbUncertainty(): none, show, save, showall. third cell:contours: show, showall, save, saveall (third cell mode 3, contour overlapping: whether we show/save just the final overlapped, or each individual contour plot from R, M, X, Y). DEFAULT FILE LOCATIONS ARE: where/youre/running/from/pics/ for solve/uncert pics. where/data/is/contours/ for text files, /where/data/is/contours/pics/ for contour's pics. "rmxy" controls options to explore contours for Ratio, Magnitude, X, and Y signals
def comprehensiveUncertainty_1(fileSet,modes="123",filename=False,sampleName="",plotting=["save","none","none","savefinal"],refit=True,	# basic args
		paramsToPerturb='',perturbBy='' , 					# assumed param args: perturbBy is percent, default all by 5(%)
		rmxy="RMXY",contourResolution=[20,20],threshold=2.5,multiThresh='', 		# contours args: overlap ratio, magnitude, etc, sample
		contourResolutions2=100,bonusCriteria="True",skipParams=""):
	# before we begin, save off modifiable settings (eg, tp may contain guesses that affect solving, so we'll deep copy before each solve)
	global tp,fitting
	oldFitting=fitting ; tpOrig=copy.deepcopy(tp)

	lp=len(tofit) ; lf=len(fileSet)
	r=np.zeros((lp)) ; e=np.zeros((4,lp)) #which mode, which parameter, error not as a percent
	
	# SOLVE (YIELDS RESULTS FOR ALL METHODS)
	conditionalPrint("comprehensiveUncertainty","preliminary work: solving for each file(set)")
	p=plotting[0]
	r1=np.zeros((lf,lp)) ; goods=[False]*lf #which file, which parameter. results from some files may be bogus, so we chuck those out
	conditionalPrint("comprehensiveUncertainty","solving for: "+str(fileSet))
	for i,f in enumerate(fileSet):
		tp=copy.deepcopy(tpOrig) #restore tp (original may have contained guesses) before solving
		s,[res,sig]=solve(f,plotting=p,refit=refit)
		if type(f)==list:	# solve(onefile) returns [param1,param2],residual. whereas solve(listoffiles) (solveSimultaneous) returns [param1,param2],[residual,for,each,file]
			res=min(res)
		#print(s,res)
		r1[i]=s ; goods[i]=(res<=threshold/100 and eval(bonusCriteria)) # eg "r1[i][1]<=300e6" to only keep if 2nd param is less than 300e6
	r=np.mean(r1[goods],axis=0)

	# REPEATABILITY
	if "1" in modes:
		conditionalPrint("comprehensiveUncertainty","calculating uncertainty from repeatability")
		e[0]=np.std(r1[goods],axis=0) #error is simply standard deviation, σ=√(Σ[(xₛ-x̄)²]/N)
		conditionalPrint("comprehensiveUncertainty","found: r:"+str(r1)+",e:"+str(e[0]))

	# INFLUENCE OF ASSUMED PARAMETERS
	if "2" in modes:
		conditionalPrint("comprehensiveUncertainty","calculating uncertainty from assumed parameters")
		#r2=np.zeros((lf,lp)) # uncertainty gets results from solve() anyway, so no need to save these off?
		e2=np.zeros((lf,lp)) #which file, which parameter
		p=plotting[1]
		conditionalPrint("comprehensiveUncertainty","running perturbUncertainty() for: "+str(fileSet))
		for i,f in enumerate(fileSet):
			#print(f)
			tp=copy.deepcopy(tpOrig) #restore tp (original may have contained guesses) before solving
			s,u,pa=perturbUncertainty(f,plotting=p,paramsToPerturb=paramsToPerturb,perturbBy=perturbBy) #solution, and uncertainty: K,G = X,Y, ± x,y (not as a percent)
			#r2[i]=s
			e2[i]=u
		#r[1]=np.sum(r2,axis=0)/lf
		e[1]=np.sum(e2[goods],axis=0)/lf #simply average of all samples' result/error: Σ[rₛ]/N, Σ[eₛ]/N . TODO, consider using √(Σ[e₁²])/N instead?
		conditionalPrint("comprehensiveUncertainty","found: r:"+str(r1)+",e:"+str(e2))

	# SENSITIVITY OF FITTED PARAMETERS
	if "3" in modes: # TODO WARNING, mode 3 won't work for 3 parameters (comprehensiveUncertainty > measureContours > generateHeatmap. GH can handle it, but MC can not. 
		conditionalPrint("comprehensiveUncertainty","calculating uncertainty from 2D contours")
		if len(tofit)!=2:
			print("error, tofit is not of length 2, skipping")
		p=plotting[2]
		r3=np.zeros((lf,lp)) ; e3=np.zeros((lf,lp))
		#if len(multiThresh)==0:
		#	multiThresh=[threshold]
		for i,fs in enumerate(fileSet):
			ranges=np.zeros((2,2)) ; ranges[:,0]=r*.5 ; ranges[:,1]=r*1.5 ##[[p1lb, p1ub], [p2lb, p2ub]], go from result ±50%
			resolutions=contourResolution
			#for n,threshold in enumerate(multiThresh):
			r3[i],limits=measureContours(fs,paramRanges=ranges,paramResolutions=resolutions,threshold=threshold/100)
			#print(ranges,limits)
			error=(limits[:,1]-limits[:,0])/2
			#ranges=limits
			e3[i][:]=error[:]
		conditionalPrint("comprehensiveUncertainty","found: r:"+str(r3)+",e:"+str(e3))
		e[2]=np.sum(e3,axis=0)/lf
	# SENSITIVITY OF FITTED PARAMETERS, version 2 (instead of generating a 2D grid of X,Y points and generating the decay curve to compare to data, we march along a 1D grid of X points, fitting for the remaining parameter. This works for 3D and beyond, and achieves better resolution / faster). 
	if "4" in modes:
		conditionalPrint("comprehensiveUncertainty","calculating 1D contour uncertainty")
		p=plotting[3] 
		e4=np.zeros((lf,lp))
		for i,fs in enumerate(fileSet): # for each file (or set of files, for multifreq)
			for j,param in enumerate(tofit):
				if param in skipParams:
					continue
				bnds,fout=measureContour1Axis(fs,paramOfInterest=param,plotting=p,resolution=contourResolutions2,threshold=threshold/100)	# and "solve" using them
				error=(bnds[1]-bnds[0])/2 # error not as a percent
				e4[i][j]=error				
				conditionalPrint("comprehensiveUncertainty",param+": "+str(bnds)+" +/- "+str(error))
		e[3]=np.sum(e4,axis=0)/lf
		conditionalPrint("comprehensiveUncertainty","found: e:"+str(e[3]))
	tp=copy.deepcopy(tpOrig) #restore tp (original may have contained guesses) before exiting
	results=r
	error=np.sqrt(np.sum(e**2,axis=0)) #√( e₁² + e₂² + e₃² ) (recall: e₁=Σ[e₁ₛ]/N,  e₂=σ(rₛ), e₃=Σ[e₃ₛ]/N, e₁ₛ came from perturbing assumptions, e₃ₛ from contours)
	conditionalPrint("comprehensiveUncertainty","FINISHED: e:"+str(e))
	conditionalPrint("comprehensiveUncertainty","SUMMED: r:"+str(results)+",e:"+str(error))
	fitting=oldFitting
	if filename:
		f=open(filename,'a')
		line=sampleName
		for p,r,e in zip(tofit,results,error):
			line=line+"\t"+p+"="+str(r)+"+/-"+str(e)
		conditionalPrint("comprehensiveUncertainty","WRITING TO FILE: "+line)
		f.write(line+"\n")
		f.close()
	return results,error #ERROR IS NOT AS A PERCENTAGE

# HOW DOES kwargSets WORK? It's a dict, with a key for each function we call. each value is another dict, which contains the keyword-arguments passable into that function. kwargSetDefaults (see below) contains some reasonable and sane defaults, but you can override these via kwargSets
def comprehensiveUncertainty(fileSet,modes="123",kwargSets="",full=False):
	kwargSets=dict(kwargSets)
	kwargSetDefaults={"solve":{"plotting":"save","refit":False},
			"perturbUncertainty":{"plotting":"none","reprocess":False},
			"measureContour1Axis":{"plotting":"savefinal"} }
	kwargSets.update(kwargSetDefaults) # adds unpopulated "default" elements into (without overwriting) user-defined dict
	error=np.zeros((3,len(tofit))) # which uncertainty calculation method, which fitted parameter
	results=np.asarray( [ solve(f,**kwargSets["solve"])[0] for f in fileSet ] ) # which file, which parameter
	resCombined=np.mean(results,axis=0)
	conditionalPrint("comprehensiveUncertainty","initial solving / averaging: "+str(results)+" --> "+str(resCombined))
	if "1" in modes:
		error[0,:]=np.std(results,axis=0)
		conditionalPrint("comprehensiveUncertainty","mode==1 : stdev = "+str(error[0,:]))
	if "2" in modes:
		paramsToPerturb=kwargSets["perturbUncertainty"].get("paramsToPerturb",'')# grab either whatever the caller
		perturbBy=kwargSets["perturbUncertainty"].get("perturbBy",'')		#    gave us, or default to empty
		paramsToPerturb,perturbBy=updatePTPPB(paramsToPerturb,perturbBy)	#    (empty will be populated with defaults)
		kwargSets["perturbUncertainty"]["paramsToPerturb"]=paramsToPerturb	# re-add these to the dict, so **kwargs (below)
		kwargSets["perturbUncertainty"]["perturbBy"]=perturbBy			#    doesn't give duplicate keyword args
		e2=np.zeros((len(fileSet),len(tofit)))
		for n,f in enumerate(fileSet):
			s2,u2,perturbResults=perturbUncertainty(f,**kwargSets["perturbUncertainty"])
			e2[n,:]=u2[:]
		error[1,:] = np.mean(e2,axis=0) # simple mean to combine multiple files worth of perturb uncertainty: Σ[rₛ]/N, Σ[eₛ]/N
		conditionalPrint("comprehensiveUncertainty","mode==2 : propagated uncertainty for each: "+str(e2)+" --> "+str(error[1,:]))
	if "3" in modes:
		e3=np.zeros((len(fileSet),len(tofit))) # which file, which fitting parameter
		for n,f in enumerate(fileSet):
			for i,param in enumerate(tofit):
				valRange,fout=measureContour1Axis(f, param, **kwargSets["measureContour1Axis"])
				e3[n,i]=(valRange[1]-valRange[0])/2
		error[2,:] = np.mean(e3,axis=0)
		conditionalPrint("comprehensiveUncertainty","mode==3 : contour uncertainty for each: "+str(e3)+" --> "+str(error[2,:]))
	# combining error: use the geometric mean (no "zero" uncertainty calculation should lessen the uncertainty from other calculation methods)
	# error=np.sqrt(np.sum(e**2,axis=0)) #√( e₁² + e₂² + e₃² )
	conditionalPrint("comprehensiveUncertainty","finished modes "+modes+" : error = "+str(error))
	if not full:
		error = np.sqrt(np.sum(error**2,axis=0)) # which calc method, which fitted param --> which param
	conditionalPrint("comprehensiveUncertainty","finished modes "+modes+" : error = "+str(error))
	return resCombined,error

def mergeDict(dictA,dictB): # ASSUMING B DOES NOT OVERWRITE A. goes 2 layers deep
	for k in dictB.keys():
		if k not in dictA.keys():
			dictA[k]={}
		for k2 in dictB[k].keys():
			if k2 not in dictA[k].keys():
				dictA[k][k2]=dictB[k][k2]

def asymmetricComprehensive(fileSet,modes="123",kwargSets="",full=False):
	kwargSets=dict(kwargSets)
	kwargSetDefaults={"solve":{"plotting":"save","refit":True},
			"asymmetricSTD":{"plotting":"save"},
			"perturbUncertainty":{"plotting":"none","reprocess":False},
			"measureContour1Axis":{"plotting":"savefinal"} }
	mergeDict(kwargSets,kwargSetDefaults) # adds unpopulated "default" elements into (without overwriting) user-defined dict
	conditionalPrint("asymmetricComprehensive","running with passed kwargSets: "+str(kwargSets))

	error=np.zeros((3,len(tofit),2)) # which uncertainty calculation method, which fitted parameter, uncertainty up vs down
	results=np.asarray( [ solve(f,**kwargSets["solve"])[0] for f in fileSet ] ) # which file, which parameter
	resCombined=np.mean(results,axis=0)
	if "1" in modes:
		for i in range(len(tofit)):
			kwargSets["asymmetricSTD"]["xlabel"]=tofit[i]
			print(kwargSets["asymmetricSTD"])
			m,(stdDn,stdUp)=asymmetricSTD(results[:,i],**kwargSets["asymmetricSTD"])
			error[0,i,0]=stdDn ; error[0,i,1]=stdUp
			resCombined[i]=m
	if "2" in modes:
		paramsToPerturb=kwargSets["perturbUncertainty"].get("paramsToPerturb",'')# grab either whatever the caller
		perturbBy=kwargSets["perturbUncertainty"].get("perturbBy",'')		#    gave us, or default to empty
		paramsToPerturb,perturbBy=updatePTPPB(paramsToPerturb,perturbBy)	#    (empty will be populated with defaults)
		for i in range(len(paramsToPerturb)):					#    and then cycle through, adding negatives
			p,v = paramsToPerturb[i],perturbBy[i]
			paramsToPerturb.append(p) ; perturbBy.append(-v)		# TODO havoc if user supplies different perturbs +/-??
		kwargSets["perturbUncertainty"]["paramsToPerturb"]=paramsToPerturb	# re-add these to the dict, so **kwargs (below)
		kwargSets["perturbUncertainty"]["perturbBy"]=perturbBy			#    doesn't give duplicate keyword args
		uncDnUp=np.zeros((len(fileSet),len(tofit),2)) # which file, which fitting parameter, uncertainty up vs down
		for n,f in enumerate(fileSet):
			s2,u2,perturbResults=perturbUncertainty(f,**kwargSets["perturbUncertainty"])		# see testing74.py for processing of perturbResults
			# perturbResults holds triplets of: paramName,perturbedBy,[delParam1,delPAram2...]
			print(n,[ pr for pr in perturbResults ])
			# and for normal symmetric uncertainty, we simply sum over perturbed parameters by: sum(delResults**2.)**.5
			dRes=np.asarray( [ pr[2] for pr in perturbResults ] )
			for i,dR in enumerate(dRes.T):
				uncDnUp[n,i,0] = np.sum( dR[dR<0]**2 )**.5
				uncDnUp[n,i,1] = np.sum( dR[dR>0]**2 )**.5
		error[1,:,:] = np.mean(uncDnUp,axis=0) # simple mean to combine multiple files worth of perturb uncertainty: Σ[rₛ]/N, Σ[eₛ]/N
		#print("std,dn,up",u,error[1,:,0],error[1,:,1]) # FASCINATING OBSERVATION: DOUBLE PERTURBS (e.g. d1 up by 5% and also d1 down by 5%) GIVES "DOUBLE" THE PERTURB UNCERT. we don't divide this sqrt(sum(squares)) by the length (because adding more ineffective perturbs, such as the last layer's thickness) should *not* serve to reduce the uncertainty). so perturbUp *and* perturbDown will give too much perturb uncertainty.
	if "3" in modes:
		contDnUp=np.zeros((len(fileSet),len(tofit),2)) # which file, which fitting parameter, uncertainty up vs down
		for n,f in enumerate(fileSet):
			for i,param in enumerate(tofit):
				valRange,fout=measureContour1Axis(f, param, **kwargSets["measureContour1Axis"])
				print(valRange,fout)
				contDnUp[n,i,0] = results[n,i]-valRange[0]
				contDnUp[n,i,1] = valRange[1]-results[n,i]
		error[2,:,:] = np.mean(contDnUp,axis=0)

	# combining error: use the geometric mean (no "zero" uncertainty calculation should lessen the uncertainty from other calculation methods)
	# from comprehensiveUncertainty: error=np.sqrt(np.sum(e**2,axis=0)) #√( e₁² + e₂² + e₃² )
	if not full:
		error = np.sqrt(np.sum(error**2,axis=0)) # which calc method, which fitted param, uncert up vs down --> which param, up vs down
	return resCombined,error


def asymmetricSTD(vals,plotting="none",xlabel=""):
	print(vals,plotting,xlabel)
	from scipy.stats import skewnorm
	mean,stdev=np.mean(vals),np.std(vals)
	skew,center,scale=skewnorm.fit(vals,loc=mean,scale=stdev) # fit a skewed normal distribution to the dataset
	xs=np.linspace(min(vals),max(vals),1000)
	ys=skewnorm.pdf(xs,skew,center,scale) # generate curve for skewed normal distribution
	n=np.argmax(ys) ; mode=xs[n] # if you had perfect binning for a histogram, and high enough sampling, your mode is the peak
	# stdev = √( Σ( (xᵢ-μ)² ) / N )
	valsAbove=vals[vals>mode] ; valsBelow=vals[vals<mode]
	if len(valsAbove)>1 and len(valsBelow)>1:
		stdUp=np.sqrt(np.sum((valsAbove-mode)**2)/len(valsAbove))
		stdDn=np.sqrt(np.sum((valsBelow-mode)**2)/len(valsBelow))
	else:
		skew=0 ; stdUp=stdev ; stdDn=stdev ; ys=skewnorm.pdf(xs,skew,mean,stdev) ; mode=mean
	if plotting!="none":
		filename=figFile("asymmSTD_"+xlabel,plotting,subfolder="pics")
		# PLOT THE HISTOGRAM AND SKEWED PROBABILITY FUNCTION
		histo,bins=np.histogram(vals,bins=int(len(vals)/5)) ; bins=(bins[1:]+bins[:-1])/2
		ys/=max(ys) ; ys*=max(histo)
		Xs=[bins,xs,[mode],[mode-stdDn]*2,[mode+stdUp]*2] ; Ys=[histo,ys,[ys[n]],[0,max(histo)],[0,max(histo)]]
		mkrs=["k","k:","r","r:","r:"] ; lbs=[""]*5
		#Xs.append([np.mean(v)]*2) ; Ys.append([0,max(histo)]) ; mkrs.append("b:") ; lbs.append("mean")
		#Xs.append([np.median(v)]*2) ; Ys.append([0,max(histo)]) ; mkrs.append("g:") ; lbs.append("median")
		lplot(Xs,Ys,xlabel=xlabel,ylabels="counts",xlim=["nonzero"],markers=mkrs,labels=lbs,title="skew="+str(skew),filename=filename)
	return mode,(stdDn,stdUp)

def readCompOut(filename): # comprehensiveUncertainty allows saving results to an output file, each line containing "sampleID \t param1=val+/-err \t param2...". here we'll parse that file for you. it's up to you to organize them (grouping up which sample ID goes in which set, etc). Please check out IBB/GaN -CNGa/GaN-CNGa.py > GaN-all3 for a good idea on how to use TDTR_tools' groupFiles() function to organize your results
	lines=open(filename,'r').readlines()
	IDs=[] ; results=[] ; error=[]
	for l in lines: # "MURI60	G1=112e6+/-8e6	Kz2=263+/-25"
		l=l.split() # ["MURI60", "G1=112e6+/-8e6", "Kz2=263+/-25"]
		ID=l[0]
		if ID in IDs:
			i=IDs.index(ID)
		else:
			IDs.append(ID) ; results.append({}) ; error.append({}) ; i=-1
		for chunk in l[1:]: # "G1=112e6+/-8e6"
			param,res=chunk.split("=") # "G1","112e6+/-8e6"
			res,err=res.split("+/-")
			results[i][param]=float(res)
			error[i][param]=float(err)
	return IDs,results,error	

layerNames=[] # you can use setVar for this, to label each layer. we either put "d1","Kz2" etc in the legend, OR, if you've supplied material names for each layer, "d_Al","Kz_Al2O3" etc
normalizeSensitivity=False ; sensitivityAsPercent=False
def sensitivity(percentPerturb=.01,plotting="show",title="",customPerturbs={'rpu':.1,'rpr':.1},xs=""): #for all parameters in "tofit", perturb them by n%, and check the change to the TDTR delay curve. this is indicative of how sensitive one is to that parameter (how easily one is able to extract this property). note: "plotting" options include: show, save, none. customPerturbs allows us to play with our percentPerturb on an individual-parameter level. Eg, maybe i'm not measuring my 10um diameter beam spot to the nanometer, and i want to see what happens if my beam spot size is doubled (perturb by 100%). default percentPerturb is fine when all parameters have equal footing (eg, for fitting), but if you're using sensitivity() to check for the influence of an assumed parameter and suspect there's a possibility that your assumed value may be way off, you can use customPerturbs
	conditionalPrint("sensitivity","",pp=True)
	if len(xs)==0:
		xs={    "TDTR":np.linspace(minimum_fitting_time,5500e-12,40) , 
			#"TDTR":np.linspace(minimum_fitting_time,1.9e-6,1000) , 
			"SSTR":np.linspace(0,getVar("Pow"),10) , 
			#"FDTR":np.logspace(2,8,100) , 
			"FDTR":np.logspace(4.5,7.1,100) ,
			"PWA":np.linspace(0,1/fm,100,endpoint=False) ,
			"FD-TDTR": np.linspace(minimum_fitting_time,5500e-12,40) }[mode]
	#f={"TDTR":TDTRfunc , "SSTR":SSTRfunc , "FDTR":FDTRfunc , "PWA":PWAfunc , "FD-TDTR":TDTRfunc}[mode]
	f=func
	xlabel={ "TDTR":"time delay (ps)" , "SSTR":"Pump power (mW)" , "FDTR":"frequency (Hz)" , "PWA":"t (ms)" , "FD-TDTR":"time delay (ps)"}[mode]
	factor={"TDTR":1e12,"SSTR":1,"FDTR":1,"PWA":1e3}[mode]
	xscale={"TDTR":"linear","SSTR":"linear","FDTR":"log","PWA":"linear"}[mode]

	initParams=getTofitVals()
	origVals=f(xs)
	dSdP=np.zeros((len(tofit)+1,len(xs)))#d signal / d [Parameter]. len+1 gives us a free line at zero at the end
	dPs=[origVals]
	for i,param in enumerate(tofit):	# for each parameter we'd be fitting for
		perturbedParams=initParams[:]	# copy initial params
		p=1+customPerturbs.get(param,percentPerturb)/100. # if param is in customPerturbs, use that, else, use percentPerturb
		perturbedParams[i]*=p		# perturb it by that much
		conditionalPrint("sensitivity","perturbing "+param+":"+str(initParams)+" --> "+str(perturbedParams))
		perturbedVals=f(xs,*perturbedParams)	# generate new TDTRfunc
		P=initParams[i] ; dP=perturbedParams[i]-initParams[i] # "how much did we change this parameter by"
		S=origVals ; dS=perturbedVals-origVals # "how much did the resulting curve change by"
		conditionalPrint("sensitivity","signal changes by"+str(np.mean(dS[1:]/S[1:])*100)+"%")
		# A. Eq 1 from: Gundrum, Cahill, Averback "Thermal Conductance of metal-metal interfaces" PRB 72, 245426 (2005)
		# Sα = d ln(-Vᵢₙ/Vₒᵤₜ) / d ln(α)
		# B. Eq 10 from: Schmidt, Cheaito, Chiesa "A frequency-domain thermoreflectance method for the characteriza..." RSI 80, 094901 (2009). 
		# Sₓ = d ϕ / d ln(x)
		# C. Hopkins' Lab Sensitivity matlab code (e.g. SensitivityCode_time.m)
		# R=-Vᵢₙ/Vₒᵤₜ (l85) ; Rperturbed=-Vᵢₙ/Vₒᵤₜ (l179) ; dP=Pperturbed-P (l150) ; S=(Rp-R)/(dP*R) (l180)
		# Note A+C are mathematically the same! d(ln(Z)) tells you the percentage change
		# so A+C are all in effect "the percent change in the curve" d(ln(curve)) "divided by the percent change in the parameter" d(ln(x))
		# BEWARE: this expression over-estimates the sensitivity where the function is small. e.g. PWA sensitivity blows up near zero-crossings
		# and TDTR sensitivity might lead you to believe it is better to measure certain parameters at long time delays. I can't say for sure,
		# but I suspect this is why Schmidt got rid of the natural log in the numerator. for PWA code using this method (don't), see old version 								
		# of TDTR_fitting.py, versions 0.162 or prior
		if sensitivityAsPercent: # PERCENT CHANGE IN CURVE vs PERCENT CHANGE IN THE PARAMETER
			#div=( np.log(S+dS)-np.log(S) )/( np.log(P+dP)-np.log(P) ) 
			div=dS/S/(percentPerturb/100) # note technically (dS/S)*(P/dP) is "correct", but this cancels out our custom perturbs
		else: # NOMINAL CHANGE IN CURVE vs PERCENT CHANGE IN THE PARAMETER
			#div=dS/(np.log(P+dP)-np.log(P))
			div=dS/(percentPerturb/100)
		dSdP[i,:]=div
		dPs.append(perturbedVals)
	setTofitVals(initParams) #cleanup! since TDTRfunc for the last set of perturbedParams updated tp
	#dSdP=[dsdp/dsdp[0] for dsdp in dSdP]
	#dSdP=[dsdp/dsdp[np.argmax(abs(dsdp))] for dsdp in dSdP]
	if normalizeSensitivity:
		i1=int(len(xs)/4) ; i2=int(len(xs)*3/4)
		for i,dsdp in enumerate(dSdP):
			n=np.argmax(abs(dsdp[i1:i2]))+1
			dSdP[i][:]/=dsdp[i1+n]

	if plotting!="none":
		mkrs=['b-','g-','r-','c-','m-','b,-.','g,-.','r,-.','c,-.','m,-.','b,--','g,--','r,--','c,--','m,--']
		bonusTitle=title
		titles=["Sens: "+fitting] + [p+"="+str(scientificNotation(v,2)) for p,v in zip(tofit,initParams)] # list of "param=val" sets
		title=[titles[0]]
		for t in titles[1:]:
			if len(title[-1]+", "+t)>40:
				title.append(t)
			else:
				title[-1]=title[-1]+", "+t
		title="\n".join(title)+bonusTitle
		filename=""
		if "save" in plotting:
			filename="sensitivities/"+"-".join(tofit)+plotting.replace("save","")+".png"
		if len(tofit)>0: # normal case: actually running uncertainty
			Ys=dSdP ; Xs=[xs*factor]*(len(tofit)+1) ; mkrs=mkrs[:len(tofit)] + ["k:"] ; dlbs=tofit+[""]
		else: 		# call sensitivity() with no tofit params? we instead just plot the expected curve
			Ys=dPs+[dSdP[0]] ; Xs=[xs*factor]*(len(tofit)+2) ; mkrs=[mkrs[0],"k:"] ; dlbs=["",""]

		#plot([xs]*len(dSdP),dPs,xlabel,datalabels=dlbs,markers=mkrs,legendLoc="inline")# ; return
		#print("PLOTTING",Xs,Ys,filename)
		#lplot(Xs, Ys, xlabel, "", labels=dlbs, markers=mkrs, title=title, filename=filename, xscale=xscale)
			#,forcedBoundsX=[0,max(Xs[0])],forcedBoundsY=[0,None])#,legendLoc="inline") #legendLoc="lower right")
		# inline legendLoc was deprecated in the switch from plotter to niceplot. use niceplot's wild-west "extras" instead and do it yourself
		def inline(axs,fig): # https://stackoverflow.com/questions/16992038/inline-labels-in-matplotlib
			for ax in axs:
				for i,(xs,ys,lb) in enumerate(zip(Xs[:-1],Ys[:-1],dlbs[:-1])):
					n=int((i+1)/(len(Xs))*len(xs))
					x=xs[n] ; y=ys[n]
					ax.text(x,y,lb)
		lplot(Xs, Ys, xlabel, "", labels=dlbs, markers=mkrs, title=title, filename=filename, xscale=xscale,extras=[inline])
	return dSdP

def deleteLayer(N): # simply deletes layer N (first layer is 1), and it's following interface
	global tp
	i=(N-1)*2
	del tp[i] ; del tp[i]

def refreshPlotGlos():
	global plotXs,plotYs,plotXlabel,plotYlabel,plotLabels,plotTitle,plotMarkers
	#global plotZs,
	plotXs=[] ; plotYs=[] ; plotXlabel="" ; plotYlabel="" ; plotLabels=[] ; plotTitle="" ; plotMarkers=[] ; fignames=[]
refreshPlotGlos()

fignames=[]
def lcontour(zs,xs,ys,xlabel="",ylabel="",useLast=False,**kwargs):
	from nicecontour import contour,getContObjs,getCS
	#sys.path.insert(1,"../niceplot")
	#kwargs["heatOrContour"]="surface"
	#from nicecontour import contour
	conditionalPrint("lcontour","received kwargs: "+str(kwargs)+",useLast="+str(useLast))
	figfiles=fignames+[kwargs.get("filename","")]			# add this passed filename to list
	figfiles=[ f for f in figfiles if len(f) > 1 ]			# filter out empty fnames (if plotting==show but fignames was pop'd, use those)
	kwargs["xlabel"]=xlabel ; kwargs["ylabel"]=ylabel
	if len(figfiles)==0:						# if empty, re-add '', which will show the fig. this gets us into the loop
		figfiles.append('')
	for f in figfiles:
		if "csv" in f:
			continue
		kwargs["filename"]=f
		conditionalPrint("lcontour","calling 'contour', with kwargs:"+str(kwargs)+",useLast="+str(useLast))
		contour(zs,xs,ys,useLast=useLast,**kwargs)

def lplot(xs,ys,xlabel="",ylabel="",**kwargs): # This is a wrapper function which EITHER saves off info to globals (for retrival and live-plotting by gui.py), OR calls into niceplot.py > plot (replacing plotter.py)
	from niceplot import plot,getPlotObjs
	kwargs["xlabel"]=xlabel ; kwargs["ylabel"]=ylabel
	#sys.path.insert(1,"../niceplot")
	# HOW DOES THE GUI GET THE DATA TO BUILD AN INTERACTIVE PLOT? gui > TDTR_fitting > niceplot. we'll create globals and populate them
	global plotXs,plotYs,plotXlabel,plotYlabel,plotLabels,plotTitle,plotMarkers
	useLast=kwargs.get("useLast",False) # common to plot multiple fits on the same plot, so then we'll append above lists, instead of overwriting
	plotXlabel=kwargs.get("xlabel","")
	plotYlabel=kwargs.get("ylabel","")
	plotTitle=kwargs.get("title","")
	labels=kwargs.get("labels",[]) # some kwargs need to be translated from plotter.py > plot() to niceplot.py > plot()
	markers=kwargs.get("markers",[])
	if useLast:
		plotXs+=list(xs) ; plotYs+=list(ys) ; plotLabels+=list(labels) ; plotMarkers+=list(markers)
	else:
		plotXs=list(xs) ; plotYs=list(ys) ; plotLabels=list(labels) ; plotMarkers=list(markers)
	
	conditionalPrint("lplot","xs:"+str(xs)+",useLast:"+str(useLast)+" -> plotXs:"+str(plotXs))
	# finally, RELOAD kwargs (eg if we're trying to reuse previous markers)
	kwargs["markers"]=plotMarkers ; kwargs["labels"]=plotLabels
	xs=plotXs ; ys=plotYs

	# fignames is a global list of bonus filenames for saving the plot (e.g. gui.py might want to save gui.png). if this is set, we STILL want to save to the passed filename, but we also want to (duplicatively) save to the files in fignames
	figfiles=fignames+[kwargs.get("filename","")]
	figfiles=[ f for f in figfiles if len(f) > 1 ]
	if len(figfiles)>0:
		figfiles.append(figfiles[0].replace(".png",".csv")) # niceplot.py > saveCSV now handles saving csvs
	
	# also save as csv
	#tosave=np.zeros(( max( [ len(x) for x in xs ] ) , len(xs)*3 ) ) # 3 columns per dataset (x,y,padding)
	#for i,(x,y) in enumerate(zip(xs,ys)):
	#	tosave[:len(x),i*3]+=x ; tosave[:len(x),i*3+1]+=y # add datasets to columns
	#ts=tosave.astype(str) ; ts[:,2::3]='' # turn into strings, and purge zeros in "padding" columns
	#header=",,,".join(labels)+"\n"+"".join( [xlabel+","+ylabel+",,"]*len(xs) ) # include datalabels and x,y labels
	#os.makedirs("/".join(figfiles[0].split("/")[:-1]),exist_ok=True)
	#np.savetxt(figfiles[0].replace(".png",".csv"),ts,header=header,delimiter=',',fmt="%s")



	if len(figfiles)>0:
		for f in figfiles:
			kwargs["filename"]=f
			plot(xs,ys,**kwargs)

		# also save as csv
		#tosave=np.zeros(( max( [ len(x) for x in xs ] ) , len(xs)*3 ) ) # 3 columns per dataset (x,y,padding)
		#for i,(x,y) in enumerate(zip(xs,ys)):
		#	tosave[:len(x),i*3]+=x ; tosave[:len(x),i*3+1]+=y # add datasets to columns
		#ts=tosave.astype(str) ; ts[:,2::3]='' # turn into strings, and purge zeros in "padding" columns
		#header=",,,".join(labels)+"\n"+"".join( [xlabel+","+ylabel+",,"]*len(xs) ) # include datalabels and x,y labels
		#np.savetxt(figfiles[0].replace(".png",".csv"),ts,header=header,delimiter=',',fmt="%s")

	else:
		plot(xs,ys,**kwargs)
	return
	#filename=kwargs.get("filename","")
	#xlabel=kwargs.get("xlabel","")
	#ylabel=kwargs.get("ylabel","")
	
	#kwargs["xlabel"]=kwargs.get()
	#if len(filename)>0:
	#	niceplot.plot(xs,ys,*kwargs)
	#else:

		
		#if not replot:
		#	plotXs=[] ; plotYs=[] ; plotXlabel="" ; plotYlabel="" ; plotLabels=[] ; plotTitle="" ; plotMarkers=[]
		#for i in range(len(xs)):
		#	plotXs.append(xs[i]) ; plotYs.append(ys[i])
		#	plotLabels.append(kwargs["labels"][i]) ; plotMarkers.append(kwargs["markers"][i])
		#plotXlabel=kwargs["xlabel"] ; plotYlabel=kwargs["ylabel"] ; plotTitle=kwargs["title"]

	#if filename in kwargs.keys():
	#xs, ys, xlabel="xlabel", ylabel="ylabel", title="TITLE", errorY='',		# BASIC DATA AND LABELS
	#markers='', datalabels='', pointlabels='', linewidths=2,			# MARKERS
	#xscale="linear", yscale="linear" , filename='', saveData='', figsize="", 	# SCALING, SAVING
	#includeZeroX=True, includeZeroY=True, forcedBoundsX="", forcedBoundsY="",	# OVERRIDE PLOT BOUNDS
	#forcedTicksX="",forcedTicksY="",
	#multiplot="off", multiplotIndices='', multiplotRatios='',
	#useLast=False, fillIndices='', colorOverride=''

# CHECK THE USERS INPUTS FOR COMMON GOTCHAS. it's not our job to hold your hand and do your analysis for you (collecting data requires rigor and attention to detail. as does analysis). that said, there are some common gotchas we can check for and warn about. this will be implemented as a "checker" which checks your work for you, rather than in-line warnings (which lead to warning fatique). 
def nanny():
	warnings=[]
	# COMMON EXPERIMENTAL PARAMETERS, e.g. fiber SSTR small spot, TDTR bigger spots, none should have meter (instead of μm) sizes
	if rpump>1e-3 or rprobe>1e-3:
		warnings.append("are your spot sizes *really* "+str(rpump)+" m and "+str(rprobe)+" m?")
	if mode=="TDTR" and (rpump<8e-6 or rprobe<4e-6):
		warnings.append("are your TDTR spot sizes *really* "+str(rpump)+" m and "+str(rprobe)+" m?")
	if mode=="SSTR" and (rpump>1.7e-6 or rprobe>1.7e-6):
		warnings.append("are your SSTR spot sizes *really* "+str(rpump)+" m and "+str(rprobe)+" m?")
	# bidirectional, warn if pump and probe depth do not line up with a layer
	if measureAt != 0 or depositAt !=0:
		ds=getCol(2) ; cumdepth=np.cumsum(ds)
		distFromMA=[ abs(measureAt-d) for d in cumdepth ]
		distFromDA=[ abs(depositAt-d) for d in cumdepth ]
		if min(distFromMA)>3e-9 or min(distFromDA)>3e-9:
			warnings.append("double check your bidirectional pump/probe depths, and your sample geometry (depths should line up with an interface) : measureAt="+str(measureAt)+", depositAt="+str(depositAt)+", ds="+str(ds))
	RsGs=getCol(0,"odds")
	# check TBR vs TBC
	if (useTBR and max(RsGs)>1e-6) or ((not useTBR) and min(RsGs)<100):
		warnings.append("you may have entered a TBR as a TBC, or vice versa: useTBR="+str(useTBR)+", "+{True:"Rs",False:"Gs"}[useTBR]+"="+str(RsGs))
	# "if Tmin is something else, like for SiO2 and Sapphire (remember someone messed that up in the GUI)" 
	# check if contours and perturb uncertainty were run for the last-fitted file?
	# check results from auto pump/probe (does running wrapper around gui's runNanny overwrite the values we used for fitting before?)
	

	print(warnings)
	return warnings

### END OTHER TOOLS ###

if __name__=='__main__': #true only if TDTR_fitting is being called directly. if imported, this will be false
	callingScript="TDTR_fitting.py"
	if len(sys.argv)>1:
		helptext=("Call \"python TDTR_fitting.py [-key] [value] ...\"", #TODO 1) add an option to update tofit 2) add "r" param that sets rpu=rpr 3) check if wildcards (*) work for -st, then try, for example, python3 TDTR_fitting.py -m matrfile.txt -tf [r,G1] -st Calubatrions/Date/*Al2O3* on 2021-01-21 cals
			"Options:",
			"  -m  import thermal properties matrix from file",
			"  -s  perform TDTR fitting for single passed data file",
			"  -ss perform simultaneous fitting on multiple, -ss file1,file2,file3 etc",
			"  -se perform sensitivity analysis on passed fitting parameters",
			"  -st solve multiple ad perform statistics, -st f1,f2,f3 etc",
			"  -h  display this help-text")
		if "-h" in sys.argv:
			print("\n".join(helptext))
		#READ IN KEY, VALUE PAIRS
		keys=[] ; values=[]
		for k,v in zip(sys.argv[1::2],sys.argv[2::2]):
			print(k,v)
			keys.append(k) ; values.append(v)
		#IMPORT MATRIX FILE FIRST
		if "-m" in keys:
			matrixFile=values[keys.index("-m")]
			importMatrix(matrixFile)
		#SOLVING, SINGULAR
		if "-s" in keys:
			verbose.append("solve") #turn on verbose printing for solve(), if called manually
			dataFile=values[keys.index("-s")]
			solve(dataFile)
		#SOLVING, SIMULTANEOUS
		if "-ss" in keys:
			dataFiles=values[keys.index("-ss")].split(",")
			solve(dataFiles)
		#SENSITIVITY
		if "-se" in keys:
			#global tofit
			tofit=values[keys.index("-se")].split(",")
			sensitivity()
		#BASIC SOLVE + STATISTICS
		if "-st" in keys:
			dataFiles=values[keys.index("-st")].split(",")
			r,e=comprehensiveUncertainty(dataFiles,modes="2",plotting=["show","show","none"],rmxy="RMXY")
			print(r,"+/-",e)
		#option=sys.argv[1]
		#if option[:5]=="-sens":
		#	sensitivity()
		#elif option[:7]=="-uncert":
		#	params="C1,C2,Kz1,rpu"
		#	params=params.split(',')
		#	f=sys.argv[2]
			#print(perturbUncertainty(f,params))
		#elif option[:5]=="-temp":
		#	plotTz()
		#elif option[:8]=="-heatgen":
			#print("TODO") #TODO
		#elif option[:8]=="-heatplo":
			#print("TODO") #TODO
		#elif option[:5]=="-mult":
		#	files=sys.argv[2:]
			#print(solveSimultaneous(files,plotting="show"))
		#elif option[:5]=="-solv":
		#	f=sys.argv[2]
			#print(solve(f,plotting="show"))
		#elif option[:4]=="-man":
		#	f=sys.argv[2]
			#print(solveManual(f))
		#elif option[:2]=="-h":
			#print(helptext)
		#else:
			#print("error, unrecognized argument: \""+option+"\"")
			#print(helptext)
	else:
		main()
else:
	import traceback				# 'File "/..../Various Code/TDTR_fitting/gui.py", line 22, in <module>
	callingScript=traceback.format_stack()[0]	#  from TDTR_fitting import * ; from plotter import *'
	callingScript=callingScript.split("\"")[1]	# /..../Various Code/TDTR_fitting/gui.py or C:\\Users\\Athena\\...gui.py
	#print("callingScript",callingScript)
	callingScript=callingScript.replace("\\","/")	# make unix paths and windows paths uniform
	callingScript=callingScript.split("/")[-1]+"_"
	callingScript=callingScript.replace("<","").replace(">","")
	#print("callingScript","'",callingScript,"'")
	#callingScript=callingScript.split("/")[-1]+"_"
#KNOWN THERMAL PROPERTIES, REFERENCE
#Heat Capacity (J/m³/K) Thermal Conductivity (W/m/K)
C_Al=2.42e6	;	K_Al=120.00	# Aluminum
C_Au=2.49e6	;	K_Au=120.00	# Gold
C_Sapph=3.06e6	;	K_Sapph=35.	# Sapphire
C_SiO2=1.63e6	;	K_SiO2=1.3	# a-SiO2
C_Si=1.64e6	;	K_Si=140	# Silicon
C_Pt=2.79e6	;	K_Pt=71.6	# Platinum
C_Ti=3e6	; 	K_Ti=25		# Titanium TODO RECHECK THESE TWO
C_GaN=2.64e6	;	K_GaN=130	# Gallium Nitride	https://www.jim.or.jp/journal/e/pdf3/48/10/2782.pdf
C_AlN=2.39e6	;	K_AlN=265 	#Aluminum Nitride 	https://www.memsnet.org/material/aluminumnitridealnbulk/
C_Ge=58/.36*1e4	;	K_Ge=58		# Germanium		http://www.ioffe.ru/SVA/NSM/Semicond/Ge/thermal.html		C=K/α
C_Diamond=1.782e6 ;	K_Diamond=2e3	# Diamond (dcc carbon)	https://en.wikipedia.org/wiki/Table_of_specific_heat_capacities
C_Quartz=1.64e6 ;	K_Quartz_in=6.5 ; K_Quartz_through=11.6 # Quartz
C_bGa2O3=92.1/187.444*5.88*100**3 ; K_bGa2O3=100  #(J/mol·K) / (g/mol) * (g/cm3) * (100cm/m)3
C_aGa2O3=92.1/187.444*6.44*100**3 ; K_aGa2O3=100  #(J/mol·K) / (g/mol) * (g/cm3) * (100cm/m)3
C_W=.134*19.3*1e6 ; 	K_W=174		# Tungsten
C_Fe=.466*7.85*1e6 ;	K_Fe=77.9	# Iron
C_Air=700*1.2	;	K_Air=0.025	# 700 (J/kg/K) * (1.29g/L) * (1kg/1000g) * (1000L/m3)
C_PCM=2.5*.9*1e6 ;	K_PCM=1
C_Mo=.25*10.22*100**3 ; K_Mo=138 	# kJ/kg/K * g/cm3 * 1/1000 kg/g * 1000J/kJ * (100cm/m)3
#helpful references:
#numpy vectorizing: https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array/35216364, https://stackoverflow.com/questions/7701429/efficient-evaluation-of-a-function-at-every-cell-of-a-numpy-array, https://realpython.com/numpy-array-programming/
#misshapen matrix broadcasting: https://stackoverflow.com/questions/23566515/multiplication-of-1d-arrays-in-numpy/23566751, https://stackoverflow.com/questions/14513222/multiplying-numpy-3d-arrays-by-1d-arrays
#multithreading: https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop, https://stackoverflow.com/questions/15143837/how-to-multi-thread-an-operation-within-a-loop-in-python
#integration: https://stackoverflow.com/questions/34346809/integrating-functions-that-return-an-array-in-python, https://docs.scipy.org/doc/scipy-1.2.1/reference/tutorial/integrate.html
#passing arbitrary lists of values to function through scipy.optimize.curve_fit: https://stackoverflow.com/questions/18326524/pass-tuple-as-input-argument-for-scipy-optimize-curve-fit, https://medium.com/understand-the-python/understanding-the-asterisk-of-python-8b9daaa4a558
#python list hacking: #https://stackoverflow.com/questions/903853/how-do-you-extract-a-column-from-a-multi-dimensional-array, https://stackoverflow.com/questions/2631189/python-every-other-element-idiom
#csv import: https://stackoverflow.com/questions/24606650/reading-csv-file-and-inserting-it-into-2d-list-in-python
#python indirection: #https://stackoverflow.com/questions/11553721/using-a-string-variable-as-a-variable-name, #https://bytes.com/topic/python/answers/35885-how-can-i-exec-global
#numpy find index closest to value we're looking for: https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
#dynamically set locals, as opposed to globals via globals()[varname], use dicts: https://stackoverflow.com/questions/8028708/dynamically-set-local-variable