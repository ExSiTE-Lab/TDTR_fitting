# v0.77 (goes with 0.167)
import matplotlib,time,datetime,threading,os,sys
import tkinter as tk
import multiprocessing #; multiprocessing.freeze_support() # https://stackoverflow.com/questions/32672596/pyinstaller-loads-script-multiple-times
from tkinter import filedialog,ttk
from tkinter import *
import tkinter.font as tkf
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
matplotlib.use("TkAgg")
from TDTR_fitting import *
matplotlib.use("svg") # needed for windows apparently...idk why i commented it out before. https://github.com/pyinstaller/pyinstaller/issues/6760 says you'll get a "ModuleNotFoundError: No module named 'matplotlib.backends.backend_svg'" with pyinstaller-compiled (but not running python on windows)
#import time,threading,os,sys

# THIS IS AN ATTEMPT AT A TOTALLY-REVAMPED GUI. TABS FOR MEASUREMENT TECHNIQUES, ETC
# I will still use this basic layout:
#  _________________________________
# |window        _________________  |
# |  _________  |  _____________  | |
# | |         | | |framePlot    | | |
# | |    B    | | |             | | | 
# | |    U    | | |  updatable  | | |
# | |    T    | | |    plot     | | |
# | |    T    | | |             | | |
# | |    T    | | |_____________| | |
# | |    O    | |  _____________  | |
# | |    N    | | |frameResu    | | |
# | |    S    | | |   results   | | |
# | |         | | |_____________| | |
# | |frameL___| |frameR___________| |
# |_________________________________|
# 
# previous gui.py, I had a mongo dict which auto-filled gui elements (buttons, entry fields, etc) which was kinda gross. 
# Instead, each tab's layout is defined by an elements_* 2D matrix, which denotes what gui elements are where, on a grid
# Elements can span multiple rows or columns. 
# The code immediately below creates the framework (a tkinter Frame for the tabbing, and results plot/output)
# processElementLists() processes those 2D matrix grids and puts them onto the tabs
# Elements include: 
# Buttons: needs a label, needs a local function (which takes only an "event" argument) which can call into TDTR_fitting.py functions
# dropdown menus: needs a label, need to define the entries in the list, and what TDTR_fitting.py global they will write to
# Entry fields: single-line text/number entry. needs a label, need to define the TDTR_fitting.py global it writes to
# Text field: same as Entry, but multi-line is allowed. 
# Both Entry and Text can optionally be given a converter function (to go between strings, as displayed, and whatever data type TDTR_fitting.py expects). Two examples include: convert2um (just does scaling) and formatTP (borrowed from gui.py's l2s2D and s2l2D). 
# "enno" and "dropno" can also be used for no-label entry fields or drop-downs
# When a button is pressed, we FIRST load all globals from the current tab (into TDTR_fitting.py), then run the function specified, then grab the matplotlib objects from TDTR_fitting.py and display them. This is all done via the wrapper() function. 
# Data structures: TDTR_fitting.py has global variables (settable via setParam or setVar, or gettable via getParam or getVar) which control the execution of all TDTR_fitting functions. e.g. "rpu" sets the pump radius, "tofit" defines the parameters we're fitting for. gui fields (entry fields, text fields, drop-downs) may reference these globals, so we keep a record of which tkinter objects should be linked, on which tabs, via the globalLookup dict (maps global name to tkinter object). this makes it relatively easy to populate all fields from globals (updateAllFieldsFromGlobals function) or globals from fields updateAllGlobalsFromFields function). There may also be the need for local (to the GUI) variables. e.g. l_pertparams controls which parameters you want fed to perturb uncertainty. local variables are stored by name in the "localVars" dict. BEWARE! if you add a new local variable to an elements_* 2D matrix below, you also need to add it to localVars! why? this defines the default value. we don't want to try to guess what the default should be for new local variables. updateAllFieldsFromGlobals and updateAllGlobalsFromFields also work with the localVars dict (inferring between gui and TDTR_fitting variables based on the presence of the name in the dict). 

# 2D MATRIX DENOTES WHERE THINGS GO:
# "button;label;functionToRun"
# "drop;label;options;globalName;[optionalCustomFormatFunc]"
# "entry;label;globalName;[optionalCustomFormatFunc]"
# "text;label;globalName;[optionalCustomFormatFunc]"
# REPEAT ELEMENTS MEAN EXPAND OVER ELEMENTS

# proposed "standard" button layout
# [ IMPORT, FIT  , REFIT  , AVGFILES ]
# [ PERT  , FAST , 2D CONT, SENSTVTY ]
# |                                  |
# |        giant param entry         |
# |                                  |
# [ rpu  , rpr  , fm      , [gamma]/RMXY  ]
# [ fp   , pscor, tmin    , tnorm    ]


tpHeader="Props: C (J/m3/K) , Kz (W/m/K) , d (m) , Kr (W/m/K)"
elements_TDTR=[ 
 [  "btn;Import Vals;matImport"  ,    "btn;Fit Data;solving"     ,       "btn;refit;refit"       ,   "btn;avg files;avgFiles"    ],
 [  "btn;Perturb Unc.;pertUnc"   ,  "btn;Fast Contour;fastCont"  ,    "btn;2D Contour;cont2D"    ,   "btn;Sensitivity;runSens"   ],
 [  "btn;(phase);checkPhase"     ,               ""              ,              ""               ,              ""               ],
 [ "text;"+tpHeader+";tp;formatTP" ,             ""              ,              ""               ,              ""               ], 
 [              ""               ,               ""              ,              ""               ,              ""               ], 
 [              ""               ,               ""              ,              ""               ,"text;"+tpHeader+";tp;formatTP"],
 [     ""     ,  "entry;fitting params;tofit;formatParamNames"   ,  "entry;fitting params;tofit;formatParamNames"   ,     ""     ],
 [  "en;pump rad (um);rpu;conv2um" , "en;probe rad (um);rpr;conv2um" ,  "en;f mod (Hz);fm"       ,"drop;R/M/X/Y;R,M,X,Y;fitting" ],
 [  "en;fpulse (Hz);fp" , "drop;phase corr.;yes,no;doPhaseCorrect;ynbool" , "en;t min (s);minimum_fitting_time" , "en;t norm (s);time_normalize" ],
 [ "label;uncommon settings"     ,               ""              ,              ""               ,              ""               ], 
 [ "en;pert. params;l_pertparams",     "en;pert. by;l_pertby"    , "en;cont. val (%);l_contval"  , "en;cont. param;l_contparam"  ]]

elements_SSTR=[
 [  "btn;Import Vals;matImport"  ,    "btn;Fit Data;solving"     ,       "btn;refit;refit"       ,   "btn;avg files;avgFiles"    ],
 [  "btn;Perturb Unc.;pertUnc"   ,  "btn;Fast Contour;fastCont"  ,              ""               ,   "btn;Sensitivity;runSens"   ],
 [ "text;"+tpHeader+";tp;formatTP" ,             ""              ,              ""               ,              ""               ], 
 [              ""               ,               ""              ,              ""               ,              ""               ], 
 [              ""               ,               ""              ,              ""               , "text;"+tpHeader+";tp;formatTP" ],
 [            "" , "entry;fitting params;tofit;formatParamNames" , "entry;fitting params;tofit;formatParamNames" , ""            ],
 [  "en;pump rad (um);rpu;conv2um" , "en;probe rad (um);rpr;conv2um" , "en;f mod (Hz);fm"        ,         "en;gamma;gamma"      ],
 [ "en;pert. params;l_pertparams",     "en;pert. by;l_pertby"    , "en;cont. val (%);l_contval" ,               ""               ]]

elements_FDTR=[ 
 [  "btn;Import Vals;matImport"  ,    "btn;Fit Data;solving"     ,       "btn;refit;refit"       ,   "btn;avg files;avgFiles"    ],
 [  "btn;Perturb Unc.;pertUnc"   ,  "btn;Fast Contour;fastCont"  ,    "btn;2D Contour;cont2D"    ,   "btn;Sensitivity;runSens"   ],
 [ "text;"+tpHeader+";tp;formatTP" ,             ""              ,              ""               ,              ""               ], 
 [              ""               ,               ""              ,              ""               ,              ""               ], 
 [              ""               ,               ""              ,              ""               ,"text;"+tpHeader+";tp;formatTP"],
 [     ""     ,  "entry;fitting params;tofit;formatParamNames"   ,  "entry;fitting params;tofit;formatParamNames"   ,     ""     ],
 [  "en;pump rad (um);rpu;conv2um" , "en;probe rad (um);rpr;conv2um" ,  "en;f mod (Hz);fm"       ,"drop;P/M/X/Y;P,M,X,Y;fitting" ],
 [  "en;fpulse (Hz);l_fp" , "en;time delay (s);minimum_fitting_time","","" ],
 [ "label;uncommon settings"     ,               ""              ,              ""               ,              ""               ], 
 [ "en;pert. params;l_pertparams",     "en;pert. by;l_pertby"    , "en;cont. val (%);l_contval"  , "en;cont. param;l_contparam"  ]]

elements_PWA=[
 [  "btn;Import Vals;matImport"  ,    "btn;Fit Data;solving"     ,       "btn;refit;refit"       ,   "btn;avg files;avgFiles"    ],
 [  "btn;Perturb Unc.;pertUnc"   ,  "btn;Fast Contour;fastCont"  ,              ""               ,   "btn;Sensitivity;runSens"   ],
 [ "text;"+tpHeader+";tp;formatTP" ,             ""              ,              ""               ,              ""               ], 
 [              ""               ,               ""              ,              ""               ,              ""               ], 
 [              ""               ,               ""              ,              ""               , "text;"+tpHeader+";tp;formatTP" ],
 [            "" , "entry;fitting params;tofit;formatParamNames" , "entry;fitting params;tofit;formatParamNames" , ""            ],
 [  "en;pump rad (um);rpu;conv2um" , "en;probe rad (um);rpr;conv2um" , "en;f mod (Hz);fm"        ,         "en;gamma;gamma"      ],
 [ "en;pert. params;l_pertparams",     "en;pert. by;l_pertby"    , "en;cont. val (%);l_contval" ,               ""               ],
["drop;waveform;sine,square;waveformPWA",     "drop;norm;yes,no;normPWA;ynbool"   , "en;n sines;sumNPWA" ,               "en;duty cycle;dutyCycle"               ]]


mfinstruct="Set your thermal properties on the TDTR tab. For each line below, enter the experiment type from the dropdown, select a file, and enter any additional parameters in the remaining field, comma-separated. e.g. a multi-frequency experiment might have: fm=1000,gamma=2.4e4, and on a separate line: fm=10e6,gamma=2.8e4. Hybrid fitting of SSTR+TDTR might have: fm=8.4e6,rpu=10e-6,rpr=5e-6 on the TDTR line, and fm=1000,rpu=1.4e-6,rpr=1.6e-6 on the SSTR line"
# programmatically break up instructions into single-line labels on multiple rows
elements_multifitting=[] ; line=[] ; words=mfinstruct.split()
for i,word in enumerate(words):
	line.append(word)
	if i==len(words)-1 or len(" ".join(line+[words[i+1]]))>60:
		line=" ".join(line)
		elements_multifitting.append(["label;"+line,"","","label;"+line])
		line=[]
# add "header" buttons and labels
elements_multifitting=elements_multifitting+[
 [ "btn;Fit Data;simult"          , "btn;Clear fields;clearMultiFields" , "btn;hypothetical;hypothetical" , "" ],
 [ "btn;Perturb Unc.;pertUncSimult" , "btn;Fast Contour;fastContSimult" , "btn;Stack 2D Cont.;cont2DSimult" , "btn;Simult. 2D;cont3DFlatSimult" ],
 [ "" , "label;file name" , "label;meas. type" , "label; custom glos" ]]
# programatically add rows of file-select buttons and entry fields, TDTR/FDTR/SSTR dropdowns, global-setting entry fields
elements_multifitting=elements_multifitting+\
[ [ "btn;file "+str(i+1)+";aFTSB"+str(i+1) , 
	"enno;filename;l_simultFile"+str(i+1) , 
		"dropno;mode;TDTR,SSTR,FDTR,pFDTR,PWA;l_simultMode"+str(i+1) , 
			"enno;glos;l_simultGlos"+str(i+1) ] for i in range(10) ]
# localVar entries are added programmatically too (and duplicates of the addFileToSimultTab function) near where simult() is declared

#elements_multifitting=[
# [ mfinstruct,"","",""],["","","",""],["","","",mfinstruct]]

elements_other=[
 [ "btn;T(r,z);runTRZ" , "drop;T(r,z) mode;X,M,gen-gif,play-gif,T(t=0 z=0 r),T(t z=0 r=0),T(t z=0 irpr);l_Trzopt",  "en;Pu Power (W);Pow" , "drop;restore settings;yes,no;l_restore" ],
 [ "en;verbose funcs;verbose;format1DList","","","en;verbose funcs;verbose;format1DList" , ],
 [ "en;pu depth (m);depositAt", "en;pr depth (m);measureAt","drop;pu profile;gaussian,gaussian,gaussian_numerical,tophat,ring,ring_numerical,offset;pumpShape","en;pr offset (m);xoff"],
 [ "drop;auto rpr;yes,no;autorpr;ynbool","drop;auto rpu;yes,no;autorpu;ynbool","drop;autofm;yes,no;autofm;ynbool","drop;use TBR;yes,no;useTBR;ynbool"],
]

#buttonTitles=["Import Vals", "Fit Data", "Perturb Unc.", "Fast Contour", "Contours2D" ,"T(r,z)" ,"Sensitivity", "View Map", "(refit)", 
#	"(phase)" , "avg files", "fibercals"]
# buttonFuncs=[ runMatImport ,  runSolve ,  runPerturbing,  runContour   , runContour2D , runTRZ  , runSens     ,  viewMap  ,  refit   , 
#	checkPhase, avgFiles   , fibercals ]

#setVar("verbose",["solveTDTR"])

cellWidth=11 # 

def main():
	global window,frameL,frameR,framePlot,frameResu,tabControl,tabs,tabTitles,te_res
	# SETTING UP THE GUI AND PLACING THINGS:
	window=Tk() ; window.title("TDTR fitting!")
	# left vs right panels
	frameL=Frame(master=window) ; frameL.grid(row=0,column=0,sticky="NSEW")
	frameR=Frame(master=window) ; frameR.grid(row=0,column=1,sticky="NSEW")
	# 2:3 ratio of width for buttons vs plot panel
	window.columnconfigure(0,weight=1,uniform="window") ; window.columnconfigure(1,weight=2,uniform="window")
	window.rowconfigure(0,weight=1,uniform="window") # one row, set weight to allow it to expand with the window

	# top vs bottom panels on the right
	framePlot=Frame(master=frameR) ; framePlot.grid(row=0,column=0,sticky="NSEW")
	frameResu=Frame(master=frameR) ; frameResu.grid(row=1,column=0,sticky="NSEW")
	# 3:1 ratio of height for plot vs results panel
	frameR.rowconfigure(0,weight=3,uniform="frameR") ; frameR.rowconfigure(1,weight=1,uniform="frameR")
	frameR.columnconfigure(0, weight=1,uniform="frameR") # one column, set weight to allow it to expand with the window

	# tabs in the buttons panel
	tabControl = ttk.Notebook(frameL) ; tabs={} ; tabTitles=["TDTR","SSTR","FDTR","PWA","multifitting","other"]
	# results panel
	lb_res=tk.Label(master=frameResu,text="RESULTS:")
	te_res=tk.Text(master=frameResu,height=7)				# text entry field object
	lb_res.grid(row=0,column=0,sticky="EW") ; te_res.grid(row=1,column=0,sticky="EW")			# add both objects to the window
	frameResu.columnconfigure(0,weight=1,uniform="window")


	for tabTitle in tabTitles:
		tabs[tabTitle]=ttk.Frame(tabControl)
		tabControl.add(tabs[tabTitle],text=tabTitle)
		tabControl.pack(expand=1, fill="both")


	# UNCOMMENT THESE TO DRAW COLOR-CODED BORDERS AROUND EACH FRAME (EG, TO CHECK THAT GRID ELEMENTS EXPAND APPROPRIATELY)
	#colors=["red","orange","yellow","green","blue","purple","black"]*10
	#for i,frame in enumerate([window,frameL,frameR,framePlot,frameResu]):
	#		frame.configure(highlightbackground=colors[i],highlightthickness=10)

	# even if a tab is currently empty, pass it an empty list (which populates globals)
	processElementLists(tabTitles,[elements_TDTR,elements_SSTR,elements_FDTR,elements_PWA,elements_multifitting,elements_other]) 
	resume()

	window.protocol("WM_DELETE_WINDOW", quit_me)

	window.mainloop()



# https://stackoverflow.com/questions/13079299/dynamically-adding-methods-to-a-class
# once you create a tkinter Text object, add this via setattr(obj, 'set', TextSetter)
#def TextSetter(self,text):
#	print("self",self)
#	self.delete('1.0', END)
#	self.insert(tk.END,text)
def createSetterGetter(tktextobj):
	def setter(text):
		tktextobj.delete('1.0', END)
		tktextobj.insert(tk.END,text)
	def getter():
		# https://github.com/python/cpython/blob/3.13/Lib/tkinter/__init__.py
		return tktextobj.tk.call(tktextobj._w, 'get', '1.0', END)
	tktextobj.set=setter
	tktextobj.get=getter

# THIS PLACES BUTTONS/DROPDOWNS/ENTRY FIELDS BASED ON 2D LISTS (GRIDS) SUCH AS elements_TDTR
def processElementLists(tabTitles,elementLists):
	# Put each tab's grid elements on each tab
	for tabKey,elementsList in zip(tabTitles,elementLists):
		processedElements=[]	# don't repeat elements (if elements span multiple rows or columns, and are thus entered twice)
		master=tabs[tabKey]	# all objects will be added to this tab
		globalLookup[tabKey]={}	# this links global names (from TDTR_fitting) to tk objects
		globalSpecialHandling[tabKey]={}
		if len(elementsList)==0:
			continue
		nrows,ncols=len(elementsList),len(elementsList[0])
		for row in range(nrows):
			for col in range(ncols):
				master.columnconfigure(col,weight=1,uniform=str(master))	# ensure grid elements can stretch horizontally! 
				elementString=elementsList[row][col]
				if len(elementString)==0 or elementString in processedElements:
					continue
				# check all other rows/cols to check if it spans multiple!
				rowspan=1 ; columnspan=1
				for r in range(nrows):
					for c in range(ncols):
						if elementsList[r][c]==elementString:
							rowspan=max(rowspan,abs(r-row)+1)# we're at row=1, we found r=3, we should span 3 rows (1,2,3)
							columnspan=max(columnspan,abs(c-col)+1)
				# process each element!
				elementType,elementLabel=elementString.split(";")[:2]
				# GRIDDING: each element actually covers two rows (label, with entry field below it, for example). start with row*2 here...
				packkwargs={"row":row*2,"column":col,"rowspan":rowspan*2,"columnspan":columnspan,"sticky":"EW"}
				if elementType=="label":
					label=tk.Label(master=master,text=elementLabel,width=cellWidth)
					label.grid(**packkwargs)# label goes on first row (even if entry field spans multiple)

				if elementType in [ "button", "btn" ]:
					button=tk.Button(master=master,text=elementLabel,width=cellWidth)# button object
					funcName=elementString.split(";")[2]
					func=globals()[funcName]
					button.bind("<Button-1>", func)			# link the dummy function to the button
					button.grid(**packkwargs)
				if elementType in [ "drop", "entry", "text", "en" ]:
					label=tk.Label(master=master,text=elementLabel,width=cellWidth)
					packkwargs["rowspan"]=1 ; label.grid(**packkwargs)# label goes on first row (even if entry field spans multiple)
					packkwargs["rowspan"]=rowspan*2-1 ; packkwargs["row"]+=1# prepare to put drop or entry objects on next line
				if elementType in [ "drop", "dropno" ]:
					options=elementString.split(";")[2].split(",")
					val=options[0]
					opt=tk.StringVar(window)			# selected option is stored in a stringVar object
					dropdown = tk.OptionMenu(master, opt, *options)	# dropdown menu object
					dropdown.grid(**packkwargs)
					dropdown.config(width=cellWidth-3)
					globalName=elementString.split(";")[3]
					specialFormatHandler=""
					if len(elementString.split(";"))==5:
						specialFormatHandler=elementString.split(";")[4]
				if elementType in [ "entry", "en" , "enno" ]:
					opt=tk.StringVar(window)			# value for entry field goes in a stringVar object
					field=tk.Entry(master=master,textvariable=opt,width=cellWidth)	# text entry field object
					field.grid(**packkwargs)
					elementString.split(";")[2]
					globalName=elementString.split(";")[2]
					specialFormatHandler=""
					if len(elementString.split(";"))==4:
						specialFormatHandler=elementString.split(";")[3]
				if elementType=="text":
					opt=tk.Text(master=master,height=rowspan*2-1,width=cellWidth*columnspan)
					opt.grid(**packkwargs)
					# https://stackoverflow.com/questions/13079299/dynamically-adding-methods-to-a-class
					createSetterGetter(opt)
					elementString.split(";")[2]
					globalName=elementString.split(";")[2]
					specialFormatHandler=""
					if len(elementString.split(";"))==4:
						specialFormatHandler=elementString.split(";")[3]
				if elementType in [ "drop", "entry", "text", "en" , "dropno", "enno" ]:
					#if globalName not in localVars.keys():
					globalLookup[tabKey][globalName]=opt	# this links global names (from TDTR_fitting) to tk objects
					if len(specialFormatHandler)>1:
						globalSpecialHandling[tabKey][globalName]=specialFormatHandler

				#print(elementString,packkwargs)
				processedElements.append(elementString)
	updateAllFieldsFromGlobals()

# FORMATTING FUNCTIONS, FOR PROCESSING DATA BETWEEN TDTR_fitting.py GLOBALS AND GUI ENTRY/DROPDOWN/ETC OBJECTS
#def formatTP(tp,whichWay="format"):
#	print("tp",whichWay,tp)
#	if whichWay=="format":
#		return "\n".join( [ ",".join([ str(v) for v in row ]) for row in tp ] )
#	rows=tp.split("\n") ; filtered=[]
#	for i,row in enumerate(rows):
#		vals=row.split(",") ; row=[]
#		for v in vals:
#			if "k" in v.lower():
#				row.append(v)
#			elif len(v)>0:
#				row.append(float(v))
#		if len(row)>0:
#			filtered.append(row)
#	return filtered
def formatTP(tp,whichWay="format"):
	if whichWay=="format": # [[a,b,c],[d,e,f]] -> "a,b,c;d,e,f"
		list2D=[[str(v) for v in row] for row in tp] # each element becomes a string
		list2D=[",".join(row) for row in list2D] # joint each row by ","
		return "\n".join(list2D) # join rows together by ";"
	else:
		list2D=tp.split("\n") # split rows into entries in a list
		list2D=[ row.split("#")[0] for row in list2D ] # strip comments off each row
		list2D=[ row.split(",") for row in list2D  if len(row)>0 ] # purge blank lines
		list2D=[[v.strip() for v in row] for row in list2D] # purge whitespace
		list2D=[["Kz" if v in ["Kz","kz","KZ"] else eval(v) for v in row] for row in list2D] # tp mayn't contain strings! 
		return list2D
def format1DList(val,whichWay="format"):
	if whichWay=="format":
		return ",".join( [ str(v) for v in val ] )
	else:
		return [ v.strip() for v in val.split(",") ]

def ynbool(val,whichWay="format"):
	if whichWay=="format":
		if val:
			return "yes"
		else:
			return "no"
	else:
		return (val=="yes")

def conv2um(val,whichWay="format"):
	if whichWay=="format":
		return str(np.round(val*1e6,2))
	else:
		return float(val)*1e-6
def formatParamNames(val,whichWay="format"):
	print("formatParamNames",val,whichWay)
	if whichWay=="format":
		return ",".join(val).replace(" ","")
	else:
		vals=val.replace(" ","").split(",")
		return [ v for v in vals if len(v)>1 ]

# this links global names (from TDTR_fitting) to tk objects
globalLookup={}			# tabName={tdtrGloName:guiEntryOrDropdownObj}
globalSpecialHandling={}	# tabName={tdtrGloName:customConverterFunc}
localVars={"l_pertparams":"all","l_pertby":"5","l_contval":"2.5","l_contparam":"Kz2","l_restore":"yes","l_Trzopt":"M","l_fp":"CW"}	# localVarName:val
def updateAllFieldsFromGlobals():
	for tab in globalLookup.keys():
		for glo in globalLookup[tab].keys():				# e.g. "rpu" as a string
			if glo in localVars.keys():
				val=localVars[glo]				# retrieve value from either the local dict
			else:
				val=getVar(glo)					# or from TDTR_fitting, e.g. 5e-6 (as a float)
			if glo in globalSpecialHandling[tab].keys():		# e.g. "rpu" linked to special function "convert2um"
				fun=globals()[globalSpecialHandling[tab][glo]]	# lookup func from string name for func
				val=fun(val,"format")				# pass value to special function
			else:
				val=str(val)					# OR, just convert to string if no special function
			globalLookup[tab][glo].set(val)				# and set the corrosponding gui entry object
			conditionalPrint("updateAllFieldsFromGlobals","RETREIVE TDTRFITTNG GLO "+str(tab)+" "+str(glo)+" "+str(val)+" "+str(type(val)))

def updateAllGlobalsFromFields(tab):
	global localVars
	for glo in globalLookup[tab].keys():					# e.g. "rpu" as a string
		val=globalLookup[tab][glo].get()				# get gui entry object's value (will be a string)
		if glo in globalSpecialHandling[tab].keys():			# e.g. "rpu" linked special function "convert2um"...
			fun=globals()[globalSpecialHandling[tab][glo]]		# ...which should have unformat option to go back (scale, unscale, etc)
			val=fun(val,"unformat")
		else:
			if glo in localVars.keys():				# OR, retrieve existing value (locals)
				oldval=localVars[glo]
			else:
				oldval=getVar(glo)				# OR, retrieve existing value (TDTR globals)
			if type(oldval)==float: 				# e.g. pump depth "1e-3+80e-9" ought to work
				val=eval(val)
			val=type(oldval)(val)					# and use it's type to convert from string back to the correct type
		if glo in localVars.keys():
			localVars[glo]=val					# and update local variable, or globals, with value
		else:
			conditionalPrint("updateAllGlobalsFromFields","setVar: "+str(glo)+" "+str(val))
			setVar(glo,val)
		conditionalPrint("updateAllGlobalsFromFields","SET GLO FROM FIELD '"+str(tab)+"' '"+str(glo)+"' '"+str(val)+"' "+str(type(val)))
		#print("SET GLO FROM FIELD","'"+str(tab)+"'","'"+str(glo)+"'","'"+str(val)+"'",type(val))

def writeSettingsToLog(tabNames):
	# save off settings: wrapper() writes "settings: " with localVars and pulled-in values for glos in globalLookup. resume() read in these dicts (as text) and restores them! special processing is required for 1D lists (e.g. "verbose") and 2D lists (e.g. "tp")
	settings={}
	for tab in tabNames:
		for var in globalLookup[tab].keys():
			if "l_" in var:
				val=localVars[var]
			else:
				val=getVar(var)
			settings[var]=val
	writeToLogFile("settings: "+" ; ".join([str(k)+":"+str(settings[k]).strip() for k in settings.keys() ]))

# read gui.log file, update TDTR_fitting.py globals, and localVars. updateAllFieldsFromGlobals can then update the fields
def resume():
	global files,localVars
	if not os.path.exists("gui.log"):
		return
	lines=open("gui.log").readlines() ; foundtp=False ; foundfiles=False
	for i in reversed(range(len(lines))):
		print(i,lines[i])
		# wrapper() writes "settings: " with localVars and pulled-in values for glos in globalLookup. resume() read in these dicts (as text) and restores them! special processing is required for 1D lists (e.g. "verbose") and 2D lists (e.g. "tp")
		if ( not foundtp ) and len(lines[i])>10 and lines[i][:10]=="settings: ":
			print("FOUND SETTINGS LINE")
			settings={}
			line=lines[i].replace("settings: ","").strip().split(" ; ")	# settings: tofit:['Kz2', 'R1'] ; rpu:9.9e-06 ; rpr:4.9e-06 ; fm:8.4e6...
			for gloval in line:					# [ "tofit:['Kz2', 'R1']" , "rpu:9.9e-06" , "rpr:4.9e-06" , "fm:8.4e6"...
				glo,val=gloval.split(":")			# "tofit", "['Kz2', 'R1']"
				if "[[" in val:					# SPECIAL HANDLING: 2D LIST
					newval=[] ; rows=val.split("], [")
					for row in rows:
						newval.append([]) ; row=row.split(",")
						for v in row:
							v=v.replace("]","").replace("[","").strip()
							if "k" in v.lower() or "_" in v.lower():
								newval[-1].append(v.replace("'",""))
							else:
								newval[-1].append(float(v))
					val=newval
				elif "[" in val:				# SPECIAL HANDLING: 1D LIST
					val=val.replace("[","").replace("]","").replace("'","").split(",")
				else:
					if glo in localVars.keys():			# 
						oldval=localVars[glo]			# retrieve existing value (locals)
					else:
						oldval=getVar(glo)			# OR, retrieve existing value (TDTR globals)
					print("flag",glo,val,"( was:",oldval,")")
					if isinstance(oldval,bool): # fun! the code: bool("False") returns True! since it treats "False" as just a string, which is treated as True (unless empty)
						val=( "True" in val )
					val=type(oldval)(val)
				settings[glo]=val
			foundtp=True
			if settings["l_restore"]=="no":
				continue
			for glo in settings.keys():
				print("glo",glo,settings[glo],type(settings[glo]))
				if "l_" in glo:
					print("SET FROM LOG: localVars",glo,"=",settings[glo])
					localVars[glo]=settings[glo]
				else:
					print("SET FROM LOG: setVar",glo,settings[glo])
					setVar(glo,settings[glo])
			#if "'l_restore': 'yes'" not in lines[i]:
			#	return
			#rows=lines[i].replace("settings: ","").split(";")[0].split("], [")
			#tp=[]
			#for row in rows:
			#	row=row.strip().replace("[","").replace("]","")
			#	row=[ v.replace("'","") if "k" in v.lower() else float(v) for v in row.split(",") ]
			#	tp.append(row)
			#	print(row)
			#	#tp.append(
			#tp=""
			#print(rows)
			#tp=[] ; Cs=[] ; Kzs=[] ; Krs=[] ; Gs=[] ; ds=[]
			#for n in range(10):
			#	print(lines[i+n])
			#	if "Cs: [" in lines[i+n]:
			#		Cs=[ float(v) for v in lines[i+n].split("[")[-1].split("]")[0].split(",") ]
			#	if "Kzs: [" in lines[i+n]:
			#		Kzs=[ float(v) for v in lines[i+n].split("[")[-1].split("]")[0].split(",") ]
			#	if "ds: [" in lines[i+n]:
			#		ds=[ float(v) for v in lines[i+n].split("[")[-1].split("]")[0].split(",") ]
			#	if "Krs: [" in lines[i+n]:
			#		Krs=[ float(v) for v in lines[i+n].split("[")[-1].split("]")[0].split(",") ]
			#	if "Gs: [" in lines[i+n]:
			#		Gs=[ float(v) for v in lines[i+n].split("[")[-1].split("]")[0].split(",") ]
			#	if len(Cs)>0 and len(Kzs)>0 and len(Krs)>0 and len(Gs)>0 and len(ds)>0:
			#		tp=[]
			#		for j in range(len(Cs)):
			#			tp.append([Cs[j],Kzs[j],ds[j],Krs[j]])
			#			if j<len(Cs)-1:
			#				RG={True:1/Gs[j],False:Gs[j]}[getVar("useTBR")]
			#				tp.append([RG])
			#		print("setVar","tp",tp)
			#		setVar("tp",tp)
			#		break
			#print("setVar","tp",tp)
			#setVar("tp",tp)
			#foundtp=True
			#break
		if ( not foundfiles) and len(lines[i])>10 and lines[i][:7]=="files:[":
			files=lines[i].replace("'","").split("[")[1].split("]")[0].split(",")
			files=[ f.strip() for f in files ] # strip required or spaces in printed list mess us up
			print("FOUND FILES LINE:",files)
			foundfiles=True
		if foundtp and foundfiles:
			break
	updateAllFieldsFromGlobals()

# WRAPPER FUNCTIONS FOR TDTR_fitting.py FUNCTIONS.

# each function also get it's own wrapper. when any button is pressed, before calling anything, we pass TDTR_fitting various text-enterable values, and after, we do some logging and plotting
def wrapper(func): # https://www.geeksforgeeks.org/function-wrappers-in-python/
	def wrapped(*args,**kwargs):
		tabIndex = tabControl.index("current")
		tabName=tabTitles[tabIndex] # print("tabName",tabName)
		# grab all globals from all OTHER tabs first, then from this tab (this tab overrides others, but settings on other tabs might matter). also reverse sort order (assume first tab, farthest left, takes precedence). TODO might be a way to track which tab has been visited last? e.g. on "other" tab we do not have a thermal properties matrix, so where should the user set that? TDTR tab? SSTR tab? imo "first tab" makes most sense but "last tab i visited" maybe makes even more sense
		tabsSorted=[ t for t in reversed(tabTitles) if t!=tabName ] ; tabsSorted.append(tabName)
		for tab in tabsSorted:
			updateAllGlobalsFromFields(tab)
		# update mode
		if tabName in ["TDTR","FDTR","SSTR","PWA"]:
			setParam("mode",tabName)
		if tabName=="FDTR":
			if localVars["l_fp"].strip().lower()!="cw":
				setParam("mode","pFDTR")
				setVar("fp",float(localVars["l_fp"]))
		# save off current state
		writeSettingsToLog(tabsSorted)
		writeToLogFile("running:"+str(func))
		window.title("TDTR fitting! - RUNNING") ; window.update()
		printtp()
		#window.configure(highlightbackground="blue",highlightthickness=10) ; window.update()
		try:
			func(*args,**kwargs)
			#window.configure(highlightbackground="green",highlightthickness=10) ; window.update()
		except Exception:
			window.title("TDTR fitting! - ERRORED")
			#window.configure(highlightbackground="red",highlightthickness=10) ; window.update()
			e=traceback.format_exc()
			printToResultsPanel("ERROR WITH FUNC:"+str(func)+", please send your gui.log file to the developer. Windows: log file can be found in the same folder as the executable. MacOS: log file can be found in your \"home\" folder.")
			printToResultsPanel(str(e))
			writeToLogFile("[FAILURE] : \n")						# and log that to file
			writeToLogFile(str(e))

		if "quit_me" in str(func):
			return
		window.title("TDTR fitting! - SUCCESS")
		updatePlot(str(func))
	return wrapped

lastDirec="./"
def ask(multiple=True,fileOrDirec="file",text=""):
	global lastDirec
	# ask user for files:
	window.update() # Trying this to see if it gets rid of hanging issues on mac: https://stackoverflow.com/questions/21866537/what-could-cause-an-open-file-dialog-window-in-tkinter-python-to-be-really-slow
	if fileOrDirec=="file":
		if multiple:
			text={True:text,False:"Open"}[len(text)>0]
			selected=list(tk.filedialog.askopenfilenames(initialdir=lastDirec,title=text))
		else:
			text={True:text,False:"Open"}[len(text)>0]
			selected=[tk.filedialog.askopenfilename(initialdir=lastDirec,title=text)]
		lastDirec="/".join(selected[-1].split("/")[:-1])
	if fileOrDirec=="direc":
		text={True:text,False:"Choose Directory"}[len(text)>0]
		selected=tk.filedialog.askdirectory(initialdir=lastDirec,title=text)
		lastDirec=selected
	return selected

@wrapper
def matImport(event): # was "runMatImport"
	files=ask(multiple=False)
	if len(files)==0:
		return
	importMatrix(str(files[0]))	# load the thermal properties
	updateAllFieldsFromGlobals()	# update them (and others) in the GUI

files=[] ; lastResult=[] #; lastRun=""
@wrapper
def solving(event,askForFiles=True): # was "runSolve"
	global files,lastResult #,lastRun ; lastRun="solve"
	if askForFiles:
		files=ask()
	if len(files)==0:
		return
	results=[]
	writeToLogFile("files:"+str(files))
	# TDTR_fitting.py > solve() > solveTDTR() > resultsPlotter() > if "gui" in stack, useLast=True > lplot() > if useLast, append new datasets to globals plotXs, plotYs, etc
	for k in ["plotXs","plotYs","plotLabels","plotMarkers"]: # TODO WE DIDN'T DO THESE SHENAIGANS IN THE OLD GUI. WHY DO WE NEED IT NOW? (currently needed to prevent adding curves to an old plot
		setVar(k,[]) 
	for f in files:
		res,err=solve(f,plotting="save")
		#print(res,err)
		results.append(res)
		resultString=" , ".join( [p+"="+sigFigs(v,4) for p,v in zip(getVar("tofit"),res) ] )+" , "+sigFigs(err[0]*100)+"%"
		resultString+=" , "+f.split("/")[-1][:77-len(resultString)]
		printToResultsPanel(resultString)
		lastResult=res
		#updatePlot("solve")
	results=np.asarray(results)
	res,err=np.mean(results,axis=0),np.std(results,axis=0)
	resultStrings=[p+" = "+sigFigs(v*getScaleUnits(p)[0],4)+"+/-"+sigFigs(dv)+" "+getScaleUnits(p)[1] for p,v,dv in zip(getVar("tofit"),res,err) ]
	printToResultsPanel("averaged +/- std:\n"+" , ".join(resultStrings))

@wrapper
def refit(event):
	solving(event,askForFiles=False)

@wrapper
def avgFiles(event):
	files=ask() ; ftypes={"SSTR":"fSSTR","TDTR":"TDTR"} ; ftype=ftypes.get(getVar("mode"),"raw")
	fo,ig=fileAverager(files,fileType=ftype)
	printToResultsPanel("files averaged, and outputted to: "+fo)

@wrapper
def pertUnc(event): # was "runPerturbing"
	perturb=localVars["l_pertparams"] 
	perturbBy=localVars["l_pertby"]
	if perturb=="all":
		perturb=""
	else:
		perturb=perturb.split(",")
	perturbBy=perturbBy.split(",") ; perturbBy=[float(pb) for pb in perturbBy]

	#if lastRun=="simult":
	#	solveFunc={"func":ss2,"kwargs":{"listOfTypes":simultTypes}} ; loopOver=[filesSimult]
	#else:
	solveFunc={"func":solve,"kwargs":{}} ; loopOver=files
	r,e=[],[]
	for f in loopOver:
		s,u,params=perturbUncertainty(f,paramsToPerturb=perturb,perturbBy=perturbBy,plotting="save",solveFunc=solveFunc) #,paramsToPerturb=paramsToPerturb,perturbBy=perturbBy)
		#print("s,u,params",s,u,params)
		for P,dP,dR in params:
			resultString="perturb "+P+" by "+str(dP)+"% --> "+",".join( ["d"+p+"="+sigFigs(v,4) for p,v in zip(getVar("tofit"),dR) ] )
			printToResultsPanel(resultString)
	#		print(P,dP,dR)
		resultString=" , ".join([p+" = "+sigFigs(v*getScaleUnits(p)[0])+"+/-"+sigFigs(dv*getScaleUnits(p)[0])+" "+getScaleUnits(p)[1] for p,v,dv in zip(getVar("tofit"),s,u) ])
		printToResultsPanel(resultString)
		r.append(s) ; e.append(u)

	if len(files)>1:
		r=np.mean(r,axis=0) ; e=np.mean(e,axis=0)
		resultString=" , ".join([p+" = "+sigFigs(v*getScaleUnits(p)[0])+"+/-"+sigFigs(dv)+" "+getScaleUnits(p)[1] for p,v,dv in zip(getVar("tofit"),r,e) ])
		printToResultsPanel("averaged:")
		printToResultsPanel(resultString)

@wrapper
def pertUncSimult(event): # was "runPerturbing"
	perturb=localVars["l_pertparams"] 
	perturbBy=localVars["l_pertby"]
	if perturb=="all":
		perturb=""
	else:
		perturb=perturb.split(",")
	perturbBy=perturbBy.split(",") ; perturbBy=[float(pb) for pb in perturbBy]

	files,settables=processAllMultiFields(exitOn="globals")
	# For solving, we would run: r,e=ss2(files,types,plotting="save",settables=settables), so these kwargs need to be set up
	kwargs={"listOfTypes":settables["mode"],"settables":settables}
	solveFunc={"func":ss2,"kwargs":kwargs}
	r,e=[],[]
	s,u,params=perturbUncertainty(files,paramsToPerturb=perturb,perturbBy=perturbBy,plotting="save",solveFunc=solveFunc) #,paramsToPerturb=paramsToPerturb,perturbBy=perturbBy)
	#print("s,u,params",s,u,params)
	for P,dP,dR in params:
		resultString="perturb "+P+" by "+str(dP)+"% --> "+",".join( ["d"+p+"="+sigFigs(v,4) for p,v in zip(getVar("tofit"),dR) ] )
		printToResultsPanel(resultString)
#		print(P,dP,dR)
	resultString=" , ".join([p+" = "+sigFigs(v*getScaleUnits(p)[0])+"+/-"+sigFigs(dv*getScaleUnits(p)[0])+" "+getScaleUnits(p)[1] for p,v,dv in zip(getVar("tofit"),s,u) ])
	printToResultsPanel(resultString)
	r.append(s) ; e.append(u)

def reloadedMessage(reloaded):
	if reloaded:
		printToResultsPanel("warning: results were reloaded from cache. delete folder \"__gui.py\" to regenerate")


@wrapper
def fastCont(event): # was "runContour"
	fs=files ; solvefunc={"func":solve,"kwargs":{}}
	for f in fs:
		p=localVars["l_contparam"]
		if getVar("mode")=="SSTR":
			p=getVar("tofit")[0]
		thresh=float(localVars["l_contval"])
		bnds,fout,reloaded=measureContour1Axis(f,paramOfInterest=p,plotting="savefinal",resolution=100,threshold=thresh/100,solveFunc=solvefunc)
		error=(bnds[1]-bnds[0])/2 ; errorp=(bnds[1]-bnds[0])/(bnds[1]+bnds[0])
		#out(p+" : "+str(bnds)+" : +/- "+str(error))
		fact,unit=getScaleUnits(p)
		printToResultsPanel(scientificNotation(bnds[0]*fact,2)+" <= "+p+" <= "+scientificNotation(bnds[1]*fact,2)+" "+unit+
			" (+/-"+scientificNotation(error*fact,2)+" "+unit+" or "+
			"+/-"+str(np.round(errorp*100,1))+"%)")
		#reloadedMessage(reloaded)

@wrapper
def fastContSimult(event): # was "runContour"
	files,settables=processAllMultiFields(exitOn="files")
	solvefunc={"func":ss2,"kwargs":{"settables":settables}} #; setVar("ss2Types",ss2t)

	p=localVars["l_contparam"]
	thresh=float(localVars["l_contval"])

	bnds,fout,reloaded=measureContour1Axis(files,paramOfInterest=p,plotting="savefinal",resolution=100,threshold=thresh/100,solveFunc=solvefunc)

	error=(bnds[1]-bnds[0])/2 ; errorp=(bnds[1]-bnds[0])/(bnds[1]+bnds[0])
	#out(p+" : "+str(bnds)+" : +/- "+str(error))
	fact,unit=getScaleUnits(p)
	printToResultsPanel(scientificNotation(bnds[0]*fact,2)+" <= "+p+" <= "+scientificNotation(bnds[1]*fact,2)+" "+unit+
		" (+/-"+scientificNotation(error*fact,2)+" "+unit+" or "+
		"+/-"+str(np.round(errorp*100,1))+"%)")
	#reloadedMessage(reloaded)

@wrapper
def cont2D(event): # was "runContour2D"
	pr=[[v*.1,v*2] for v in lastResult ]
	D="2D"
	thresh=float(localVars["l_contval"])/100
	fileOut,reloaded=genContour2D(files[0],paramRanges=pr)
	globstr=files[0].split("/")[:-1] + ["gui.py_","contours","*.txt"] ; globstr="/".join(globstr)
	displayContour2D(fileOut,plotting="save",threshold=thresh)
	#reloadedMessage(reloaded)

# THERE ARE TWO BEHAVIORS TO EXPECTR FROM genContour2D:
# 1) each file can be treated individually, "solve()" is run for each file to get the residual at each point for that file. this generates N contour plots for N files. If you have 2 fitting parameters, the candidate area is simply the overlap between them.
# 2) multiple files can be solved simultaneously (calling "ss2()"), to get the worst residual across the files. this generates a single contour plot. If you have 2 fitting parameters, the canndidate area should match the overlapped for methid (1). 
# if you have more than 2 fitting parameters, we iterate through "all combinations" of the first two, then  solve for the remaining. If you have 3 parameters, this is effectivelly constructing a 3D contour volume, then projecting it down (or flattening it) across the 3rd axis
# BEWARE: for >2 fitting parameters, (1) yields each file's contour volume projected (if each contour is huge, the projection is huge) whereas (2) yields *only the intersection of the contour volumes* projected. i.e., you will get the projection of the Boolean Union vs Boolean Interesection. YOU SHOULD NOT USE (1) IF YOU HAVE MORE THAN TWO PARAMETERS. 
# EACH FILE CAN BE TR
@wrapper
def cont2DSimult(event): # was "runContour2D"
	pr=[[v*.1,v*3] for v in lastResult ]
	#print("filesSimult",filesSimult)
	files,settables=processAllMultiFields(exitOn="files") # e.g. {"mode":[...],"d2":[...],"fitting":[...]}
	D="2D"
	thresh=float(localVars["l_contval"])/100
	# genContour2D can take solveFunc={ "func" : solve or ss2, "kwargs" : {dict of kwargs} } and ALSO takes settables=
	fileOut,reloaded=genContour2D(files,paramRanges=pr,settables=settables) # generateHeatmap accepts a LIST of files, which it just loops through
	displayContour2D(fileOut,plotting="save",threshold=thresh) # list -> generateHeatmap -> list -> displayHeatmap also accepts a list (and 
	#reloadedMessage(reloaded)


@wrapper
def cont3DFlatSimult(event): # was "runContour2D"
	pr=[[v*.1,v*3] for v in lastResult ]
	#print("filesSimult",filesSimult)
	files,settables=processAllMultiFields(exitOn="files") # e.g. {"mode":[...],"d2":[...],"fitting":[...]}
	D="2D"
	thresh=float(localVars["l_contval"])/100
	# genContour2D can take solveFunc={ "func" : solve or ss2, "kwargs" : {dict of kwargs} } and ALSO takes settables=
	fileOut,reloaded=genContour2D(files,paramRanges=pr,solveFunc={"func":ss2,"kwargs":{"settables":settables}}) # generateHeatmap accepts a LIST of files, which it just loops through
	displayContour2D(fileOut,plotting="save",threshold=thresh) # list -> generateHeatmap -> list -> displayHeatmap also accepts a list (and 
	#reloadedMessage(reloaded)


@wrapper
def runSens(event):
	sensitivity()

@wrapper
def checkPhase(event):
	if len(files)==0:
		out("run fitting first")
		return
	ts,data=readTDTR(files[-1],plotPhase=True)

# FUNCTION FOR BUTTON WHICH GENERATES T(r,z) PLOT
@wrapper
def runTRZ(event):
	maxrad={False:getParam("rpu"),True:max(getVar("xoff"),getParam("rpu"),getParam("rpr"))}["offset" in getVar("pumpShape")]*1.5
	print("maxrad",maxrad,getVar("pumpShape"),getVar("xoff"),getParam("rpu"),getParam("rpr"))
	#maxrad=getParam("rpu")*1.5
	nrz=50 ; nt=1000 ; npics=250
	omegas=np.asarray([0.001,getParam("fm")*2*pi]) # TRUE TEMPERATURE RISE IS SUM OF SS + MODULATED

	TRZopt=localVars["l_Trzopt"]

	# Options include: X;M;gen-gif;play-gif;T(r,z=0,t=0);T(rpr,z=0,t)
	if TRZopt in ["X","M"]:
		T,d,r,Ts=Tz(rsteps=nrz,dsteps=nrz,maxdepth=2*1.5e-6,maxradius=maxrad,full=True,omegas=omegas) # Tz returns T[d,r],depths[d],radii[r]
		Ts=np.sum(Ts,axis=0)
		T={ "X":Ts.real , "M":np.sqrt(Ts.real**2+Ts.imag**2) }[ TRZopt ]
		#T,dT=melt(T,75,10e6,3e6)
		showTrzs(T,d,r,includeTPD=True,savefile="gui_Trz.txt") #,bonusContours=[Tmelt,Tmelt+dT])
	elif TRZopt in ["gen-gif","T(t 0,z 0,r)","T(t,z 0,r 0)","T(t,z 0,irpr)"]:
		T,t,d,r=Ttzr(mindepth=0,maxdepth=1500e-9*2,dsteps=nrz,rsteps=nrz,maxradius=maxrad,tsteps=nt)
		#T=np.asarray( [ melt(Ts,50,100e6,3e6)[0] for Ts in T ] ) #; print(T,np.shape(T))
		T=np.asarray( [ melt(Ts,50,100e6,3e6)[0] for Ts in T ] ) #; print(T,np.shape(T))
		Tmax=max([np.amax(Ts) for Ts in T]) ; Tmin=min([np.amin(Ts) for Ts in T])
		if TRZopt=="gen-gif":
			for i in range(0,npics):
				showTrzs(T[i*int(nt/npics)],d,r,includeTPD=False,plotting="saveguigif/gui"+str(i)+".png")#cbounds=[Tmin,Tmax,11])#,bonusContours=[50-.1,50+.1])
				#img.config(file="guigif/gui"+str(i)+".png")
				updatePlot("runTRZ")
				frameR.update()
			printToResultsPanel("find your frames in folder \"guigif\"...")
		elif TRZopt=="T(t 0,z 0,r)":
			lplot([r*1e6], [T[0,0,:]], xlabel="radius (μm)", ylabel="T (K)", title="T(t=0,z=0,r)", labels=[""], markers=["k-"])#,forcedBoundsX=[0,14])
			updatePlot("runTRZ")
			frameR.update()
		elif TRZopt=="T(t,z 0,r 0)":
			T=T[:,0,0] #; T=np.roll(T,int(len(T)/2))
			lplot([t*1e6],[T],"time (μs)","T (K)","T(t,z=0,r=0)",datalabels=[""],markers=["k-"])#,forcedBoundsX=[0,max(t*1e6)],forcedBoundsY=[0,None])
		elif TRZopt=="T(t,z 0,irpr)":
			Tr=T[:,0,:] ; probeweight=2/np.pi/getParam("rpr")*np.exp(-2*r**2/getParam("rpr")**2)
			Tr=integrateRadial(Tr*probeweight[None,:],r)/integrateRadial(probeweight,r) #; readings.append(Tr)
			#Tr=np.roll(Tr,int(len(Tr)/2))
			lplot([t*1e6],[Tr],"time (μs)","T (K)","T(t,z=0,irpr)",datalabels=[""],markers=["k-"])#,forcedBoundsX=[0,max(t*1e6)],forcedBoundsY=[0,None])

	elif TRZopt=="play-gif":
		updateStatus("playing","green")
		for i in range(npics):
			img.config(file="guigif/gui"+str(i)+".png")
			frameR.update()
			time.sleep(.03)

		#	setParam("fm",0.01)
		#	T,d,r,Ts2=Tz(rsteps=nrz,dsteps=nrz,maxdepth=1.5e-6*2,maxradius=maxrad,full=True)
		#	M2=np.sqrt(Ts2.real**2+Ts2.imag**2)
		#	for i in range(N):
		#		phi=i/N*2*np.pi
		#		T=Ts*np.exp(1j*phi) ; T=T.real+M2
				#T,dT=melt(T,Tmelt,Hmelt,C)
		#		showTrzs(T,d,r,includeTPD=False,plotting="saveguigif/gui"+str(i)+".png") #,bonusContours=[Tmelt,Tmelt+dT])
		#		img.config(file="guigif/gui"+str(i)+".png")
				#times.append(i) #; readings.append(T[0][0])
				#plot([r],[T[0,:]],"r (m)","T (K)") ; sys.exit()
				#Tr=np.trapz(T[0,:]*r,x=r)*np.pi*2 ; readings.append(Tr) # try it yourself: 
				#Tr=T[0,:] ; probeweight=2/np.pi/getParam("rpr")*np.exp(-2*r**2/getParam("rpr")**2)
				#Tr=integrateRadial(Tr*probeweight,r) ; readings.append(Tr)
		#		frameR.update()
			#os.system("convert -delay 20 -loop 0 *jpg animated.gif") # TODO, should consider adding gif saving too
		#plot([times],[readings],"time","probe reading",includeZeroY=False)

def addFileToSimultTab(event):
	global localVars
	# How do we detect which add file button was pressed? event.widget returns the button object. then we can get the button ID: 
	# rowID=int(str(event.widget)[-1])-3 	# third button we added is row 0 (after fit and hypo) so get button name...
	# but this is janky. adding new buttons to the top of the tab will affect the "-3" offset needed. instead, get the button text:
	buttonText=event.widget.config('text')[-1]
	rowID=int(buttonText.split()[-1])
	newfiles=list(tk.filedialog.askopenfilenames())
	for i,en in enumerate(newfiles):
		key="l_simultFile"+str(i+rowID)
		localVars[key]=en
		globalLookup["multifitting"][key].set(en)
		#globalLookup["multifitting"][key].xview_moveto("end")

for i in range(10):
	localVars["l_simultFile"+str(i+1)]=""			# a bunch of numbered localVars
	localVars["l_simultMode"+str(i+1)]="TDTR"
	localVars["l_simultGlos"+str(i+1)]=""
	globals()["aFTSB"+str(i+1)]=addFileToSimultTab		# dummy functions as copies of addFileToSimultTab()

def processAllMultiFields(exitOn="files"):
	files=[] ; types=[] ; magic=[] 
	# loop through all lines
	for i in range(10):
		if exitOn=="files" and len(localVars["l_simultFile"+str(i+1)])==0:
			break
		if exitOn=="globals" and len(localVars["l_simultGlos"+str(i+1)])==0:
			break
		files.append(localVars["l_simultFile"+str(i+1)])
		types.append(localVars["l_simultMode"+str(i+1)])
		magic.append(localVars["l_simultGlos"+str(i+1)])
	# initialize settables with mode
	settables={"mode":types}				# should be a {gloName:[values,for,each,file]} structure
	# first pass, detect what globals the user has requestd
	for glos in magic:
		for glovar in glos.split(","):				# "gamma=1,fm=1000" --> ["gamma=1","fm=1000"]
			if "=" not in glovar:
				continue
			glo,var=glovar.strip().split("=")		# ["gamma","1"]
			if glo not in settables.keys():
				settables[glo]=[ getVar(glo) for i in range(len(types)) ]	# default with current value
	# second pass, set whatever the user has requested
	for i,glos in enumerate(magic):
		for glovar in glos.split(","):
			if "=" not in glovar:
				continue
			glo,var=glovar.strip().split("=")
			settables[glo][i]=type(settables[glo][i])(var)
	return files,settables

def clearMultiFields(event):
	for i in range(10):
		localVars["l_simultFile"+str(i+1)]="" ; localVars["l_simultGlos"+str(i+1)]=""
	updateAllFieldsFromGlobals()

@wrapper
def simult(event):
	#global lastRun ; lastRun="simultaneous"
	files,settables=processAllMultiFields(exitOn="files") ; types=settables["mode"]
	global filesSimult ; filesSimult=files
	#global simultTypes ; simultTypes=types
	writeToLogFile("files:"+str(files)+","+str(types))
	r,e=ss2(files,types,plotting="save",settables=settables)
	global lastResult ; lastResult=r
	printToResultsPanel(str(r)+","+str(e)) # TODO we had a sneaky bug here, where this line crashed with "out(str(r,e))", and we never noticed, because i guess the newWin process crashed, not the main, or something like that (onclick just ended, no problem!) we only noticed because lastSolution wasn't correctly populated. could there be other stuff like this?



@wrapper
def hypothetical(event):
	files,settables=processAllMultiFields(exitOn="globals") #; types=settables["mode"]
	print(settables)
	thresh=float(localVars["l_contval"])/100
	valRange,generatedFiles=predictUncert(threshold=thresh,settables=settables)

	# set up lastResult global for 2D contours to run after. 
	global lastResult,filesSimult
	lastResult=[ getParam(v) for v in getVar("tofit") ]
	filesSimult=generatedFiles


# borrow the plot object which TDTR_fitting > niceplot generated, and display them
setVar("fignames",["gui.png","gui.svg","gui.csv"]) # required to suppress free-floating plot! wild! 
from matplotlib.figure import Figure
from niceplot import getPlotObjs ; from nicecontour import getContObjs
funcsUseContours=["runTRZ","viewMap","runMonte","cont2D","cont2DSimult","cont3DFlatSimult"]
funcsDIY=["runContour2D"]
liveplot={} ; customPlotted=False
def updatePlot(whatWasRunning):
	global liveplot
	whatWasRunning=whatWasRunning.split("function ")[-1].split()[0] # "<function runContour2D at 0x7f190489e700>" --> "runContour2D"
	if whatWasRunning in funcsUseContours:
		ax,fig=getContObjs()
	else:
		ax,fig=getPlotObjs()
		# if there are multiple datasets, update the color of later ones
		newcols=['g','orange','b','r','purple', # https://matplotlib.org/stable/gallery/color/named_colors.html
				'firebrick','darkorange','darkgreen','darkblue','indigo',
				'tomato','goldenrod','yellowgreen','cornflowerblue','mediumslateblue']
		newcols=newcols+newcols+newcols+newcols+newcols
		[ l.set_color(newcols[int((i-1)/2)]) for i,l in enumerate(ax.get_lines()) if i%2==1 ]
		# if there are more than 5 datasets, only show the first 5 in the legend!
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles[:10], labels[:10])
	liveplot["plot"],liveplot["fig"]=ax,fig
	if len(liveplot)>2:
		liveplot["canvas"].get_tk_widget().destroy()
		liveplot["toolbar"].destroy()
	# merely replacing the fig,ax objects is not enough. must and update canvas and toolbar objects
	liveplot["fig"].set_tight_layout(True)
	liveplot["fig"].set_dpi(100) 
	liveplot["canvas"] = FigureCanvasTkAgg(liveplot["fig"], master = framePlot)	# figure goes on a tkinter canvas
	liveplot["toolbar"] = NavigationToolbar2Tk(liveplot["canvas"],framePlot)	# toolbar, goes on the window, references the canvas
	liveplot["toolbar"].update()		
	liveplot["canvas"].get_tk_widget().pack(fill='both',expand=True)
	window.update() # problem with start_event_loop, it takes control over the main loop from tkinter! but this (https://matplotlib.org/stable/api/backend_bases_api.html#matplotlib.backend_bases.FigureCanvasBase.draw_idle) says we "redraw once control returns to the GUI event loop", so how do we do that without stealing? just update the window mainloop.
#def updatePlot(funcName,fig=None,ax=None):


def printToResultsPanel(printstring): # previously "out()"
	te_res.insert(tk.END,printstring+"\n")	# write to results Text field, including a new line
	te_res.see("end")			# and scroll the text entry field to the bottom
	frameR.update()				# update the frame containing the text field
	writeToLogFile("[output] : "+printstring)

def writeToLogFile(logstring): # previously "log()"
	f=open("gui.log",'a+')
	now=datetime.datetime.now() ; now=now.strftime("%Y-%m-%d_%H:%M:%S")
	f.write(now+"\n")
	f.write(logstring+"\n")
	f.close()

@wrapper
def quit_me():				# weird thing, when we generate a matplotlib plot, tkinter main loop doesn't exit when we click the x.
	writeToLogFile("Quitting")	# to deal with that, we detect a "delete window" and then use that to quit.
	window.quit()			# https://stackoverflow.com/questions/55201199/the-python-program-is-not-ending-when-tkinter-window-is-closed
	window.destroy()
	#global done
	#done=True

main()

# https://stackoverflow.com/questions/3702675/catch-and-print-full-python-exception-traceback-without-halting-exiting-the-prog
# https://stackoverflow.com/questions/14000944/finding-the-currently-selected-tab-of-ttk-notebook
# https://stackoverflow.com/questions/16373887/how-to-set-the-text-value-content-of-an-entry-widget-using-a-button-in-tkinter