# v0.71 (goes with 0.161)
import matplotlib,time,threading,os,sys
import tkinter as tk
import multiprocessing ; multiprocessing.freeze_support() # https://stackoverflow.com/questions/32672596/pyinstaller-loads-script-multiple-times
from tkinter import filedialog,ttk
from tkinter import *
import tkinter.font as tkf
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
matplotlib.use("TkAgg")
#if False: # 
#	matplotlib.use("svg")

#import tkinter.filedialog ; import tkinter as tk ; from tkinter import ttk ; import tkinter.font as tkf
#window=tk.Tk()	# create the tkinter window object
from TDTR_fitting import * #; from plotter import *
#try:
#	sys.path.insert(1,"../niceplot")
#except:
#	pass
#from niceplot import getPlotObjs # TDTR_fitting.py handles a clever conditional path import
#from nicecontour import getContObjs
matplotlib.use("svg") # needed for windows apparently...idk why i commented it out before. https://github.com/pyinstaller/pyinstaller/issues/6760 says you'll get a "ModuleNotFoundError: No module named 'matplotlib.backends.backend_svg'" with pyinstaller-compiled (but not running python on windows)
#import time,threading,os,sys


#  _____________________  in theory this should all be
# |  ___   ___________  | automatically expandable as
# | |   | |           | | we resize the window. we
# | | B | |           | | learned how to do this via
# | | U | | updatable | | ThermoreflectanceControl.py,
# | | T | |   plot    | | so we'll follow that general
# | | T | |           | | strategy here too.
# | | T | |___________| |
# | | O |  ___________  | buttons itself will also be
# | | N | |           | | broken down into N frames
# | | S | | results   | | for different classes of
# | |___| |___________| | entries/dropdowns/etc
# |_____________________|

window=Tk() ; window.title("TDTR fitting!")
def makeExpandable(master,rowWeights,colWeights):
	for r,w in enumerate(rowWeights):
		master.rowconfigure(r,weight=w,uniform=str(master))
	for c,w in enumerate(colWeights):
		master.columnconfigure(c,weight=w,uniform=str(master))
# left vs right panels
frameL=Frame(master=window) ; frameL.grid(row=0,column=0,sticky="NSEW")
frameR=Frame(master=window) ; frameR.grid(row=0,column=1,sticky="NSEW")
makeExpandable(window,[1],[2,3])
# top vs bottom panels on the right
framePlot=Frame(master=frameR) ; framePlot.grid(row=0,column=0,sticky="NSEW")
frameResu=Frame(master=frameR) ; frameResu.grid(row=1,column=0,sticky="NSEW")
makeExpandable(frameR,[3,1],[1])

# UNCOMMENT THESE TO DRAW COLOR-CODED BORDERS AROUND EACH FRAME (EG, TO CHECK THAT GRID ELEMENTS EXPAND APPROPRIATELY)
#colors=["red","orange","yellow","green","blue","purple","black"]*10
#for i,frame in enumerate([frameL,frameR,framePlot,frameResu]):
#		frame.configure(highlightbackground=colors[i],highlightthickness=10)






# How do we write GUIs? https://realpython.com/python-gui-tkinter/
# and how do we turn them into windows executables? https://stackoverflow.com/questions/48299396/converting-tkinter-to-exe
# how to read (and reload) images? https://blog.furas.pl/python-tkinter-how-to-load-display-and-replace-image-on-label-button-or-canvas-gb.html
# how to write function wrappers? https://www.geeksforgeeks.org/function-wrappers-in-python/
# how to add border to tkinter frames: https://stackoverflow.com/questions/50691267/how-to-put-border-for-frame-in-python-tkinter
# choose a file dialog: https://stackoverflow.com/questions/3579568/choosing-a-file-in-python-with-simple-dialog
# dropdown menus: https://www.geeksforgeeks.org/dropdown-menus-tkinter/
# tk.PhotoImage playing poorly with matplotlib: https://stackoverflow.com/questions/38602594/how-do-i-fix-the-image-pyimage10-doesnt-exist-error-and-why-does-it-happen
# HOW TO COMPILE:
#   Linux: python3 -m pip install pyinstaller ; pyinstaller --onefile gui.py
#   Windows: install python3 in wine (wine python_installer.exe, be sure to check "add to path"), make sure you have pip (https://pip.pypa.io/en/stable/installation/#get-pip-py), "wine python3 -m pip install [dependencies]", "wine pyinstaller --onefile gui.py")
#   MacOS: install homebrew, "brew install python@3.8", "brew install [dependencies]" or "wine python3 -m pip install [dependencies] --user", install ext4fuse if you want to read your linux partition ( https://www.maketecheasier.com/mount-access-ext4-partition-mac/ , https://github.com/gerard/ext4fuse/issues/66 ), "python3 -m PyInstaller --onefile gui.py"
# macOS-specific changes: multiprocessing needs to be forcefully killed (else, it'll just keep spawning windows, ffs), and "import matplotlib ; import matplotlib.pyplot as plt ; plt.plot(range(10)) ; plt.show() ; import tkinter as tk; window=tk.Tk()" will crash all to hell. Unclear why. So simply import TDTR_fitting after creating the window object.
# TODO NEED BETTER VALIDATION, eg, if Kr is used anywhere, but system is isotropic (Kr set to "Kz"), warn the user instead of just crashing. 
# TODO: do people want KZF? fiber map plotting? any logging missing? can we trick the code into giving us all the solve(multi) plots on one plot, by having some sort of "don't clear the plot before making the new one" flag? AND/OR (and i'm not sure if these are compatible), have plotting code generate the plot TWICE, once to save it off with the data (so the user can go look at it later if they like), AND again here as gui.png so we can display it?
# TODO should have more error validation (you put in wrong params, don't just crash, warn the user. you chose the wrong file type (TDTR vs SSTR vs params matric), don't just crash, warn the user).
#import multiprocessing ; multiprocessing.freeze_support() # https://stackoverflow.com/questions/32672596/pyinstaller-loads-script-multiple-times
#import tkinter.filedialog ; import tkinter as tk ; from tkinter import ttk ; import tkinter.font as tkf
#window=tk.Tk()	# create the tkinter window object
#from TDTR_fitting import * ; from plotter import *
#matplotlib.use("svg") # needed for windows apparently...
#import time,threading,os,sys
#from multiprocessing import set_start_method
#if __name__=='__main__':
#	set_start_method("spawn")
def l2s2D(list2D,fieldChar=",",lineChar="\n"):	# if 2D, [a,b,c],[d,e,f]] -> "a,b,c;d,e,f"
	list2D=[[str(v) for v in row] for row in list2D] # each element becomes a string
	list2D=[fieldChar.join(row) for row in list2D] # joint each row by ","
	return lineChar.join(list2D) # join rows together by ";"
def s2l2D(string2D,fieldChar=",",lineChar="\n"): # inverse of above
	list2D=string2D.split(lineChar) # split rows into entries in a list
	list2D=[ row.split("#")[0] for row in list2D ] # strip comments off each row
	list2D=[ row.split(fieldChar) for row in list2D  if len(row)>0 ] # purge blank lines
	list2D=[[v.strip() for v in row] for row in list2D] # purge whitespace
	list2D=[["Kz" if v in ["Kz","kz","KZ"] else eval(v) for v in row] for row in list2D] # tp mayn't contain strings! 
	return list2D
def l2s1D(list1D):
	return ",".join([str(v) for v in list1D])
def s2l1D(string1D):
	list1D=string1D.split(",")
	list1D=[float(v) if isNum(v) else v.strip() for v in list1D]
	return list1D

#setFigNames(["gui.png","gui.svg","gui.csv"])
setVar("fignames",["gui.png","gui.svg","gui.csv"])

# SET UP WINDOW STRUCTURE
#window.title("TDTR fitting") #; window.wm_state('zoomed') # window title, and initialize it maximized
#window.resizable(height = None, width = None) # make sure window is resizable (default on linux/windows, but appears to need to be explicitly declared on mac)
icofile=os.path.dirname(__file__) # current working directory: this is where files are included when we use pyinstaller arg "--add-data"
icofile=icofile.replace("\\","/") # https://stackoverflow.com/questions/41870727/pyinstaller-adding-data-files
icofile=icofile.split("/")
icofile.append("TDTR_fitting.png")
icofile="/".join(icofile) ; print(icofile)

#window.tk.call('wm', 'iconphoto', window._w, tk.PhotoImage(file=icofile)) # https://stackoverflow.com/questions/18537918/why-isnt-ico-file-defined-when-setting-windows-icon
#window.wm_iconphoto(False,tk.PhotoImage(file=icofile))
#window.iconbitmap(icofile.replace(".png",".ico"))
cellWidth=11 ; numCells=5

frameLL=frameL
canvas = tk.Canvas(frameLL)
scrollbar = ttk.Scrollbar(frameLL, orient="vertical", command=canvas.yview)
frameL = ttk.Frame(canvas)
frameL.bind( "<Configure>", lambda e: canvas.configure( scrollregion=canvas.bbox("all") ) )
canvas.create_window((0, 0), window=frameL, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# ADDS A SCROLLBAR TO LEFT FRAME: https://blog.teclado.com/tkinter-scrollable-frames/
#frameLL=ttk.Frame(master=window)#,highlightbackground="black",highlightthickness=1)
#canvas = tk.Canvas(frameLL)
#scrollbar = ttk.Scrollbar(frameLL, orient="vertical", command=canvas.yview)
#frameL = ttk.Frame(canvas)
#frameL.bind( "<Configure>", lambda e: canvas.configure( scrollregion=canvas.bbox("all") ) )
#canvas.create_window((0, 0), window=frameL, anchor="nw")
#canvas.configure(yscrollcommand=scrollbar.set)
#frameLL.grid(row=0,column=0,sticky='nsew')
#canvas.pack(side="left", fill="both", expand=True)
#scrollbar.pack(side="right", fill="y")

#canvasL=tk.Canvas(master=window) ; canvasL.grid(row=0,column=0)
#scroll=ttk.Scrollbar(canvasL,orient="vertical",command=canvasL.yview)
#scroll.pack(side=tk.RIGHT, fill=tk.Y)
#frameL=tk.Frame(master=canvasL,highlightbackground="black",highlightthickness=1)	# |''''''''''|'''''''''| right side takes the plot, with a results
#frameL.pack(side=tk.LEFT) 								# | |BU''''| |         | panel below it. left side gets buttons, 
#frameL.bind( "<Configure>", lambda e: canvasL.configure( scrollregion=canvasL.bbox("all") ) )
#canvasL.create_window((0, 0), window=frameL, anchor="nw")
#frameR=tk.Frame(master=window,highlightbackground="black",highlightthickness=1)		# | |______| |         | entry field for props and fitting, and
#frameR.grid(row=0,column=1)								# | |US    | |         | the rest of the individual entry fields.
frames={"BU":"Actions:","US":"Universal settings:","TDTR":"TDTR settings:",		# | |______| |         | entry fields are sorted by relevancy:
	"SSTR":"SSTR settings:","PWA":"PWA settings:",					# | |TS    | |         | universally relevant, or TDTR/SSTR/PWA
	"US3":"Universal settings (uncommon) :"}					# | |______| |         | -specific.

#window.columnconfigure(0,weight=1) # only allow expansion of frameL
#window.columnconfigure(1) # image stays it's original size
											# |  ...     |         |
for i,key in enumerate(frames.keys()):							# |L_________|R________|
	frame=tk.Frame(master=frameL,width=numCells*cellWidth)#,highlightbackground="black",highlightthickness=1)
	frame.grid(row=i,column=0,sticky=tk.NW) #,highlightbackground="black",highlightthickness=1)
	frame.columnconfigure(0, weight=1)
	lb=tk.Label(master=frame,text=frames[key],font=("bold"),fg="green") ; lb.grid(row=0,column=0,sticky=tk.NW,columnspan=numCells)
	frames[key]=frame




RC=dict(zip(frames.keys(),[ [1,0] for i in range(len(frames)) ]))	# a set of row/column counters, so we can easily add a new item to various frames
	
lb_running=tk.Label(master=frames["BU"],text="") ; lb_running.grid(row=0, column=3) # instead of a real progress bar, we'll just have a little text label, color-coded too, which says if we're running, errored, or completed. (progbar is "hard". you need to use threading to run whatever you're running while simultaneously updating the progress bar. then there are complications with having the threads communicate with the main thread (either to update the progress bar, write outputs, or interact with the user, eg, for file selection)

# THERMAL PROPERTIES TEXT ENTRY FIELD, goes in "universal settings" frame, full width! 
tpheader= "           C         ,              Kz or G⁽⁻¹⁾           ,       d       ,    Kr"
tpdef=          "  C_Al   ,    K_Al    , 80e-9 ,  Kz      # Layer 1\n"
tpdef=tpdef+    "            1/200e6                      # Interface 1\n"
tpdef=tpdef+    "C_Sapph  ,  K_Sapph   ,   1   ,  Kz      # Layer 2"
tpfooter= "      (J/m³/K)   ,  (W/m/K) or (W/m²/K)⁽⁻¹⁾ ,    (m)    ,   (W/m/K)"
lb=tk.Label(master=frames["US"],text="Thermal Properties:") ; lb.grid(row=1,column=0,columnspan=numCells)
lb=tk.Label(master=frames["US"],text=tpheader) ; lb.grid(row=2,column=0,columnspan=numCells,sticky="NW")
te_tprops=tk.Text(master=frames["US"],width=numCells*cellWidth,height=5) ; te_tprops.insert("1.0",tpdef)
te_tprops.grid(row=3,column=0,columnspan=numCells,rowspan=5) # text entry field object
lb=tk.Label(master=frames["US"],text=tpfooter) ; lb.grid(row=8,column=0,columnspan=numCells,sticky="NW")

# FITTING PARAMETERS TEXT ENTRY FIELD, goes in "universal settings" frame, right below tprops above
lb_tofit=tk.Label(master=frames["US"],text="fitting params:")				# label object for fitted params field				
en_tofit=tk.Entry(master=frames["US"])							# text entry field object
en_tofit.insert(0,l2s1D(tofit))								# default in parameters
lb_tofit.grid(row=9,column=0,columnspan=numCells) ; en_tofit.grid(row=10,column=0,columnspan=numCells)	# add both objects to the window

RC["US"][0]=11 ; RC["US"][1]=0								# update row/columns counters, accounting for manually-added above

# values entered (from entry fields or dropdowns) either directly write to TDTR_fitting parameters or globals, or we can use these custom setter/getter functions, for storing values locally, or wrapping (eg, "yes/no" --> TDTR_fitting's boolean globals)
def setGuiVar(name,val):
	global fields
	fields[name]["value"]=val
def getGuiVar(name):
	return fields[name]["value"]
def setBool(name,val):
	val=(val=="yes") # convert "yes" / "no" from dropdowns into True / False
	setVar(name,val)
def getBool(name):
	val=getVar(name)
	return {True:"yes",False:"no"}[val]
def setFigSize(name,val):
	screenHeight=window.winfo_height()-tkf.Font(font='TkDefaultFont').metrics('linespace')*8.5 # max fig height should be window minus the size of the results output field (can also get screensize via winfo_screenheight)
	if 6*val > screenHeight:
		val=int(screenHeight/6)
	if val<10:
		val=10
	setFigDPI(val)
	setGuiVar(name,val)
def setList(name,val):
	if len(val)>0:
		val=s2l1D(val)
	else:
		val=[]
	setVar(name,val)
def getList(name):
	return l2s1D(getVar(name))
def getRad(var): # simply converts TDTR_fitting.py's "m" into gui.py's "um"
	return getParam(var)*1e6
def setRad(var,val):
	setParam(var,val*1e-6)
# each variable needs an internal name and an alias for an externally-vieable name, a set and get function, a type (text entry, dropdown), a default, and which frame it will be placed in
fields={"rpr"      :{"alias":"probe r (μm)"   ,"setter":setRad  ,"getter":getRad   ,
		     "type":"entry","value":getRad("rpr")  ,"where":"US"},
	"rpu"      :{"alias":"pump r (μm)"    ,"setter":setRad ,"getter":getRad   ,
		     "type":"entry","value":getRad("rpu") ,"where":"US"},
	"fm"       :{"alias":"fmod (Hz)"     ,"setter":setParam  ,"getter":getParam   ,
		     "type":"entry","value":getParam("fm")   ,"where":"US"},
	"fitting"  :{"alias":"R/M/X/Y"       ,"setter":setVar    ,"getter":getVar     ,
		     "type":"drop" ,"value":"R;R;M;X;Y;P"    ,"where":"TDTR"},
	"doPhaseCorrect":{"alias":"phase corr.","setter":setBool ,"getter":getBool    ,
		     "type":"drop" ,"value":"yes;yes;no"     ,"where":"TDTR"},
	"gamma"    :{"alias":"gamma"         ,"setter":setParam  ,"getter":getParam   ,
		     "type":"entry","value":getParam("gamma"),"where":"SSTR"},
	"autorpu"  :{"alias":"auto pu r"     ,"setter":setBool   ,"getter":getBool    ,
		     "type":"drop" ,"value":"yes;yes;no"     ,"where":"US3"},
	"autorpr"  :{"alias":"auto pr r"     ,"setter":setBool   ,"getter":getBool    ,
		     "type":"drop" ,"value":"yes;yes;no"     ,"where":"US3"},
	"autofm"   :{"alias":"auto fmod"     ,"setter":setBool   ,"getter":getBool    ,
		     "type":"drop" ,"value":"yes;yes;no"     ,"where":"US3"},
	"da"       :{"alias":"pu depth (m)"  ,"setter":setParam  ,"getter":getParam   ,
		     "type":"entry","value":getParam("da")   ,"where":"US3"},
	"ma"       :{"alias":"pr depth (m)"  ,"setter":setParam  ,"getter":getParam   ,
		     "type":"entry","value":getParam("ma")   ,"where":"US3"},
	"Pow"        :{"alias":"Pu Power (W)"  ,"setter":setVar    ,"getter":getVar     ,
		     "type":"entry","value":getVar("Pow")      ,"where":"US3"},
	"thresh"   :{"alias":"cont. val (%)" ,"setter":setGuiVar ,"getter":getGuiVar  ,
		     "type":"entry","value":2.5              ,"where":"US3"},
	"contParam":{"alias":"cont. param"   ,"setter":setGuiVar ,"getter":getGuiVar  ,
		     "type":"entry","value":"Kz2"            ,"where":"US3"},
	"pertParam":{"alias":"pert. params"  ,"setter":setGuiVar ,"getter":getGuiVar  ,
		     "type":"entry","value":"all"            ,"where":"US3"},
	"perturbBy":{"alias":"pert. by (%)"  ,"setter":setGuiVar ,"getter":getGuiVar  ,
		     "type":"entry","value":"5"              ,"where":"US3"},
	"mode"     :{"alias":"experiment"    ,"setter":setVar    ,"getter":getVar     ,
		     "type":"drop" ,"value":"TDTR;TDTR;SSTR;PWA;FDTR;FD-TDTR"      ,"where":"US"},
	"pumpShape":{"alias":"pu profile"    ,"setter":setVar    ,"getter":getVar     ,
		     "type":"drop" ,"value": "gaussian;gaussian;gaussian_numerical;tophat;ring;ring_numerical","where":"US3"},
	"tshift"   :{"alias":"t shift (s)"   ,"setter":setVar    ,"getter":getVar     ,
		     "type":"entry","value":getVar("tshift") ,"where":"PWA" },
	"chopwidth":{"alias":"sq. width (%)" ,"setter":setVar    ,"getter":getVar     ,
		     "type":"entry","value":getVar("chopwidth"),"where":"PWA"  },
	"minimum_fitting_time":{"alias":"t min (s)","setter":setVar,"getter":getVar   ,
		     "type":"entry","value": getVar("minimum_fitting_time")     ,"where":"TDTR" },
	"fp"       :{"alias":"f pulse (Hz)"  ,"setter":setParam  ,"getter":getParam   ,
		     "type":"entry","value":getParam("fp")   ,"where":"TDTR"},
	"persistent":{"alias":"save settings","setter":setGuiVar ,"getter":getGuiVar  ,
		     "type":"drop" ,"value":"no;yes;no"      ,"where":"US3"},
	#"figdpi"   :{"alias":"fig. dpi"      ,"setter":setFigSize,"getter":getGuiVar  ,
	#	     "type":"entry","value":72               ,"where":"US3"},
	"verbose"  :{"alias":"verbose funcs" ,"setter":setList   ,"getter":getList    ,
		     "type":"entry","value":""               ,"where":"US3"},
	"normPWA"  :{"alias":"normalize"     ,"setter":setBool   ,"getter":getBool    ,
		     "type":"drop" ,"value":"yes;yes;no"     ,"where":"PWA"},
	"timeNormPWA":{"alias":"norm ts (%)" ,"setter":setVar    ,"getter":getVar     ,
		     "type":"entry","value":"25,75,10"       ,"where":"PWA"},
	"yshiftPWA":{"alias":"y shift"       ,"setter":setVar    ,"getter":getVar     ,
		     "type":"entry","value":0                ,"where":"PWA"},
	"timeMaskPWA":{"alias":"mask (%)"    ,"setter":setVar    ,"getter":getVar     ,
		     "type":"entry","value":"0:100" 	       ,"where":"PWA"},
	"asgif"    :{"alias":"T(r,z) as:"    ,"setter":setGuiVar ,"getter":getGuiVar  ,
		     "type":"drop" ,"value":"X;X;M;gen-gif;play-gif;T(t 0,z 0,r);T(t,z 0,r 0);T(t,z 0,irpr)"  ,"where":"US3"},
	"waveformPWA":{"alias":"waveform"    ,"setter":setVar    ,"getter":getVar     ,
		     "type":"drop" ,"value":"square;square;square-gauss;dirac;triangle;sine;arbitrary","where":"PWA"},
	"waveformReference":{"alias":"wave file","setter":setVar ,"getter":getVar     ,
		     "type":"entry" ,"value":""              ,"where":"PWA"},
	"plusMinus":{"alias":"+/- param 1"   ,"setter":setVar    ,"getter":getVar     ,
		     "type":"entry","value":0                ,"where":"US3"},
	"sumNPWA"  :{"alias":"N sines"       ,"setter":setVar    ,"getter":getVar     ,
		     "type":"entry","value":10000            ,"where":"PWA"},
	"normalizeSensitivity":{"alias":"norm sens.","setter":setBool,"getter":getBool,
		     "type":"drop" ,"value":"no;yes;no"      ,"where":"US3"},
	"sensitivityAsPercent":{"alias":"sens. as %","setter":setBool,"getter":getBool,
		     "type":"drop" ,"value":"no;yes;no"      ,"where":"US3"},
	"runAvgPWA":{"alias":"run. avg."     ,"setter":setVar    ,"getter":getVar     ,
		     "type":"entry","value":0                ,"where":"PWA"},
	"mapMode"  :{"alias":"map mode"      ,"setter":setGuiVar ,"getter":getGuiVar  ,
		     "type":"drop" ,"value":"dR/R;dR/R;Aux;fitted","where":"SSTR"},
	"auxRange" :{"alias":"aux min,max"   ,"setter":setGuiVar ,"getter":getGuiVar  ,
		     "type":"entry","value":"0,500"          ,"where":"SSTR"},
	"mapRange" :{"alias":"map min,max"   ,"setter":setGuiVar ,"getter":getGuiVar  ,
		     "type":"entry","value":"-1e4,1e4"          ,"where":"SSTR"},
	"neighborify":{"alias":"neighborify" ,"setter":setGuiVar ,"getter":getGuiVar  ,
		     "type":"drop" ,"value":"no;yes;no"      ,"where":"SSTR"},
	"useTBR":{"alias":"use TBR"          ,"setter":setBool   ,"getter":getBool    ,
		     "type":"drop" ,"value":"yes;yes;no"     ,"where":"US3"},
	#"phase"  :{"alias":"FDTR phase slope"  ,"setter":setParam  ,"getter":getParam   ,
	#	     "type":"entry","value":getParam("phase")   ,"where":"US3"},
	"time_normalize"  :{"alias":"t norm"  ,"setter":setVar  ,"getter":getVar   ,
		     "type":"entry","value":getVar("time_normalize")   ,"where":"TDTR"},
	"alpha":{"alias":"opt. pen. (m)"          ,"setter":setVar   ,"getter":getVar    ,
		     "type":"entry" ,"value":"0"     ,"where":"US3"}
}

settings={} ; files=[] ; lastrun="" ; ss2f=[] ; ss2t=[]
if os.path.exists("gui.log"):
	lines=open("gui.log",'r').readlines()
	for l in reversed(lines):
		if "settings:" not in l:
			continue
		break
	print(l)
	if "persistent=yes" in l:					# if persistent=="yes", we'll re-import all settings from the log
		settings=l.split("\" \"")
		settings[0]=settings[0].replace("settings: \"","")
		settings[-1]=settings[-1].replace("\"\n","")
		settings=[ s.split("=") for s in settings ]
		settings=dict(settings)
		# special handling of tprops field
		te_tprops.delete("1.0", tk.END)
		te_tprops.insert("1.0",settings["tp"].replace(";","\n"))
		en_tofit.delete(0,tk.END)
		en_tofit.insert(0,settings["tf"])
	elif "figdpi=" in l:								# otherwise, at *least* import the figsize
		l=l.split("figdpi=")[1] ; l=l.split("\"")[0]
		settings["figdpi"]=int(float(l))	

	for i,l in reversed(list(enumerate(lines))):
		# eg files:['.../testscripts/DATA/2022_04_28_Fiber/meas_pwa_20220428_134841.txt']
		# simultaneous:['/media/Alexandria/U Virginia/Research/Various Code/TDTR_fitting/testscripts/DATA/2022_02_15_Fiber/02152022_Al2O3_Cal_094260_1000Hz.txt', '/media/Alexandria/U Virginia/Research/Various Code/TDTR_fitting/testscripts/DATA/2022_02_15_Fiber/02152022_Al2O3_Cal_095242_10000000Hz.txt', '/media/Alexandria/U Virginia/Research/Various Code/TDTR_fitting/testscripts/DATA/2022_02_15_Fiber/magicModifiers.txt'],['SSTR', 'SSTR', 'SSTR']
		#print("CHECKING LINE",i,"FOR FILES:",l)
		if "files:" not in l: 		# 3 functions which we can "refit" off the bat: solve, simult, and viewMap.
			continue		# so first look for a line with filenames
		for fun in lines[i-1:i-3:-1]:	# then check the previous *two* lines for the function name
			if "runSolve" in fun or "viewMap" in fun or "simult" in fun:
				break
		#print("FOUND FUN",fun)
		if "runSolve" in fun: # e.g. "running:<function runSolve at 0x7f9dbaa5a440>"
			l=l.split(":")[-1].replace("[","").replace("]","").replace("'","").split(",")
			files=[f.strip() for f in l]
			print(files)
			lastrun="solve"
		elif "viewMap" in fun: # e.g. "running:<function viewMap at 0x7fe2ff902f80>"
			l=l.split(":")[-1].replace("[","").replace("]","").replace("'","").split(",")
			files=[f.strip() for f in l]
			print(files)
			lastrun="map"
		elif "simult" in fun: # e.g. "running:<function simult at 0x7f55faefb640>"
			ss2f,ss2t=l.split(":")[-1].split("],[")
			ss2f=ss2f.replace("[","").replace("'","").split(",")
			ss2t=ss2t.replace("]","").replace("'","").split(",")
			ss2f=[ f.strip() for f in ss2f ]
			ss2t=[ f.strip() for f in ss2t ]
			print(ss2f,ss2t)
			lastrun="simult"

		break
	

# A function for updating our row/column counter (one counter pair for each frame, ensuring subsequent entry fields don't lay atop previous)
def getRC(framename,n=numCells,s=2):
	global RC
	r,c=RC[framename]
	c_new=c+1 ; r_new=r+int(c_new/n)*s ; c_new=c_new%n
	RC[framename][0]=r_new ; RC[framename][1]=c_new
	return r,c

# loop through all entry fields from above:
for glo in fields.keys():
	where=fields[glo]["where"]						# which frame will it be placed on
	lb=tk.Label(master=frames[where],text=fields[glo]["alias"],width=cellWidth) 	# both text entry fields and dropdowns get labels, based on the alias
	if fields[glo]["type"]=="entry":
		field=tk.Entry(master=frames[where],width=cellWidth)			# text entry field
		val=fields[glo]["value"]					# default value
		if type(val)!=str:						# convert to string
			val=str(scientificNotation(val))
		if glo in settings.keys():
			val=settings[glo]
		field.insert(0,val)						# insert default value
		fields[glo]["field"]=field					# entry field object is stored off, so we can retrieve from it later
	elif fields[glo]["type"]=="drop":
		options=fields[glo]["value"].split(";")				# for dropdowns, "value" holds ";"-delimited list of options
		opt=tk.StringVar(window)					# selected option is stored in a stringVar object
		val=options[0] ; options=options[1:]				# first in list is default, the rest are the options
		if glo in settings.keys():
			val=settings[glo]
		opt.set(val)							# set the default; first entry in list of options
		fields[glo]["value"]=val					# update "value" to hold just current value
		field = tk.OptionMenu(frames[where], opt, *options)		# dropdown menu object
		field.config(width=cellWidth-3)
		fields[glo]["field"]=opt					# option object is stored off, not field (we call ".get" from it later)
	r,c=getRC(where)							# increment rows/columns, before placing both label and field
#	print(fields[glo]["alias"],r,c)
	lb.grid(row=r, column=c, sticky=tk.NW)						# field (entry field, or dropdown, goes below label)
	if fields[glo]["type"]=="entry":
		field.grid(row=r+1,column=c,sticky=tk.NW)
	else:
		field.grid(row=r+1,column=c)

# kind of a dumb hack to force all grids to a width of numCells, by adding extra grid entries to fill out the remainder of the last row
for where in RC:
	if where=="BU":
		continue
#	print(where)
	r,c=RC[where]
	c-=1
#	print(r,c)
	while c<numCells-1:
		r,c=getRC(where)
#		print("(",r,c,")")
		lb=tk.Label(master=frames[where],text="",width=cellWidth) 
		lb.grid(row=r, column=c, sticky=tk.NW)		


# GENERATED PLOT DISPLAYED
# THREE STRATEGIES:
# OPTION ONE: simply display the .png file which TDTR_fitting > niceplot saved off
"""
img = tk.PhotoImage(master=window)				# "gui.png" image is displayed, and reloaded if you run something
if os.path.exists("gui.png"):
	img.config(file="gui.png")
lb_img = tk.Label(master=framePlot,image=img)
lb_img.pack()
#"""
# OPTION TWO: create a canvas, and read from the globals TDTR_fitting sets up, and refresh the plot here manually
"""
from matplotlib.figure import Figure
liveplot={}
liveplot["fig"]=Figure(dpi = 100)	 #figsize=(5,5),		# matplotlib figure object
liveplot["plot"] = liveplot["fig"].add_subplot()				# subplot on the figure
liveplot["fig"].set_tight_layout(True) 	# https://stackoverflow.com/questions/6774086/how-to-adjust-padding-with-cutoff-or-overlapping-labels
liveplot["canvas"] = FigureCanvasTkAgg(liveplot["fig"], master = framePlot)	# figure goes on a tkinter canvas
liveplot["toolbar"] = NavigationToolbar2Tk(liveplot["canvas"],framePlot)	# toolbar, goes on the window, references the canvas
liveplot["toolbar"].update()		
liveplot["canvas"].get_tk_widget().pack(fill='both',expand=True) # fill/expand will mean figure auto-expands into space given (grid cell)
def updatePlot(): # "lp" = "local persistence": dicts as function arguments are persistent between function calls, so we take advantage of that to store our matplotlib plot object and tkinter canvas object, to we can easily access and update on subsequent calls
	global window,liveplot
	liveplot["plot"].clear()						# Whether this is the first run or not, clear the plot
	xs=getVar("plotXs") ; ys=getVar("plotYs") ; xlabel=getVar("plotXlabel") ; ylabel=getVar("plotYlabel")
	labels=getVar("plotLabels") ; title=getVar("plotTitle") ; markers=getVar("plotMarkers")
	print("updatePlot: xs",xs)
	for x,y,l,m in zip(xs,ys,labels,markers):
		c=m[0] ; mk=m[1]
		kwargs={"label":l,"color":c}
		if mk in list(matplotlib.lines.lineStyles.keys()):
			kwargs["linestyle"]=mk ; kwargs["marker"]=''
		else:
			kwargs["marker"]=mk ; kwargs["linestyle"]=''
		print(kwargs,x)
		liveplot["plot"].plot(x,y,**kwargs)					# plot the new dataset
	liveplot["plot"].set_xlabel(xlabel) ; liveplot["plot"].set_ylabel(ylabel)
	liveplot["plot"].legend()
	liveplot["canvas"].draw_idle()						# and refresh the canvas
	window.update() # problem with start_event_loop, it takes control over the main loop from tkinter! but this (https://matplotlib.org/stable/api/backend_bases_api.html#matplotlib.backend_bases.FigureCanvasBase.draw_idle) says we "redraw once control returns to the GUI event loop", so how do we do that without stealing? just update the window mainloop.
#"""
# OPTION 3: simply use niceplot's optionally passed-back fig,ax objects, and refresh the canvas here instead of replotting
"""
from matplotlib.figure import Figure
liveplot={}
def updatePlot(): 
	global window,liveplot
	if len(liveplot)>0:
		#liveplot["plot"].clear()						# Whether this is the first run or not, clear the plot
		liveplot["canvas"].get_tk_widget().destroy()
		liveplot["toolbar"].destroy()
	xs=getVar("plotXs") ; ys=getVar("plotYs") ; xlabel=getVar("plotXlabel") ; ylabel=getVar("plotYlabel")
	labels=getVar("plotLabels") ; title=getVar("plotTitle") ; markers=getVar("plotMarkers")
	liveplot["plot"],liveplot["fig"]=plot(xs,ys,xlabel=xlabel,ylabel=ylabel,labels=labels,title=title,markers=markers,filename="PLOTOBJ")
	# merely replacing the fig,ax objects is not enough. must and update canvas and toolbar objects
	liveplot["fig"].set_tight_layout(True)
	liveplot["fig"].set_dpi(100) 
	liveplot["canvas"] = FigureCanvasTkAgg(liveplot["fig"], master = framePlot)	# figure goes on a tkinter canvas
	liveplot["toolbar"] = NavigationToolbar2Tk(liveplot["canvas"],framePlot)	# toolbar, goes on the window, references the canvas
	liveplot["toolbar"].update()		
	liveplot["canvas"].get_tk_widget().pack(fill='both',expand=True)
	window.update() # problem with start_event_loop, it takes control over the main loop from tkinter! but this (https://matplotlib.org/stable/api/backend_bases_api.html#matplotlib.backend_bases.FigureCanvasBase.draw_idle) says we "redraw once control returns to the GUI event loop", so how do we do that without stealing? just update the window mainloop.
#"""
# OPTION 4: we probably don't want to replot it. we probably just want to borrow the plot object which TDTR_fitting > niceplot generated, and could conceivably just keep around for us!
from matplotlib.figure import Figure
from niceplot import getPlotObjs ; from nicecontour import getContObjs
#def updatePlot(whatWasRunning,fig=N
# conundrum: the "right" way to handle "most functions, by default, need their plots displayed" and "some functions, either all the time or some of the time, update their own plots"....idk. we can't set a global "update or not" flag, because once it's set to "not", how do we reset it to "update" when something else (anything else) is called? or maybe "wrapper" should only call updatePlot conditionally based on some flag?
funcsUseContours=["runTRZ","viewMap","runMonte","runContour2D"]
funcsDIY=["runContour2D"]
liveplot={} ; customPlotted=False
def updatePlot(whatWasRunning,fig=None,ax=None):
	global customPlotted,liveplot
	if customPlotted:
		return
	whatWasRunning=whatWasRunning.split("function ")[-1].split()[0] # "<function runContour2D at 0x7f190489e700>" --> "runContour2D"
	if fig is None or ax is None:
		if whatWasRunning in funcsUseContours:
			ax,fig=getContObjs()
		else:
			ax,fig=getPlotObjs()
			# if there are multiple datasets, update the color of later ones
			newcols=['r','orange','g','b','purple', # https://matplotlib.org/stable/gallery/color/named_colors.html
					'firebrick','darkorange','darkgreen','darkblue','indigo',
					'tomato','goldenrod','yellowgreen','cornflowerblue','mediumslateblue']
			newcols=newcols+newcols+newcols+newcols+newcols
			[ l.set_color(newcols[int((i-1)/2)]) for i,l in enumerate(ax.get_lines()) if i%2==1 ]


			# if there are more than 5 datasets, only show the first 5 in the legend!
			handles, labels = ax.get_legend_handles_labels()
			ax.legend(handles[:10], labels[:10])
	else:
		customPlotted=True
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
	
"""
liveplot={} ; lastPlot="plot"
def updatePlot1(whatWasRunning,fig=None,ax=None):
	global window,liveplot,lastPlot

	for fun in ["runTRZ","runContour2D","viewMap","runMonte"]:
		if fun in whatWasRunning:
			break
	else:
		lastPlot="plot"

	print("UPDATE PLOT",fig,ax,lastPlot,whatWasRunning)


	# STEP 1 IS RETREIVE fig/ax OBJS
	if fig is None or ax is None:
		

		if lastPlot=="plot":
			print("lastPlot",lastPlot)
			ax,fig=getPlotObjs()
			# if there are multiple datasets, update the color of later ones
			newcols=['r','orange','g','b','purple', # https://matplotlib.org/stable/gallery/color/named_colors.html
					'firebrick','darkorange','darkgreen','darkblue','indigo',
					'tomato','goldenrod','yellowgreen','cornflowerblue','mediumslateblue']
			newcols=newcols+newcols+newcols+newcols+newcols
			[ l.set_color(newcols[int((i-1)/2)]) for i,l in enumerate(ax.get_lines()) if i%2==1 ]


			# if there are more than 5 datasets, only show the first 5 in the legend!
			handles, labels = ax.get_legend_handles_labels()
			ax.legend(handles[:10], labels[:10])
		else:
			print("lastPlot",lastPlot)
			ax,fig=getContObjs()
	#else:
	#	lastPlot="other"
		#liveplot={}
	# TODO BUG: contour > 3D > updates plot itself > wrapper subsequently updates plot again > lastPlot=="plot" means we re-query fig/ax from whatever was run before. okay, we fix it by reordering stuff. but interestingly, there's still a big: run 3D ("play-gif" and "contour2D") it works. run 2D, it works. re-run 3D, it doesn't work. until you close and re-open. i assume because it is successfully querying the contour fig/ax objects? idk how to avoid this! i think the "right" answer is to quit with the bullshit of the wrapper function (the core underlying issue here is that wrapper re-calls updatePlot)
	# UPDATE THOSE
	liveplot["plot"],liveplot["fig"]=ax,fig
	

	#if fig is not None and ax is not None:
	#	liveplot["plot"],liveplot["fig"]=ax,fig
	#	liveplot["canvas"].get_tk_widget().destroy()
	#	liveplot["canvas"] = FigureCanvasTkAgg(liveplot["fig"], master = framePlot)
	#	liveplot["canvas"].get_tk_widget().pack(fill='both',expand=True)
	#	window.update()
	#	return


	#truths = [ fun in whatWasRunning for fun in ["runTRZ","contour2D"] ]
	#plotOrCont={False:"plot",True:"cont"}[True in truths]
	#global window,liveplot
	if len(liveplot)>2:
		print(liveplot)
		#liveplot["plot"].clear()						# Whether this is the first run or not, clear the plot
		liveplot["canvas"].get_tk_widget().destroy()
		liveplot["toolbar"].destroy()
	#xs=getVar("plotXs") ; ys=getVar("plotYs") ; xlabel=getVar("plotXlabel") ; ylabel=getVar("plotYlabel")
	#labels=getVar("plotLabels") ; title=getVar("plotTitle") ; markers=getVar("plotMarkers")
	#liveplot["plot"],liveplot["fig"]=plot(xs,ys,xlabel=xlabel,ylabel=ylabel,labels=labels,title=title,markers=markers,filename="PLOTOBJ")

	"#""	
	for fun in ["runTRZ","runContour2D","viewMap","runMonte"]: # a "whitelist" of functions which output contour/heatmaps instead of 2D plots
		if fun in whatWasRunning:
			liveplot["plot"],liveplot["fig"]=getContObjs()
			break
	else:
		liveplot["plot"],liveplot["fig"]=getPlotObjs()
	"#""
	#global lastPlot

	
	
	# merely replacing the fig,ax objects is not enough. must and update canvas and toolbar objects
	liveplot["fig"].set_tight_layout(True)
	liveplot["fig"].set_dpi(100) 
	liveplot["canvas"] = FigureCanvasTkAgg(liveplot["fig"], master = framePlot)	# figure goes on a tkinter canvas
	liveplot["toolbar"] = NavigationToolbar2Tk(liveplot["canvas"],framePlot)	# toolbar, goes on the window, references the canvas
	liveplot["toolbar"].update()		
	liveplot["canvas"].get_tk_widget().pack(fill='both',expand=True)
	window.update() # problem with start_event_loop, it takes control over the main loop from tkinter! but this (https://matplotlib.org/stable/api/backend_bases_api.html#matplotlib.backend_bases.FigureCanvasBase.draw_idle) says we "redraw once control returns to the GUI event loop", so how do we do that without stealing? just update the window mainloop.
"""

# RESULTS OUTPUT AREA:
lb_res=tk.Label(master=frameResu,text="RESULTS:")
te_res=tk.Text(master=frameResu,height=7)				# text entry field object
lb_res.pack() ; te_res.pack()					# add both objects to the window
#setVar("clf",False)						# this overrides solve > solverPlotting > lplot's plt.clf, allowing plotting of multiple

# TODO, consider a scroll bar: https://stackoverflow.com/questions/30669015/autoscroll-of-text-and-scrollbar-in-python-text-box
# when any button is pressed, before calling anything, we pass TDTR_fitting various text-enterable values, and after, we do some logging
def wrapper(func): # https://www.geeksforgeeks.org/function-wrappers-in-python/
	def wrapped(*args,**kwargs):
		#pltclf() # gui > TDTR_fitting.pltclf() > plot.killFigAx() > deletes figAx global which basically only solve dumps to
		global customPlotted ; customPlotted=False
		refreshPlotGlos()
		tp=te_tprops.get("1.0", tk.END) ; setVar("tp",s2l2D(tp))	# thermal properties matrix
		tf=en_tofit.get() #; setVar("tofit",s2l1D(tf))			# list of parameters we're fitting for
		setList("tofit",tf)		
		gloStrs=["\"tp="+tp.replace("\n",";")+"\"" , "\"tf="+tf+"\"" ]
		for glo in fields.keys():					# loop through every entry field...
			current=fields[glo]["getter"](glo)			#	getting its previous value (from the code)
			new=fields[glo]["field"].get()				# 	its new value (from the GUI)
			gloStrs.append("\""+glo+"="+new+"\"")			#	logging it
			if type(current)!=str:					#	converting types appropriately
				new=float(eval(new))				#	eval converts, say, "1e-3+80e-9" to a number
			fields[glo]["setter"](glo,new)				#	and setting its new value (in the code)
		log("settings: "+" ".join(gloStrs))
		log("running:"+str(func))
		# to handle detecting errors in func(), we'll just do a try/except. 
		updateStatus("RUNNING","blue")					# status field set to "running"
		try:
			func(*args,**kwargs)						# run the function, if no crashes, update status to "complete"
			if done:
				return
			updateStatus("complete","green")
			log("[success]")
		except: # https://stackoverflow.com/questions/8238360/how-to-save-traceback-sys-exc-info-values-in-a-variable/25212045
			out("ERROR WITH FUNC:"+str(func)+", please send your gui.log file to the developer. Windows: log file can be found in the same folder as the executable. MacOS: log file can be found in your \"home\" folder.") # if we DID crash, tell the user
			lb_running.configure(text="ERRORED",fg="red")
			exc=traceback.format_exc()					# get the call stack / crash log
			log("[FAILURE] : \n")						# and log that to file
			log(exc)
			out(exc)
		#if os.path.exists("gui.png"):					# finally, update the image
		#	img.config(file="gui.png")
		#	frameR.update()
		updatePlot(str(func))
	return wrapped
def updateStatus(status,color):
	lb_running.configure(text=status,fg=color) ; frames["BU"].update()

lastSolution=[] ; loggableResults={} # "HQ_Al2O3":{"03172022":{"f1":[K,G],"f2":[K,G],"f3":[K,G]}}} "place","date","results"
lastDirec="./"
def ask(multiple=True,fileOrDirec="file",text=""):
	global lastDirec
	# if lastDirec is still the default, check the file list
	if lastDirec=="./" and len(files)>0:
		lastDirec="/".join(files[-1].split("/")[:-1])
	# ask user for files:
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
	log("selected files:"+str(selected))
	return selected
def out(printstring):
	te_res.insert(tk.END,printstring+"\n")	# write to results Text field, including a new line
	te_res.see("end")			# and scroll the text entry field to the bottom
	frameR.update()				# update the frame containing the text field
	log("[output] : "+printstring)
def log(logstring):
	f=open("gui.log",'a+')
	f.write(logstring+"\n")
	f.close()

# FUNCTION FOR BUTTON WHICH RUNS SENSITIVITY PLOT GENERATION
@wrapper # "wrapper" wraps runSens, doing pre- and post- functionality appropriately
def runSens(event):
	sensitivity(plotting="save")

# FUNCTION FOR BUTTON WHICH KICKS OFF TDTRsolve
@wrapper
def runSolve(event,rerun=False):
	global files,lastrun ; lastrun="solve"
	if not rerun:
		files=ask()
	if len(files)==0:
		return
	results=[]
	log("files:"+str(files))
	if getVar("mode")=="FD-TDTR": # here we can "trick" our loop below, so we pass the whole set into the function, instead of each file individually.
		files=[files]	# janky, i know. deal with it. O-O'''-
	for f in files:
		res,err=solve(f,plotting="save")
		print(res,err)
		results.append(res)
		resultString=" , ".join( [p+"="+sigFigs(v,4) for p,v in zip(getVar("tofit"),res) ] )+" , "+sigFigs(err[0]*100)+"%"
		resultString+=" , "+f.split("/")[-1][:77-len(resultString)]
		out(resultString)
		#updatePlot("solve")
		if getVar("autoFailed"):
			out("WARNING: auto rpu/rpr/fm failed. check your file headers and/or radii.txt file! or change auto to \"no\" and set the values yourself")
	results=np.asarray(results)
	res,err=np.mean(results,axis=0),np.std(results,axis=0)
	resultStrings=[p+" = "+sigFigs(v*getScaleUnits(p)[0],4)+"+/-"+sigFigs(dv)+" "+getScaleUnits(p)[1] for p,v,dv in zip(getVar("tofit"),res,err) ]
	out("averaged +/- std:\n"+" , ".join(resultStrings))
	#for rs in resultStrings:
	#	out(" --> "+rs)
	#if mode!="FD-TDTR" and len(files)>1: # TODO would be nice if we could plot each individually-solved measurement together
	#	for 
	global lastSolution ; lastSolution=res
	global loggableResults # [where+"_"+what]=[]
	
	for f,r in zip(files,results):
		FR=fittedResult(f,getVar("tofit"),r)
		loggableResults[f]=FR

def refit(event): # runSolve is wrapped, no need to wrap refit 
	if len(lastrun)==0:
		out("Please run fitting first!")
		return
	if lastrun=="solve":
		runSolve(event,rerun=True)
	elif lastrun=="simult":
		simult(event,rerun=True)
	elif lastrun=="map":
		viewMap(event,rerun=True)

@wrapper
def avgFiles(event):
	files=ask() ; ftypes={"SSTR":"fSSTR","TDTR":"TDTR"} ; ftype=ftypes.get(getVar("mode"),"raw")
	fo,ig=fileAverager(files,fileType=ftype)
	out("files averaged, and outputted to: "+fo)

@wrapper
def checkPhase(event):
	if len(files)==0:
		out("run fitting first")
		return
	ts,data=readTDTR(files[-1],plotPhase=True)
@wrapper
def checkDelta(event):
	if len(files)==0:
		out("run fitting first")
		return
	r,e=solve(files[-1])
	ts,data=readFile(files[-1])
	Rs=func(ts,*r)
	lplot([ts,ts,ts],[data-Rs,data-Rs,np.zeros(len(ts))],markers=["k.","r:","k:"],labels=[" "," "," "],filename="gui.png")


# FUNCTION FOR BUTTON WHICH GENERATES T(r,z) PLOT
@wrapper
def runTRZ(event):
	global lastPlot ; lastPlot="contour"
	maxrad={True:getParam("rpu"),False:max(getVar("xoff"),getParam("rpu"))}["offset" in getVar("pumpShape")]*1.5
	#maxrad=getParam("rpu")*1.5
	nrz=50 ; nt=1000 ; npics=250
	omegas=np.asarray([0.01,getParam("fm")*2*pi]) # TRUE TEMPERATURE RISE IS SUM OF SS + MODULATED

	# Options include: X;M;gen-gif;play-gif;T(r,z=0,t=0);T(rpr,z=0,t)
	if fields["asgif"]["value"] in ["X","M"]:
		T,d,r,Ts=Tz(rsteps=nrz,dsteps=nrz,maxdepth=2*1.5e-6,maxradius=maxrad,full=True,omegas=omegas) # Tz returns T[d,r],depths[d],radii[r]
		Ts=np.sum(Ts,axis=0)
		T={ "X":Ts.real , "M":np.sqrt(Ts.real**2+Ts.imag**2) }[ fields["asgif"]["value"] ]
		#T,dT=melt(T,75,10e6,3e6)
		showTrzs(T,d,r,includeTPD=True,savefile="gui_Trz.txt") #,bonusContours=[Tmelt,Tmelt+dT])
	elif fields["asgif"]["value"] in ["gen-gif","T(t 0,z 0,r)","T(t,z 0,r 0)","T(t,z 0,irpr)"]:
		T,t,d,r=Ttzr(mindepth=0,maxdepth=1500e-9*2,dsteps=nrz,rsteps=nrz,maxradius=maxrad,tsteps=nt)
		#T=np.asarray( [ melt(Ts,50,100e6,3e6)[0] for Ts in T ] ) #; print(T,np.shape(T))
		T=np.asarray( [ melt(Ts,50,100e6,3e6)[0] for Ts in T ] ) #; print(T,np.shape(T))
		Tmax=max([np.amax(Ts) for Ts in T]) ; Tmin=min([np.amin(Ts) for Ts in T])
		if fields["asgif"]["value"]=="gen-gif":
			for i in range(0,npics):
				showTrzs(T[i*int(nt/npics)],d,r,includeTPD=False,plotting="saveguigif/gui"+str(i)+".png")#cbounds=[Tmin,Tmax,11])#,bonusContours=[50-.1,50+.1])
				#img.config(file="guigif/gui"+str(i)+".png")
				updatePlot("runTRZ")
				frameR.update()
			out("find your frames in folder \"guigif\"...")
		elif fields["asgif"]["value"]=="T(t 0,z 0,r)":
			lplot([r*1e6], [T[0,0,:]], xlabel="radius (μm)", ylabel="T (K)", title="T(t=0,z=0,r)", labels=[""], markers=["k-"])#,forcedBoundsX=[0,14])
			updatePlot("runTRZ")
			frameR.update()
		elif fields["asgif"]["value"]=="T(t,z 0,r 0)":
			T=T[:,0,0] #; T=np.roll(T,int(len(T)/2))
			lplot([t*1e6],[T],"time (μs)","T (K)","T(t,z=0,r=0)",datalabels=[""],markers=["k-"])#,forcedBoundsX=[0,max(t*1e6)],forcedBoundsY=[0,None])
		elif fields["asgif"]["value"]=="T(t,z 0,irpr)":
			Tr=T[:,0,:] ; probeweight=2/np.pi/getParam("rpr")*np.exp(-2*r**2/getParam("rpr")**2)
			Tr=integrateRadial(Tr*probeweight[None,:],r)/integrateRadial(probeweight,r) #; readings.append(Tr)
			#Tr=np.roll(Tr,int(len(Tr)/2))
			lplot([t*1e6],[Tr],"time (μs)","T (K)","T(t,z=0,irpr)",datalabels=[""],markers=["k-"])#,forcedBoundsX=[0,max(t*1e6)],forcedBoundsY=[0,None])

	elif fields["asgif"]["value"]=="play-gif":
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
		


# FUNCTION FOR CONTOUR UNCERTAINTY TODO should have separete buttons for perturbParams, and contours. need an entry field for which params to perturb and by how much. need an entry field for each paramOfInterest for contouring, including a "both" option, where if "both" is selected, we can also generate actual contour plot + walking along for each param (1Axis)
@wrapper
def runContour(event):
	if len(lastrun)==0:
		out("Please run fitting first. we'll do contours on your last-fitted file(s)")
		return
	fs=files ; solvefunc={"func":solve,"kwargs":{}}
	if lastrun=="simult":
		fs=[ss2f] ; solvefunc=solvefunc={"func":ss2,"kwargs":{"listOfTypes":ss2t}} #; setVar("ss2Types",ss2t)
	if getVar("mode")=="FD-TDTR":
		fs=[fs]
	for f in fs:
		p=fields["contParam"]["getter"]("contParam")
		thresh=fields["thresh"]["getter"]("thresh")
		bnds,fout=measureContour1Axis(f,paramOfInterest=p,plotting="savefinal",resolution=100,threshold=thresh/100,solveFunc=solvefunc)
		error=(bnds[1]-bnds[0])/2 ; errorp=(bnds[1]-bnds[0])/(bnds[1]+bnds[0])
		#out(p+" : "+str(bnds)+" : +/- "+str(error))
		fact,unit=getScaleUnits(p)
		out(scientificNotation(bnds[0]*fact,2)+" <= "+p+" <= "+scientificNotation(bnds[1]*fact,2)+" "+unit+
			" (+/-"+scientificNotation(error*fact,2)+" "+unit+" or "+
			"+/-"+str(np.round(errorp*100,1))+"%)")

@wrapper
def runContour2D(event):
	global lastPlot ; lastPlot="contour"
	#pr=[[v*.5,v*2] for v in lastSolution]
	pr=[[v*.1,v*2] for v in lastSolution]
	#pr[1][1]*=1.5
	#print(pr) ; sys.exit()
	#pr=[[50e6, 2000e6], [500, .5e6]]
	#pr=[[48.32448319018158, 193.29793276072633], [60051084.84246936, 240204339.36987743], [28220160.135147046, 112880640.54058819]]

	D="2D"
	if len(getVar("tofit"))==3 and fields["asgif"]["value"]=="play-gif":
		D="3D"

	thresh=fields["thresh"]["getter"]("thresh")/100
	print(pr,lastrun,ss2t,ss2f,thresh)
	if lastrun=="simult":
		fileOut=genContour2D(ss2f,paramRanges=pr,settables={"mode":ss2t}) # generateHeatmap accepts a LIST of files, which it just loops through
		displayContour2D(fileOut,plotting="save",threshold=thresh) # list -> generateHeatmap -> list -> displayHeatmap also accepts a list (and just loops through)
		#ax,fig=getContObjs()
		#updatePlot("runContour2D",fig,ax)
	else:
		if D=="2D":
			fileOut=genContour2D(files[0],paramRanges=pr)
		else:
			fileOut=genContour3D(files[0],paramRanges=pr)

		if D=="3D":
		#if len(getVar("tofit"))==3 and fields["asgif"]["value"]=="play-gif":
			#for i in range(100):
			#	print(i)
				#setVar("altFnames",["guigif/gui"+str(i)+".png"])
			fig,ax=displayContour3D(fileOut,plotting="save",projected=True,elevAzi=[36,i*3.6])
				#liveplot["plot"],liveplot["fig"]=fig,ax
				#img.config(file="guigif/gui"+str(i)+".png")
			updatePlot("runContour2D",fig,ax)
		#		window.update()
		#		frameR.update()
		#	for i in range(100):
		#		img.config(file="guigif/gui"+str(i)+".png")
		#		frameR.update()
				#time.sleep(.03)
			#setVar("altFnames",altFnames)
		else:
			globstr=files[0].split("/")[:-1] + ["gui.py_","contours","*.txt"] ; globstr="/".join(globstr)
			candidates=glob.glob(globstr)
			#print(candidates)
			bonusCurveFiles=[ f for f in candidates if files[0].split("/")[-1].replace(".txt","") in f ]
			print("bonusCurveFiles",bonusCurveFiles,globstr,files[0])
			displayContour2D(fileOut,plotting="save",bonusCurveFiles=bonusCurveFiles,threshold=thresh)
			# OOPS, TODO, bonusCurveFiles wasn't re-implemented when we rewrote diplayHeatmap!

@wrapper
def runPerturbing(event):
	perturb=fields["pertParam"]["getter"]("pertParam")
	perturbBy=fields["perturbBy"]["getter"]("perturbBy")
	if perturb=="all":
		perturb=""
	else:
		perturb=perturb.split(",")
	perturbBy=perturbBy.split(",") ; perturbBy=[float(pb) for pb in perturbBy]

	if lastrun=="simult":
		solveFunc={"func":ss2,"kwargs":{"listOfTypes":ss2t}} ; loopOver=[ss2f]
	else:
		solveFunc={"func":solve,"kwargs":{}} ; loopOver=files
	r,e=[],[]
	for f in loopOver:
		s,u,params=perturbUncertainty(f,paramsToPerturb=perturb,perturbBy=perturbBy,plotting="save",solveFunc=solveFunc) #,paramsToPerturb=paramsToPerturb,perturbBy=perturbBy)
		#print("s,u,params",s,u,params)
		for P,dP,dR in params:
			resultString="perturb "+P+" by "+str(dP)+"% --> "+",".join( ["d"+p+"="+sigFigs(v,4) for p,v in zip(getVar("tofit"),dR) ] )
			out(resultString)
	#		print(P,dP,dR)
		resultString=" , ".join([p+" = "+sigFigs(v*getScaleUnits(p)[0])+"+/-"+sigFigs(dv*getScaleUnits(p)[0])+" "+getScaleUnits(p)[1] for p,v,dv in zip(getVar("tofit"),s,u) ])
		out(resultString)
		r.append(s) ; e.append(u)

	if len(files)>1:
		r=np.mean(r,axis=0) ; e=np.mean(e,axis=0)
		resultString=" , ".join([p+" = "+sigFigs(v*getScaleUnits(p)[0])+"+/-"+sigFigs(dv)+" "+getScaleUnits(p)[1] for p,v,dv in zip(getVar("tofit"),r,e) ])
		out("averaged:")
		out(resultString)	
	
lastMat=""
@wrapper
def runMatImport(event):
	files=ask(multiple=False)
	if len(files)==0:
		return
	importMatrix(str(files[0]))
	te_tprops.delete("1.0", tk.END)
	te_tprops.insert("1.0",l2s2D(getVar("tp")))
	global lastMat ; lastMat=files[0]

@wrapper
def viewMap(event,rerun=False):
	global files,lastrun,lastPlot ; lastrun="map"
	if not rerun:
		files=ask(multiple=True)
	if len(files)==0:
		return

	Amin,Amax=[ float(v) for v in fields["auxRange"]["value"].split(",") ]
	Zmin,Zmax=[ float(v) for v in fields["mapRange"]["value"].split(",") ]

	if files[0].split(".")[-1]=="mat":
		posXs,posYs,pumpMs,probeMs,aux1=readMap(files[0])
		print(np.shape(aux1))
		aux1[aux1<Amin]=np.nan ; aux1[aux1>Amax]=np.nan

		mapMode=fields["mapMode"]["value"]
		if mapMode=="dR/R":
			Zs=probeMs/pumpMs/aux1*100
		elif mapMode=="Aux":
			Zs=aux1
		elif mapMode=="fitted":
			I,J=np.shape(aux1) ; Zs=np.zeros((I,J)) ; Zs[:,:]=np.nan
			for i in range(I):
				for j in range(J):
					x,y=probeMs[i,j]/aux1[i,j],pumpMs[i,j]
					if np.isnan(x):
						continue
					x=np.asarray([0,x]) ; y=np.asarray([0,y])
					#print(x,y)
					r,e=solveSSTR(P=x,M=y,plotting="none")
					Zs[i,j]=r[0]
			#Z=[ [ v for v in row ] for row in probeMs/pumpMs/aux1 ]
			print(Zs)
		#Zs={"dR/R":probeMs/pumpMs/aux1*100,"Aux":aux1}[]
		title=fields["mapMode"]["value"]
		Xs=posXs[0,:] ; Ys=posYs[:,0] ; xlb="x (um)" ; ylb="y (um)" ; title=mapMode+" - "+getVar("fitting")
	elif len(files)>0:
	# 	THIS IS WHERE WE MIGHT ADD IN PLOTTING OF VARIABLE-WAVELENGTH DATA??	
		Zs=[] ; wavelengths=[]
		for f in files:
			print(f)
			ts,data=readTDTR(f)
			Zs.append(data) 
			wavelengths.append(getVar("wavelength"))
		Xs=ts ; Ys=wavelengths ; Zs=np.asarray(Zs) ; xlb="time delay (s)" ; ylb="wavelength (um)" ; title=getVar("fitting")

	
	print(Zs)
	Zs[Zs<Zmin]=np.nan ; Zs[Zs>Zmax]=np.nan



	if fields["neighborify"]["value"]=="yes":				# look for nans, and fill them in using neighboring non-nan values
		sx,sy=np.shape(Zs)
		while True:
			nans=np.where(np.isnan(Zs))				# find nans
			if len(nans[0])==0 or len(nans[0])==len(Zs.flat): 	# if no nans, or all nans, quit
				break
			for i,j in zip(*nans): #[[x1,x2,x3],[y1,y2,y3]]		# for all pairs of coordinates for where nans are found...
				for ii,jj in [[-1,-1],[1,-1],[-1,1],[1,1]]:	# for each diagonal from a nan...
					if i+ii==0 or i+ii==sx or j+jj==0 or j+jj==sy: # (skip this diagonal if it's out of bounds)
						continue
					if not np.isnan(Zs[i+ii,j+jj]):		# and if this diagonal is not a nan, use that value
						Zs[i,j]=Zs[i+ii,j+jj]
						break				# non-nan diagonal was found, so quit
				else:
					continue
				break						# if loop finishes prematurely, break again, out second loop

	#plotHeatmapContour(Zs, posXs, posYs, xlabel="x (um)", ylabel="y (um)", title=mapMode, # GENERAL ARGS
	#	colorBounds="auto",heatColor="inferno", # HEATMAP ARGS: "none", "auto", or a list of bounds as [lower,upper,nticks]
	#	levels="none", styles='' , widths=1, colors="black" , alpha=1 , linelabels=False , useLast=False)
	#print(posXs,posYs)
	sy,sx=np.shape(Zs)
	#print(sx,sy,Zs)
	print(np.shape(Zs),np.shape(Xs),np.shape(Ys))
	if sy<sx/10 or sy==1:
		Zs=np.nansum(Zs,axis=0) # nans will mess up our summing!
		lplot([Xs[0]],[Zs],xlabel="position (um)",ylabel=mapMode,filename="gui.png",title="",labels=[""],xlim=["nonzero"]) ; lastPlot="plot"
	else:
		lcontour(Zs,Xs,Ys,heatOrContour="heat",xlabel=xlb,ylabel=ylb,title=title) ; lastPlot="contour"
		#lcontour(Zs,Xs,Ys,heatOrContour="heat",xlabel=xlb,ylabel=ylb,title=title,filename=files[0].replace(".mat",".png"))
	
@wrapper
def steadyFreq(event): # secret function, generated Jeff's Fig 2 from
	materials=["","Al2O3","Quartz","Si","SiO2"]
	freqs=np.logspace(-1,7,1000)
	Ms=[]
	for mat in materials:
		if len(mat)>0:
			importMatrix("testscripts/calmats/Fiber_"+mat+"_cal_matrix.txt")
		popGlos()
		Z=delTomega(2*np.pi*freqs)
		M=np.sqrt(Z.real**2+Z.imag**2)
		Ms.append(M/np.amax(M))
	fm=getParam("fm") ; i=np.argmin(abs(freqs-fm)) ; M2=Ms[0][i]
	freqs=[freqs]*len(Ms) ; mkrs=["k-","r:","g:","b:","k:","r."] ; dlbs=materials ; dlbs[0]="entered" ; dlbs.append("")
	Ms.append([M2]) ; freqs.append([fm])
	print("gui > steadyFreq",Ms,freqs)
	lplot(freqs,Ms,xlabel="frequency (Hz)",ylabel="M (-)",title="RSI 90, 024905, Fig. 2",markers=mkrs,labels=dlbs,xscale="log",filename="gui.png")

@wrapper
def getRadii(event):
	radiiFile=ask(multiple=False)[0]
	try:
		importRadii(radiiFile)
	except:
		try:
			solve(radiiFile)
		except:
			pass
	
	for glo in ["rpu","rpr","gamma"]:
		r=getParam(glo)
		if "r" in glo:
			r*=1e6
		fields[glo]["field"].delete(0, tk.END)
		fields[glo]["field"].insert(0,str(r))

@wrapper
def refresh(event):
	return

@wrapper
def simult(event,rerun=False):
	global lastrun ; lastrun="simult"
	if rerun:
		r,e=ss2(ss2f,ss2t,plotting="save")
		#r,e=ss3(ss2f,ss2t,plotting="save")
		out(str((r,e)))
		global lastSolution ; lastSolution=r
		return
	simultRunning=True
	newWin=tk.Toplevel(window)
	newWin.title("super simultaneous") #; newWin.minsize(500,10)
	global files
	files=[]
	field1=tk.Entry(master=newWin,width=100)
	field1.pack()
	def addFile(event):
		global files
		newfiles=list(tk.filedialog.askopenfilenames())
		previousfiles=list(field1.get().split(","))
		files=previousfiles+newfiles
		files= [ f for f in files if len(f)>1 ]	

		#field1.insert(tk.END,",".join(fs))
		field1.delete(0,tk.END) ; field1.insert(0,",".join(files))
	bu1=tk.Button(master=newWin,text="add file")
	bu1.bind("<Button-1>",addFile)
	bu1.pack()
	lb=tk.Label(master=newWin,text="Enter types:")
	lb.pack()
	field2=tk.Entry(master=newWin)
	field2.pack()
	bu2=tk.Button(master=newWin,text="[OK]")
	def onclick(event):
		global simultRunning
		files=field1.get().split(",")
		types=field2.get().split(",")
		types=types+[types[0]]*(len(files)-len(types))
		types=[ t.strip() for t in types ]
		types=[ {"S":"SSTR","T":"TDTR","F":"FDTR","P":"PWA"}[t[0].upper()] for t in types]
		print(types)
		newWin.destroy()
		global ss2f,ss2t ; ss2f=files ; ss2t=types
		log("files:"+str(files)+","+str(types))
		r,e=ss2(files,types,plotting="save")
		#r,e=ss3(files,types,plotting="save")
		if getVar("autoFailed"):
			out("WARNING: auto rpu/rpr/fm failed. check your file headers and/or radii.txt file! or change auto to \"no\" and set the values yourself")
		out(str(r)+","+str(e)) # TODO we had a sneaky bug here, where this line crashed with "out(str(r,e))", and we never noticed, because i guess the newWin process crashed, not the main, or something like that (onclick just ended, no problem!) we only noticed because lastSolution wasn't correctly populated. could there be other stuff like this?
		global lastSolution ; lastSolution=r
		simultRunning=False
	bu2.bind("<Button-1>",onclick)
	bu2.pack()

@wrapper
def predUnc(event):
	thresh=fields["thresh"]["getter"]("thresh")
	predictUncert(threshold=thresh/100,settables={})	

@wrapper
def knife(event):
	files=ask() ; direc="/".join(files[0].split("/")[:-1])
	out(str(knifeAll(direc)))

@wrapper
def fibercals(event):
	# FIBER-SPECIFIC SETUP (mode, what are we fitting for, assumptions, etc)
	setVar("mode","SSTR") ; setVar("tofit",["rpu","gamma"]) ; setParam("rpr",1.5e-6) ; setVar("autorpu",False)
	# QUERY THE USER FOR DIRECTORIES
	out("first select data folder, then select calmats folder")
	fileDirec=ask(fileOrDirec="direc",text="select folder with data files") #; print(files) ; fileDirec="/".join(files[0].split("/")[:-1])
	calmatDirec=ask(fileOrDirec="direc",text="select calmats folder") #; print(files) ; calmatDirec="/".join(files[0].split("/")[:-1])
	# ACTUALLY PERFORM THE FITTING, USING GENERALIZED calsForSpots FUNCTION
	r,e,matDict=calsForSpots(fileDirec,calmatDirec)
	# PRINT THE RESULTS TO THE USER
	outString="rpu="+str(np.round(r[0]*1e6,2))+"um, gamma="+str(np.round(r[1]))+", rpr="+str(getParam("rpr")*1e6)+"um (assumed)"
	out(outString)
	# LETS ALSO UPDATE THE FIELDS WITHIN THE GUI! 
	glos   =[ "rpu"                ,"rpr", "gamma"       ]
	newvals=[ np.round(r[0]*1e6,2) , 1.5 , np.round(r[1])]
	for glo,val in zip(glos,newvals):
		fields[glo]["field"].delete(0, tk.END)
		fields[glo]["field"].insert(0,str(val))
	en_tofit.delete(0,tk.END)
	en_tofit.insert(0,"Kz2")
	fields["mode"]["field"].set("SSTR") # need to do this so runSolve() works! 
	updatePlot("fibercals")
	# LETS ALSO CYCLE THROUGH MATERIALS AND FIT FOR ALL (some of this rote copied from calsForSpots)
	#setVar("tofit",["Kz2"])
	global files
	for mat in matDict.keys():
		matfile=matDict[mat]["matfile"] ; files=matDict[mat]["files"]
		importMatrix(matfile) #; setVar("tofit",["Kz2"])
		te_tprops.delete("1.0", tk.END)
		tp=str(getVar("tp")).replace("[","").replace("],","\n").replace("]","").replace("'","")
		te_tprops.insert("1.0",tp)
		runSolve(event,rerun=True)

@wrapper
def calphase(event):
	files=ask() ; print(files) ; fileDirec="/".join(files[0].split("/")[:-1])
	files=ask() ; print(files) ; calmatDirec="/".join(files[0].split("/")[:-1])
	calsForPhase(fileDirec,calmatDirec)

@wrapper
def pico(event):
	files=ask()
	out(str(picoAcoustics(files[0])))

@wrapper
def runTerm(event):						# allows the user to type code directly into the results box. enter an arrow (">"), then
	term=te_res.get("1.0", tk.END)				# type your code, then press "ctrl+e" toexecute it. code may include multiple lines (and 
	term=term.split("\n")					# ebeware, imports don't persist between xecutions of runTerm). 
	commands=[]						# For example, we can run checkTechnique via:
	for row in reversed(term):				# ">from canIMeasureThat import checkTechnique"
		if len(row)>5:					# ">checkTechnique("mfSSTR")				
			if row[0]==">":
				commands.append(row[1:])
			else:
				break
	commands=list(reversed(commands))
	print(commands)
	for c in commands:
		print("executing",c)
		exec(c)

@wrapper
def runMonte(event):
	global lastPlot ; lastPlot="contour"
	pr=[[v*.05,v*3] for v in lastSolution]
	results=monteCarlo(files[0],N=300,returnFull=True)
	# monteCarlo(returnFull=True) returns a list of (r,e) pairs from solve. [([fittedParam1,fittedParam2],[residual,[cor1,cor2]]),(...)...]
	tf=getVar("tofit")
	xfact=getScaleUnits(tf[0])[0] ; yfact=getScaleUnits(tf[1])[0]
	xs=[ r[0][0]*xfact for r in results ]
	ys=[ r[0][1]*yfact for r in results ]
	residual=[ r[1][0] for r in results ]
	for x,y,r in zip(xs,ys,residual):
		print(tf[0],x,tf[1],y,r*100,"%")
	fileOut=genContour2D(files[0],paramRanges=pr)
	displayContour2D(fileOut,plotting="save",bonusXY=[xs,ys,residual])
	out(tf[0]+"="+str(np.round(np.mean(xs),2))+"+/-"+str(np.round(np.std(xs),2))+" "+unitsDict[tf[0][:-1]]+", "+tf[1]+"="+str(np.round(np.mean(ys),2))+"+/-"+str(np.round(np.std(ys),2))+" "+unitsDict[tf[1][:-1]])
	#lplot([xs],[ys],useLast=True)

@wrapper
def whichTechnique(event):
	thresh=fields["thresh"]["getter"]("thresh")
	ranges=whichTechniqueShouldIUse(thresh/100)
	dr={ k:ranges[k][1]-ranges[k][0] for k in ranges.keys() }


def help(event):
	out("Hidden options:\nCtrl+j = Jeff's Fig 2\nCtrl+s = import spot sizes from radii.txt\nCtrl+r = refresh\nCtrl+m = multi-technique fitting\nCtrl+u = contour uncertainty simulation\nCtrl+k = knife-edge (whole folder)\nCtrl+p = picosecond acoustics")

def fittedResult(filename,tofit='',values=''):
	date=filename.split("/")[-1].split("_")[0]
	date=list(date) ; date.insert(4,"/") ; date.insert(2,"/") ; date="".join(date)
	params={"date":date}
	for location in ["HQ","PLSB","Fiber","HQ-SSTR"]:
		if location in filename:
			params["where"]=location
	#params["technique"]=getVar("mode")
	tofit=list(tofit) ; values=list(values)
	for p,v in zip(tofit,values):
		params[p]=v
	params["matfile"]=lastMat
	if filename in loggableResults.keys(): # WE RAN BEFORE, AND FOUND THIS BEFORE
		FR=loggableResults[filename]
		for k in FR.keys():
			if k not in params.keys():
				params[k]=FR[k]
	return params



def logToGdocs(event): # see testing53 for google sheets api append
	# WHAT GETS LOGGED? date and location, for starters
	# for TDTR, we use K_sapph, G_sapph, K_siO2, G_SiO2
	# for SSTR, we record gamma, and then conductivity for each cal
	columnsTDTR=["date","user","Kz2_Al2O3","G1_Al2O3","Kz2_SiO2","G1_SiO2"]
	columnsSSTR=["date","user","gamma","Kz2_SiO2","Kz2_Al2O3","Kz2_Quartz","Kz2_Si"]
	# notice the little tricky business involved with gamma for SSTR? the user will use one (or more) of the samples to record gamma. which means from a data structures perspective, we need to make sure we store off both (don't overwrite your gamma fit when you d your K fit). 

	# Step 1, use the last set of fitted files (global variable "files") to infer the date and location. and use "mode" (in TDTR_fitting) for cols
	FR=fittedResult(files[-1])
	where=FR["where"] ; columns={"TDTR":columnsTDTR,"SSTR":columnsSSTR}[getVar("mode")] ; date=FR["date"]
	row=[date,""]+[ [] for i in range(len(columns[2:])) ]
	for i,tofit in enumerate(columns[2:]):		# for each reportable parameter (eg, "K_Al2O3")
		for FR in loggableResults.values():	# and for each file we've fitted so far (each entry is the dict output from fittedResult)
			if "_" in tofit:
				p,mat=tofit.split("_")	# "K_Al2O3" --> "K","Al2O3"
				if mat+"_" not in FR["matfile"]: # why the dopey underscore? because "Si" matches on "SiO2" otherwise
					continue # # if material doesn't match the one we're looking for, skip this file. 
			else:
				p=tofit 		# could also be looking for "gamma"
			if FR["where"]!=where:		# if this scan was not taken in the place we're looking for (skip)
				continue
			if FR["date"]!=date:		# or on the date we're looking for 
				continue
			#paramFitted=[ p in k for k in FR.keys() ] # check if string "Kz" in any of the keys (eg, "G1", "Kz2", "date", and so on)
			#if True not in paramFitted:
			#	continue
			if p not in FR.keys():
				continue
			# THIS IS SUCCESS, ADD THE VALUE TO THE ROW
			row[i+2].append(FR[p])
	for i in range(2,len(row)):
		vals=row[i]
		if len(vals)==0:
			row[i]=""
		else:
			row[i]=scientificNotation(np.mean(vals),3)
	print(row)
	logWindow(columns,row,where)

window.bind('<Control-l>',logToGdocs)

def logWindow(colNames,logRow,where):
	newWin=tk.Toplevel(window)
	newWin.title("GDOCS LOGGING") #; newWin.minsize(500,10)
	bu=tk.Button(master=newWin,text="submit")
	bu.grid(row=0,column=0,sticky='ew')
	entries=[]
	for i in range(len(colNames)):
		lb=tk.Label(master=newWin,text=colNames[i])
		lb.grid(row=0,column=i*2+1,sticky='ew') #; print(i*2+1)
		en=tk.Entry(master=newWin)
		en.delete(0,tk.END)
		en.insert(0,str(logRow[i]))
		en.grid(row=0,column=i*2+2,sticky='ew') #; print(i*2+2)
		entries.append(en)
	for i in range(2*len(colNames)+1):
		newWin.columnconfigure(i,weight=1,uniform="logWin")
	def onclick(event):
		submitRow(where, [ en.get() for en in entries ] )
		newWin.destroy()
	bu.bind("<Button-1>",onclick)

def submitRow(sheet,values):
	# https://developers.google.com/sheets/api/quickstart/python
	# NEED TO CREATE A PROJECT, AND NEED TO ENABLE TO API FOR THE PROJECT
	import os.path
	from google.auth.transport.requests import Request
	from google.oauth2.credentials import Credentials
	from google_auth_oauthlib.flow import InstalledAppFlow
	from googleapiclient.discovery import build
	from googleapiclient.errors import HttpError

	# If modifying these scopes, delete the file token.json.
	SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

	# The ID and range of a sample spreadsheet.
	SAMPLE_SPREADSHEET_ID = '1ojtljq_zVofIUZzwx_Bp1rvbnfDOqkqR9ugxcQjvw_8' # https://docs.google.com/spreadsheets/d/1ojtljq_zVofIUZzwx_Bp1rvbnfDOqkqR9ugxcQjvw_8/edit?usp=sharing
	SAMPLE_RANGE_NAME = sheet+"!"+"A2:"+"ABCDEFGHIJKLMNOPQRSTUVWXYZ"[len(values)-1] # eg, "HQ!A2:F" 

	if os.path.exists('token.json'):
		creds = Credentials.from_authorized_user_file('token.json', SCOPES)
	else:
		flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
		creds = flow.run_local_server(port=0)
		# Save the credentials for the next run
		with open('token.json', 'w') as token:
			token.write(creds.to_json())

	service = build('sheets', 'v4', credentials=creds)

	service.spreadsheets().values().append( # https://stackoverflow.com/questions/46274040/append-a-list-in-google-sheet-from-python
		spreadsheetId=SAMPLE_SPREADSHEET_ID ,
		range=SAMPLE_RANGE_NAME,
		body={ "majorDimension": "ROWS",
			"values": [values] },
		valueInputOption="USER_ENTERED"
	).execute()

#@wrapper
def runNanny(event):
	warnings=nanny()
	if len(warnings)==0:
		out("all checks passed")
	for w in warnings:
		out(w)
	

# BUTTONS EACH GET A TITLE AND A FUNCTION, AND WE ADD THEM IN ORDER TO A SINGLE ROW IN LEFT TOP
buttonTitles=["Import Vals", "Fit Data", "Perturb Unc.", "Fast Contour", "Contours2D" ,"T(r,z)" ,"Sensitivity", "View Map", "(refit)", 
	"(phase)" , "avg files", "fibercals"]
buttonFuncs=[ runMatImport ,  runSolve ,  runPerturbing,  runContour   , runContour2D , runTRZ  , runSens     ,  viewMap  ,  refit   , 
	checkPhase, avgFiles   , fibercals ]
for t,f in zip(buttonTitles,buttonFuncs):
	r,c=getRC("BU")
	bu=tk.Button(master=frames["BU"],text=t,width=cellWidth-2)
	bu.bind("<Button-1>", f)	# "<Button-1>" is for left click
	bu.grid(row=r, column=c)

#https://stackoverflow.com/questions/16082243/how-to-bind-ctrl-in-python-tkinter
window.bind('<Control-j>',steadyFreq)
window.bind('<Control-s>',getRadii)
window.bind('<Control-r>',refresh)
window.bind('<Control-m>',simult)
window.bind('<Control-u>',predUnc)
window.bind('<Control-h>',help)
window.bind('<Control-e>',runTerm)
window.bind('<Control-k>',knife)
window.bind('<Control-p>',pico)
window.bind('<Control-d>',checkDelta)
window.bind('<Control-M>',runMonte)
window.bind('<Control-f>',calphase)
window.bind('<Control-w>',whichTechnique)
window.bind('<Control-q>',runNanny)

done=False
@wrapper # why wrap quit_me? because we want to store off the log of settings at the end! 
def quit_me():				# weird thing, when we generate a matplotlib plot, tkinter main loop doesn't exit when we click the x.
	#log("Quitting")		# to deal with that, we detect a "delete window" and then use that to quit.
	window.quit()			# https://stackoverflow.com/questions/55201199/the-python-program-is-not-ending-when-tkinter-window-is-closed
	window.destroy()
	global done
	done=True
window.protocol("WM_DELETE_WINDOW", quit_me)

window.mainloop() # run the tkinter event loop (displays the window object)
# last step, we'll set all frames on the left side to have the same width:


