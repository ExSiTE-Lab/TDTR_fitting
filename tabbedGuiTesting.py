import tkinter as tk
from tkinter import *
from tkinter import ttk
#import sys ; sys.path.insert(1,"../")
from TDTR_fitting import *

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
matplotlib.use("TkAgg")

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

# 2D MATRIX DENOTES WHERE THINGS GO:
# "button;label;functionToRun"
# "drop;label;options;globalName;[optionalCustomFormatFunc]"
# "entry;label;globalName;[optionalCustomFormatFunc]"
# "text;label;globalName;[optionalCustomFormatFunc]"
# REPEAT ELEMENTS MEAN EXPAND OVER ELEMENTS
tpHeader="Thermal properties: C (J/m3/K) , Kz (W/m/K) , d (m) , Kr (W/m/K)"
elements_TDTR=[
	[ "button;Fit Data;runSolve"    , "button;Sensitivity;runSens"         ,            ""                ,              ""                  ],
	[ "text;"+tpHeader+";tp;formatTP" ,              ""                   ,            ""                ,              ""                  ], 
	[              ""               ,                 ""                   ,            ""                ,              ""                  ],
	[              ""               ,                 ""                   ,            ""                , "text;"+tpHeader+";tp;formatTP" ],
	[ "entry;pump rad (um);rpu;convert2um" , "entry;probe rad (um);rpr;convert2um" , "drop;R/M/X/Y;R,M,X,Y;fitting" , "entry;fmod (Hz);fm"   ]]

elements_SSTR=[
	[ "button;Fit Data;runSolve"    , "button;Sensitivity;runSens"         ,            ""                ,              ""                  ],
	[ "text;"+tpHeader+";tp;formatTP" ,              ""                   ,            ""                ,              ""                  ], 
	[              ""               ,                 ""                   ,            ""                ,              ""                  ],
	[              ""               ,                 ""                   ,            ""                , "text;"+tpHeader+";tp;formatTP" ],
	[ "entry;pump rad (um);rpu;convert2um" , "entry;probe rad (um);rpr;convert2um"  ,  "entry;gamma;gamma" ,    "entry;fmod (Hz);fm"         ]]



# SETTING UP THE GUI AND PLACING THINGS:
window=Tk() ; window.title("TDTR fitting!")
# left vs right panels
frameL=Frame(master=window) ; frameL.grid(row=0,column=0,sticky="NSEW")
frameR=Frame(master=window) ; frameR.grid(row=0,column=1,sticky="NSEW")
# top vs bottom panels on the right
framePlot=Frame(master=frameR) ; framePlot.grid(row=0,column=0,sticky="NSEW")
frameResu=Frame(master=frameR) ; frameResu.grid(row=1,column=0,sticky="NSEW")
# 2:3 ratio of width for buttons vs plot panel
window.columnconfigure(0,weight=2,uniform="window") ; window.columnconfigure(1,weight=3,uniform="window")
# 3:1 ratio of height for plot vs results panel
frameR.rowconfigure(0,weight=3,uniform="frameR") ; frameR.rowconfigure(1,weight=1,uniform="frameR")
# tabs in the buttons panel
tabControl = ttk.Notebook(frameL) ; tabs={} ; tabTitles=["TDTR","SSTR","FDTR","PWA","multifitting"]
# results panel
lb_res=tk.Label(master=frameResu,text="RESULTS:")
te_res=tk.Text(master=frameResu,height=7)				# text entry field object
lb_res.pack() ; te_res.pack()					# add both objects to the window

for tabTitle in tabTitles:
	tabs[tabTitle]=ttk.Frame(tabControl)
	tabControl.add(tabs[tabTitle],text=tabTitle)
	tabControl.pack(expand=1, fill="both")


# UNCOMMENT THESE TO DRAW COLOR-CODED BORDERS AROUND EACH FRAME (EG, TO CHECK THAT GRID ELEMENTS EXPAND APPROPRIATELY)
colors=["red","orange","yellow","green","blue","purple","black"]*10
for i,frame in enumerate([frameL,frameR,framePlot,frameResu]):
		frame.configure(highlightbackground=colors[i],highlightthickness=10)

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
		nrows,ncols=len(elementsList),len(elementsList[0])
		processedElements=[]	# don't repeat elements (if elements span multiple rows or columns, and are thus entered twice)
		master=tabs[tabKey]	# all objects will be added to this tab
		globalLookup[tabKey]={}	# this links global names (from TDTR_fitting) to tk objects
		globalSpecialHandling[tabKey]={}
		for row in range(nrows):
			for col in range(ncols):
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
				packkwargs={"row":row*2,"column":col,"rowspan":rowspan*2,"columnspan":columnspan,"sticky":"NSEW"}
				if elementType=="button":
					button=tk.Button(master=master,text=elementLabel)# button object
					funcName=elementString.split(";")[2]
					func=globals()[funcName]
					button.bind("<Button-1>", func)			# link the dummy function to the button
					button.grid(**packkwargs)
				if elementType in ["drop","entry","text"]:
					label=tk.Label(master=master,text=elementLabel)
					packkwargs["rowspan"]=1 ; label.grid(**packkwargs)# label goes on first row (even if entry field spans multiple)
					packkwargs["rowspan"]=rowspan*2-1 ; packkwargs["row"]+=1# prepare to put drop or entry objects on next line
				if elementType=="drop":
					options=elementString.split(";")[2].split(",")
					val=options[0]
					opt=tk.StringVar(window)			# selected option is stored in a stringVar object
					dropdown = tk.OptionMenu(master, opt, *options)	# dropdown menu object
					dropdown.grid(**packkwargs)
					globalName=elementString.split(";")[3]
					specialFormatHandler=""
					if len(elementString.split(";"))==5:
						specialFormatHandler=elementString.split(";")[4]
				if elementType=="entry":
					opt=tk.StringVar(window)			# value for entry field goes in a stringVar object
					field=tk.Entry(master=master,textvariable=opt)	# text entry field object
					field.grid(**packkwargs)
					elementString.split(";")[2]
					globalName=elementString.split(";")[2]
					specialFormatHandler=""
					if len(elementString.split(";"))==4:
						specialFormatHandler=elementString.split(";")[3]
				if elementType=="text":
					opt=tk.Text(master=master,height=rowspan*2-1)
					opt.grid(**packkwargs)
					# https://stackoverflow.com/questions/13079299/dynamically-adding-methods-to-a-class
					createSetterGetter(opt)
					elementString.split(";")[2]
					globalName=elementString.split(";")[2]
					specialFormatHandler=""
					if len(elementString.split(";"))==4:
						specialFormatHandler=elementString.split(";")[3]
				if elementType in ["drop","entry","text"]:
					globalLookup[tabKey][globalName]=opt	# this links global names (from TDTR_fitting) to tk objects
					if len(specialFormatHandler)>1:
						globalSpecialHandling[tabKey][globalName]=specialFormatHandler

				print(elementString,packkwargs)
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
def convert2um(val,whichWay="format"):
	if whichWay=="format":
		return str(np.round(val*1e6,2))
	else:
		return float(val)*1e-6
# this links global names (from TDTR_fitting) to tk objects
globalLookup={}			# tdtrGloName:guiEntryOrDropdownObj
globalSpecialHandling={}	# tdtrGloName:customConverterFunc
def updateAllFieldsFromGlobals():
	for tab in globalLookup.keys():
		for glo in globalLookup[tab].keys():				# e.g. "rpu" as a string
			val=getVar(glo)						# retrieve value from TDTR_fitting, e.g. 5e-6 (as a float)
			if glo in globalSpecialHandling[tab].keys():		# e.g. "rpu" linked to special function "convert2um"
				fun=globals()[globalSpecialHandling[tab][glo]]	# lookup func from string name for func
				val=fun(val,"format")				# pass value to special function
			else:
				val=str(val)					# OR, just convert to string if no special function
			globalLookup[tab][glo].set(val)				# and set the corrosponding gui entry object
def updateAllGlobalsFromFields(tab):
	for glo in globalLookup[tab].keys():					# e.g. "rpu" as a string
		val=globalLookup[tab][glo].get()				# get gui entry object's value (will be a string)
		if glo in globalSpecialHandling[tab].keys():			# e.g. "rpu" linked special function "convert2um"...
			fun=globals()[globalSpecialHandling[tab][glo]]	# ...which should have unformat option to go back (scale, unscale, etc)
			val=fun(val,"unformat")
		else:
			oldval=getVar(glo)				# OR, retrieve existing value
			val=type(oldval)(val)				# and use it's type to convert from string back to the correct type
		print("updateAllGlobalsFromFields",glo,"-->",val)
		setVar(glo,val)

# WRAPPER FUNCTIONS FOR TDTR_fitting.py FUNCTIONS.

# each function also get it's own wrapper. when any button is pressed, before calling anything, we pass TDTR_fitting various text-enterable values, and after, we do some logging and plotting
def wrapper(func): # https://www.geeksforgeeks.org/function-wrappers-in-python/
	def wrapped(*args,**kwargs):
		tabIndex = tabControl.index("current")
		tabName=tabTitles[tabIndex] # print("tabName",tabName)
		updateAllGlobalsFromFields(tabName)
		if tabName in ["TDTR","FDTR","SSTR","PWA"]:
			setParam("mode",tabName)
		func(*args,**kwargs)
		updatePlot(str(func))
	return wrapped
@wrapper
def runSolve(event):
	print("RUN SOLVE!")

@wrapper
def runSens(event):
	print("RUN SENS")
	sensitivity()

# borrow the plot object which TDTR_fitting > niceplot generated, and display them
setVar("fignames",["gui.png","gui.svg","gui.csv"]) # required to suppress free-floating plot! wild! 
from matplotlib.figure import Figure
from niceplot import getPlotObjs ; from nicecontour import getContObjs
funcsUseContours=["runTRZ","viewMap","runMonte","runContour2D"]
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

def printToResultsPanel(printstring):
	te_res.insert(tk.END,printstring+"\n")	# write to results Text field, including a new line
	te_res.see("end")			# and scroll the text entry field to the bottom
	frameR.update()				# update the frame containing the text field
	log("[output] : "+printstring)


		

processElementLists(["TDTR","SSTR"],[elements_TDTR,elements_SSTR])


def quit_me():				# weird thing, when we generate a matplotlib plot, tkinter main loop doesn't exit when we click the x.
	#log("Quitting")		# to deal with that, we detect a "delete window" and then use that to quit.
	window.quit()			# https://stackoverflow.com/questions/55201199/the-python-program-is-not-ending-when-tkinter-window-is-closed
	window.destroy()
	global done
	done=True
window.protocol("WM_DELETE_WINDOW", quit_me)



window.mainloop()

# https://stackoverflow.com/questions/14000944/finding-the-currently-selected-tab-of-ttk-notebook
# https://stackoverflow.com/questions/16373887/how-to-set-the-text-value-content-of-an-entry-widget-using-a-button-in-tkinter