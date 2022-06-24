import json
from matplotlib.font_manager import json_load
import matplotlib.pyplot as plt
import os
import glob
import itertools
from statistics import median, mean

from numpy import sort

MAP = {0:"dataset_size",1:"rmse",2:"mape",3:"r_square"}

class experimentdata():
    def __init__(self,file) -> None:
        self.jsondata = json_load(file)
        metadata = self.jsondata["metadata"]
        self.timestamp = metadata["timestamp"]
        self.optimisation_func = metadata["optimisation_func"]
        self.sys_name = metadata["sys_name"]
        self.N = metadata["N"]
        self.iterations = metadata["iterations"]
        self.selectiontyp = metadata["selectiontyp"]
        self.seed = metadata["seed"]
        self.stop = metadata["stop"]
    pass


def plotexperiment1(data):
    with open(data, "r") as f:
        data = json.load(f)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        for val,ax in zip((1,2,3),(ax1, ax2, ax3)):
            
            
            activedata = [data["fwscore"]]+data["activescore"]
            minx = activedata[0][0]
            maxx = activedata[~0][0]
            x = [i[0] for i in activedata]
            y = [i[val] for i in activedata]
            ax.plot(x,y)
            ax.axhline(data["pwscore"][val])
            ax.axhline(data["randomscore"][val])
            ax.axvline(data["randomscore"][0])
        plt.savefig(os.path.join("messurments",  "plot.png"))

def plot1(data1,data2,datalabel,sys_name):
    #((ax1,bx1),(ax2,bx2),(ax3,bx3))
    fig, plots   = plt.subplots(3, 1)
    for i,plot in enumerate(plots):
        for data,color,linelabel in zip([data1,data2],['blue','orange'],datalabel):
        
            activedatas = list(itertools.chain(*[[x.jsondata["fwscore"]]+x.jsondata["activescore"] for x in data]))
            
            xs = list(set([x[0] for x in activedatas]))

            #print(xs)
            ys = []
            for x in xs:
                val = []
                for activedata in activedatas:
                    if activedata[0] == x:
                        val+=[activedata[i+1]]
                ys+=[val]
            means = [mean(b) for b in ys]
            medians = [median(b) for b in ys]
            plot.plot(xs,medians,color=color,label=linelabel)
        if i == 2:
            plot.legend(loc="lower right")
        else:
            plot.legend(loc="upper right")
    #fig = plt.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.savefig(os.path.join("plots",  sys_name+"_weighted_unweighted.png"))

def plot1new(systemsa,systemsb,datalabel,sys_names):
    fig, plots   = plt.subplots(len(systemsa), 1)
    for plot,sysa,sysb,sys_name in zip(plots,systemsa,systemsb,sys_names):
        for data,color,linelabel in zip([sysa,sysb],['blue','orange'],datalabel):
            activedatas = list(itertools.chain(*[[x.jsondata["fwscore"]]+x.jsondata["activescore"] for x in data]))
            xs = list(set([x[0] for x in activedatas]))
            ys = []
            for x in xs:
                val=[]
                for activedata in activedatas:
                    if activedata[0] == x:
                        val+=[activedata[2]]
                ys+=[val]
            means = [mean(b) for b in ys]
            medians = [median(b) for b in ys]
            plot.plot(xs,medians,color=color,label=linelabel)
            plot.legend(loc="upper right")
    plt.savefig(os.path.join("plots",  str(sys_names)+"_Ns.png"))    

def plot2(data1,data2,data3,data4,datalabel,sys_name):
    #((ax1,bx1),(ax2,bx2),(ax3,bx3))
    fig, plots   = plt.subplots(3, 1)
    for i,plot in enumerate(plots):
        for data,color,linelabel in zip([data1,data2,data3,data4],['blue','orange','red','green'],datalabel):
        
            activedatas = list(itertools.chain(*[[x.jsondata["fwscore"]]+x.jsondata["activescore"] for x in data]))
            
            xs = list(set([x[0] for x in activedatas]))

            #print(xs)
            ys = []
            for x in xs:
                val = []
                for activedata in activedatas:
                    if activedata[0] == x:
                        val+=[activedata[i+1]]
                ys+=[val]
            means = [mean(b) for b in ys]
            medians = [median(b) for b in ys]
            plot.plot(xs,medians,color=color,label=linelabel)
        if i == 2:
            plot.legend(loc="lower right")
        else:
            plot.legend(loc="upper right")
    #fig = plt.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.savefig(os.path.join("plots",  sys_name+"_Ns.png"))

def plot1new(systemsa,systemsb,datalabel,sys_names):
    fig, plots   = plt.subplots(len(systemsa), 1)
    for plot,sysa,sysb,sys_name in zip(plots,systemsa,systemsb,sys_names):
        for data,color,linelabel in zip([sysa,sysb],['blue','orange'],datalabel):
            activedatas = list(itertools.chain(*[[x.jsondata["fwscore"]]+x.jsondata["activescore"] for x in data]))
            xs = list(set([x[0] for x in activedatas]))
            ys = []
            for x in xs:
                val=[]
                for activedata in activedatas:
                    if activedata[0] == x:
                        val+=[activedata[2]]
                ys+=[val]
            means = [mean(b) for b in ys]
            medians = [median(b) for b in ys]
            plot.plot(xs,medians,color=color,label=linelabel)
            plot.legend(loc="upper right")
    plt.savefig(os.path.join("plots",  str(sys_names)+"_Ns.png"))  

experiments = []
for f in glob.glob("meassurments/**/results.json",recursive=True):
    experiments+=[experimentdata(f)]
#[[LLVM_1,LLVM_2,LLVM_4,LLVM_8,LLVM_16],[x264_1,x264_2,x264_4,x264_8,x264_16],[BerkeleyDBC_1,BerkeleyDBC_2,BerkeleyDBC_4,BerkeleyDBC_8,BerkeleyDBC_16]]
def plot2new(systemss,datalabel,sys_names,plotname = "plot"):
    fig, plots   = plt.subplots(len(systemss), 1)
    for plot,syss,sys_name in zip(plots,systemss,sys_names):
        for data,color,linelabel in zip(syss,['blue','orange','red','green','black'],datalabel):
            for x in data:
                    
                	activedatas = list(itertools.chain(*[[x.jsondata["fwscore"]]+x.jsondata["activescore"] for x in data]))
            xs = list(set([x[0] for x in activedatas]))
            xs=sort(xs)
            ys = []
            for x in xs:
                val=[]
                for activedata in activedatas:
                    if activedata[0] == x:
                        val+=[activedata[2]]
                ys+=[val]
            means = [mean(b) for b in ys]
            medians = [median(b) for b in ys]
            plot.plot(xs,means,color=color,label=linelabel)
            plot.legend(loc="upper right")
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join("plots",  str(sys_names)+"_"+plotname+".png"))  

experiments = []
for f in glob.glob("*/results.json",recursive=True):
    experiments+=[experimentdata(f)]

#filter experiments
x264 = list(filter(lambda e: e.sys_name == "x264" and e.selectiontyp == "featurewise", experiments))
LLVM = list(filter(lambda e: e.sys_name == "LLVM" and e.selectiontyp== "featurewise", experiments))
BerkeleyDBC = list(filter(lambda e: e.sys_name == "BerkeleyDBC"  and e.selectiontyp== "featurewise", experiments))

#LLVM = list(filter(lambda e: e.N > 4, LLVM))

x264_weighted = list(filter(lambda e: e.optimisation_func == "weighted" and e.N< 9, x264))
LLVM_weighted = list(filter(lambda e: e.optimisation_func == "weighted" and e.N< 9, LLVM))
BerkeleyDBC_weighted = list(filter(lambda e: e.optimisation_func == "weighted" and e.N< 9, BerkeleyDBC))
x264_unweighted = list(filter(lambda e: e.optimisation_func == "unweighted" and e.N< 9, x264))
LLVM_unweighted = list(filter(lambda e: e.optimisation_func == "unweighted" and e.N< 9, LLVM))
BerkeleyDBC_unweighted = list(filter(lambda e: e.optimisation_func == "unweighted" and e.N< 9, BerkeleyDBC))

#print(len(x264_unweighted))
#print(len(x264_weighted))

#plot1(LLVM_weighted,LLVM_unweighted,["weighted","unweighted"],"LLVM")
#plot1(x264_weighted,x264_unweighted,["weighted","unweighted"],"x264")
#plot1(BerkeleyDBC_weighted,BerkeleyDBC_unweighted,["weighted","unweighted"],"BerkeleyDBC")
plot2new([[LLVM_weighted,LLVM_unweighted],[x264_weighted,x264_unweighted],[BerkeleyDBC_weighted,BerkeleyDBC_unweighted]],["weighted","unweighted"],["LLVM","x264","BerkeleyDBC"],plotname="weighted vs unweighted")
#plot1new([LLVM_weighted,x264_weighted,BerkeleyDBC_weighted],[LLVM_unweighted,x264_unweighted,BerkeleyDBC_unweighted],["weighted","unweighted"],["LLVM","x264","BerkeleyDBC"])

x264  = list(filter(lambda e: e.optimisation_func == "weighted", x264))
LLVM  = list(filter(lambda e: e.optimisation_func == "weighted", LLVM))
BerkeleyDBC  = list(filter(lambda e: e.optimisation_func == "weighted", BerkeleyDBC))

#print(len(LLVM))
#print(set([e.N for e in x264]))

LLVM_1 = list(filter(lambda e: e.N == 1, LLVM))
LLVM_2 = list(filter(lambda e: e.N == 2, LLVM))
LLVM_4 = list(filter(lambda e: e.N == 4, LLVM))
LLVM_8 = list(filter(lambda e: e.N == 8, LLVM))
LLVM_16 = list(filter(lambda e: e.N == 16, LLVM))
BerkeleyDBC_1 = list(filter(lambda e: e.N == 1, BerkeleyDBC))
BerkeleyDBC_2 = list(filter(lambda e: e.N == 2, BerkeleyDBC))
BerkeleyDBC_4 = list(filter(lambda e: e.N == 4, BerkeleyDBC))
BerkeleyDBC_8 = list(filter(lambda e: e.N == 8, BerkeleyDBC))
BerkeleyDBC_16 = list(filter(lambda e: e.N == 16, BerkeleyDBC))
x264_1 = list(filter(lambda e: e.N == 1, x264))
x264_2 = list(filter(lambda e: e.N == 2, x264))
x264_4 = list(filter(lambda e: e.N == 4, x264))
x264_8 = list(filter(lambda e: e.N == 8, x264))
x264_16 = list(filter(lambda e: e.N == 16, x264))

FWLLVM = LLVM_4
FWBerkeleyDBC = BerkeleyDBC_4
FWx264 = x264_4
#plot2(LLVM_1,LLVM_2,LLVM_4,LLVM_8,["N = 1","N = 2","N = 4","N = 8","N = 16"],"LLVM")
#plot2(x264_1,x264_2,x264_4,x264,["N = 1","N = 2","N = 4","N = 8","N = 16"],"x264")
#plot2(BerkeleyDBC_1,BerkeleyDBC_2,BerkeleyDBC_4,BerkeleyDBC,["N = 1","N = 2","N = 4","N = 8","N = 16"],"BerkeleyDBC")

plot2new([[LLVM_1,LLVM_2,LLVM_4,LLVM_8,LLVM_16],[x264_1,x264_2,x264_4,x264_8,x264_16],[BerkeleyDBC_1,BerkeleyDBC_2,BerkeleyDBC_4,BerkeleyDBC_8,BerkeleyDBC_16]],["N = 1","N = 2","N = 4","N = 8","N = 16"],["LLVM","x264","BerkeleyDBC"],plotname="FWN")


x264 = list(filter(lambda e: e.sys_name == "x264" and e.selectiontyp == "pairwise" and e.optimisation_func == "weighted", experiments))
LLVM = list(filter(lambda e: e.sys_name == "LLVM" and e.selectiontyp== "pairwise" and e.optimisation_func == "weighted", experiments))
BerkeleyDBC = list(filter(lambda e: e.sys_name == "BerkeleyDBC"  and e.selectiontyp== "pairwise" and e.optimisation_func == "weighted", experiments))

LLVM_3 = list(filter(lambda e: e.N == 3, LLVM))
LLVM_4 = list(filter(lambda e: e.N == 4, LLVM))
LLVM_5 = list(filter(lambda e: e.N == 5, LLVM))
BerkeleyDBC_3 = list(filter(lambda e: e.N == 3, BerkeleyDBC))
BerkeleyDBC_4 = list(filter(lambda e: e.N == 4, BerkeleyDBC))
BerkeleyDBC_5 = list(filter(lambda e: e.N == 5, BerkeleyDBC))
x264_3 = list(filter(lambda e: e.N == 3, x264))
x264_4 = list(filter(lambda e: e.N == 4, x264))
x264_5 = list(filter(lambda e: e.N == 5, x264))

PWLLVM = LLVM_5
PWBerkeleyDBC = BerkeleyDBC_5
PWx264 = x264_5

plot2new([[LLVM_3,LLVM_4,LLVM_5],[x264_3,x264_4,x264_5],[BerkeleyDBC_3,BerkeleyDBC_4,BerkeleyDBC_5]],["N = 3","N = 4","N = 5"],["LLVM","x264","BerkeleyDBC"],plotname="PWN")

x264 = list(filter(lambda e: e.sys_name == "x264" and e.selectiontyp == "powerset" and e.optimisation_func == "weighted", experiments))
LLVM = list(filter(lambda e: e.sys_name == "LLVM" and e.selectiontyp== "powerset" and e.optimisation_func == "weighted", experiments))
BerkeleyDBC = list(filter(lambda e: e.sys_name == "BerkeleyDBC"  and e.selectiontyp== "powerset" and e.optimisation_func == "weighted", experiments))

LLVM_2 = list(filter(lambda e: e.N == 2, LLVM))
LLVM_3 = list(filter(lambda e: e.N == 3, LLVM))
LLVM_4 = list(filter(lambda e: e.N == 4, LLVM))
BerkeleyDBC_2 = list(filter(lambda e: e.N == 2, BerkeleyDBC))
BerkeleyDBC_3 = list(filter(lambda e: e.N == 3, BerkeleyDBC))
BerkeleyDBC_4 = list(filter(lambda e: e.N == 4, BerkeleyDBC))
x264_2 = list(filter(lambda e: e.N == 2, x264))
x264_3 = list(filter(lambda e: e.N == 3, x264))
x264_4 = list(filter(lambda e: e.N == 4, x264))

PSLLVM = LLVM_4
PSBerkeleyDBC = BerkeleyDBC_4
PSx264 = x264_4

plot2new([[LLVM_2,LLVM_3,LLVM_4],[x264_2,x264_3,x264_4],[BerkeleyDBC_2,BerkeleyDBC_3,BerkeleyDBC_4]],["N = 2","N = 3","N = 4"],["LLVM","x264","BerkeleyDBC"],plotname="PSN")
plot2new([[FWLLVM,PWLLVM,PSLLVM],[FWx264,PWx264,PSx264],[FWBerkeleyDBC,PWBerkeleyDBC,PSBerkeleyDBC]],["FW","PW","PS"],["LLVM","x264","BerkeleyDBC"],plotname="FWPWPS")

#plot2new([[LLVM_1,LLVM_2,LLVM_4,LLVM_8,LLVM_16],[x264_1,x264_2,x264_4,x264_8,x264_16],[BerkeleyDBC_1,BerkeleyDBC_2,BerkeleyDBC_4,BerkeleyDBC_8,BerkeleyDBC_16]],["N = 1","N = 2","N = 4","N = 8","N = 16"],["LLVM","x264","BerkeleyDBC"],plotname="FWN")
#plot2new([[LLVM_1,LLVM_2,LLVM_4,LLVM_8,LLVM_16],[x264_1,x264_2,x264_4,x264_8,x264_16],[BerkeleyDBC_1,BerkeleyDBC_2,BerkeleyDBC_4,BerkeleyDBC_8,BerkeleyDBC_16]],["N = 1","N = 2","N = 4","N = 8","N = 16"],["LLVM","x264","BerkeleyDBC"],plotname="FWN")