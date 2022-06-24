from importlib.metadata import metadata
from p4 import DistBasedRepo
from p4 import P4Regressor
from sklearn.metrics.regression import r2_score
from sklearn.metrics import mean_squared_error
import math
import itertools
import os
from ActiveSampler import SmartCNFExpression 
from ActiveSampler import SmartCoverageSampler
import pandas as pd
import numpy as np
from ActiveSampler import ActiveSampler
from Dimacs_creator import DimacsCreator
import json
from datetime import datetime
from time import perf_counter

class Logger():
    def __init__(self,filename) -> None:
        self.filename = filename
        if not os.path.exists("measurements"):
            os.mkdir("measurements")
        if os.path.exists(os.path.join("measurements",  filename)):
            os.remove(os.path.join("measurements",  filename))
        f = open(os.path.join("measurements",  filename), "x")
        f.write("Log")
        f.write("\n")
        f.close()

    def log(self, *msg):
        print(*msg)
        f = open(os.path.join("measurements",  self.filename), "a")
        f.write(" ".join([str(txt) for txt in msg]))
        f.write("\n")
        f.close()

logger = Logger(filename = "log.txt")


class Experiment():
    def __init__(self,sys_name: str,selectiontyp,attribute = "Performance",time_attribute = "Performance",N: int = 5,maxiterations: int = 3,seed :int= None,optimisation_func = None,goal=None) -> None:
        if seed:
            np.random.seed(seed)

        self.timestamp = str(datetime.now())

        self.optimisation_func = optimisation_func
        self.sys_name = sys_name
        self.N = N
        self.iterations = maxiterations
        self.selectiontyp = selectiontyp
        self.rng = np.random.default_rng(seed)

        dc = DimacsCreator(sys_name=sys_name)
        dc.create()

        self.fm_active = SmartCNFExpression()
        self.fm_active.from_dimacs('dimacs/'+sys_name+'.dimacs')

        self.fm_static = SmartCNFExpression()
        self.fm_static.from_dimacs('dimacs/'+sys_name+'.dimacs')

        #attribute = "Performance"
        sys_dir = "/application/Distance-Based_Data/SupplementaryWebsite"
        self.cfg_sys = DistBasedRepo(sys_dir, sys_name, attribute=attribute)
        self.cfg_sys_time = DistBasedRepo(sys_dir, sys_name, attribute=time_attribute)

        self.position_map = self.cfg_sys.position_map
    
        if not os.path.exists("measurements"):
            os.mkdir("measurements")
        os.mkdir(os.path.join("measurements",self.timestamp.replace(":","a").replace(".","a") ))
        self.resultsfolder = os.path.join("measurements", self.timestamp.replace(":","a").replace(".","a"))
        self.results = {}
        self.results["metadata"] = {}
        self.results["metadata"]["timestamp"] = self.timestamp
        self.results["metadata"]["optimisation_func"] = self.optimisation_func
        self.results["metadata"]["sys_name"] = self.sys_name
        self.results["metadata"]["N"] = self.N
        self.results["metadata"]["iterations"] = self.iterations
        self.results["metadata"]["selectiontyp"] = self.selectiontyp
        self.results["metadata"]["seed"] = seed
        self.results["metadata"]["stop"] = "unexpected"
        self.results["metadata"]["attribute"] = attribute
        
        self.goal = goal
        self.results["times"] = {}
        self.results["times"]["samplings"] = []
        self.results["times"]["trainings"] = []
        self.results["times"]["measurments"] = []


        #evaluations
        #self.stepssize = []
        #self.static_evaluation = []
        #self.active_evaluation = []


    def df_to_matrix(self,df):
        names = list(df.columns)
        dataraw = []
        for _, row in df.iterrows():
            featurelist = []
            for name in names:
                if row[name]:
                    featurelist += [name]
            v = self.create_vector(self.position_map,featurelist)
            dataraw +=[v]
        data = np.asarray(dataraw)
        return data

    def create_vector(self,position_map, options):
        v = np.zeros(len(position_map))
        for option in options:
            if option in position_map:
                v[position_map[option]] = 1.0
        return v

    def get_all_data(self):
        configs = pd.DataFrame(list(self.cfg_sys.all_configs.keys()))
        config_attrs = pd.DataFrame(list(self.cfg_sys.all_configs.values()))
        df_configs = pd.concat([configs, config_attrs], axis=1)
        all_xs = np.array(df_configs.iloc[:, :-1])
        all_ys = list(df_configs.iloc[:, -1])
        return all_xs.copy(),all_ys.copy()

    def get_all_time(self):
        configs = pd.DataFrame(list(self.cfg_sys_time.all_configs.keys()))
        config_attrs = pd.DataFrame(list(self.cfg_sys_time.all_configs.values()))
        df_configs = pd.concat([configs, config_attrs], axis=1)
        all_xs = np.array(df_configs.iloc[:, :-1])
        all_ys = list(df_configs.iloc[:, -1])
        return all_xs.copy(),all_ys.copy()

    def evaluate(self, model, n_samples: int = None):
        xs, actual_ys = self.get_all_data()
        prediction_ys = model.predict(xs, n_samples = n_samples)

        #rss
        rmse = math.sqrt(mean_squared_error(actual_ys, prediction_ys))

        mape = np.mean(np.abs((actual_ys - prediction_ys) / actual_ys)) * 100

        r_square = r2_score(actual_ys, prediction_ys)
        return [rmse,mape,r_square]

    def get_value(self,xs):
        ys = np.empty(len(xs))
        all_xs, all_ys = self.get_all_data()
        for i,x in enumerate(xs):
            #print(list(cfg_sys.all_configs.keys()))
            for x2,y in zip(all_xs,all_ys):
                if np.array_equal(x,x2):
                    ys[i] = y
                    break
                if np.array_equal(x2,all_xs[~0]):
                    logger.log(x,"not found")
                    logger.log("positionmap:" ,self.position_map)
                    exit()
        return ys

    def get_times(self,xs):
        ys = np.empty(len(xs))
        all_xs, all_ys = self.get_all_data()
        for i,x in enumerate(xs):
            #print(list(cfg_sys.all_configs.keys()))
            for x2,y in zip(all_xs,all_ys):
                if np.array_equal(x,x2):
                    ys[i] = y
                    break
                if np.array_equal(x2,all_xs[~0]):
                    logger.log(x,"not found")
                    logger.log("positionmap:" ,self.position_map)
                    exit()
        return ys

    def _check_for_doubles(self,M):
        for i,a in enumerate(M):
            for j,b in enumerate(M):
                if i>=j:
                    continue
                if np.array_equal(a,b):
                    logger.log("double:" ,a)
                    logger.log("positionmap:" ,self.position_map)
                    exit()

    def execute(self):
        # static data
        staticssampler = SmartCoverageSampler(self.fm_static)
        fwsamples = staticssampler.sample(1)        
        staticdata = self.df_to_matrix(fwsamples)
        pwsamples = staticssampler.sample(2)
        pwdata = self.df_to_matrix(pwsamples)
        staticdataoffset = 0  
        #check can be removed
        self._check_for_doubles( np.concatenate((staticdata , pwdata)))
        
        self.rng.shuffle(pwdata)



        
        
        logger.log("Staticdata sampled.")
        #first training
        staticreg = P4Regressor("/tmp/results-debugging")
        try:
            staticreg.fit(staticdata, self.get_value(staticdata), feature_names= list(self.position_map.keys()))  
        except:
            logger.log("Experiment failed most likly to Lasso feature selection selected no options of interactions.")
            return
        logger.log("Initial training done.")
        self.stepssize += [len(staticdata)]
        self.static_evaluation += [self.evaluate(staticreg)]
        self.active_evaluation += [self.evaluate(staticreg)]

        #first activeregression is featurewise regression # wording
        activereg = staticreg

        stop = False
        
        #basicly set fw solutions as constrains in active fm
        activesampler = SmartCoverageSampler(self.fm_active)
        activesamples = activesampler.sample(1)
        activedata = self.df_to_matrix(activesamples)

        for i in range(self.iterations):
            if stop:
                break
            #sample new activedata
            activesampler = ActiveSampler(self.fm_active)            
            activesamples = activesampler.sample(activereg,self.N,self.selectiontyp,self.optimisation_func)
            new_activedata = self.df_to_matrix(activesamples)

            if len(new_activedata)+staticdataoffset > len(pwdata):
                new_activedata = new_activedata[:len(new_activedata)+staticdataoffset-len(pwdata)]
                stop = True

            if len(new_activedata) == 0:
                logger.log("no new activedata")
                if stop:
                    logger.log("Experiment finished early, because activedata is as large as pairwise data.")
                break

            activedata = np.concatenate((activedata,new_activedata)).copy()

            #choose new staticdata
            staticdata = np.concatenate((staticdata,pwdata[staticdataoffset:staticdataoffset+len(new_activedata)])).copy()
            staticdataoffset += len(new_activedata)

            logger.log("New data sampled.")

            print("check staticdata")
            self._check_for_doubles(staticdata)
            print("check activedata")
            self._check_for_doubles(activedata)

            #training
            
            staticreg = P4Regressor("/tmp/results-debugging")
            try:
                staticreg.fit(staticdata, self.get_value(staticdata), feature_names= list(self.position_map.keys()))    
            except:
                logger.log("Experiment failed most likly to Lasso feature selection selected no options of interactions.")
                return
            activereg = P4Regressor("/tmp/results-debugging")
            try:
                activereg.fit(activedata, self.get_value(activedata), feature_names= list(self.position_map.keys()))    
            except:
                logger.log("Experiment failed most likly to Lasso feature selection selected no options of interactions.")
                return
            #evaluation
            self.stepssize += [len(new_activedata)]
            self.static_evaluation += [self.evaluate(staticreg)]
            self.active_evaluation += [self.evaluate(activereg)] 

            if stop:
                logger.log("Experiment finished early, because activedata is as big as pairwise data.")

            logger.log("Iteration",i,"finished.")

    def experiment1(self, staticiterations= 1):
        staticssampler = SmartCoverageSampler(self.fm_static)
        pwsamples = staticssampler.sample(2)        
        staticdata = self.df_to_matrix(pwsamples)

        # train 10 static models
        staticreg = P4Regressor("/tmp/results-debugging")
        staticreg.fit(staticdata, self.get_value(staticdata), feature_names= list(self.position_map.keys()))
        self.staticmapescore = self.evaluate(staticreg)
        self.results["pwscore"] = [len(staticdata)]+self.evaluate(staticreg)
        logger.log("pairwise score",self.staticmapescore, "dataset size", len(staticdata))

        randomdata,_ = self.get_all_data()
        randomdata = randomdata.copy()
        self.rng.shuffle(randomdata)
        randomdata= randomdata[:len(staticdata)].copy()
        randomreg = P4Regressor("/tmp/results-debugging")
        randomreg.fit(randomdata, self.get_value(randomdata), feature_names= list(self.position_map.keys()))
        self.randomscore = self.evaluate(randomreg)
        self.results["randomscore"] = [len(randomdata)]+self.evaluate(randomreg)
        logger.log("random score",self.randomscore, "dataset size", len(randomdata))


        activesampler = SmartCoverageSampler(self.fm_active)
        fwsamples = activesampler.sample(1)        
        activedata = self.df_to_matrix(fwsamples)
        activereg = P4Regressor("/tmp/results-debugging")
        activereg.fit(activedata, self.get_value(activedata), feature_names= list(self.position_map.keys()))
        self.featurewisescore = self.evaluate(activereg)
        logger.log("featurewise score",self.evaluate(activereg), "dataset size", len(activedata))
        self.results["fwscore"] = [len(activedata)]+self.evaluate(activereg)
        self.results["activescore"] = []
        for i in range(self.iterations):
            logger.log("Iteration", i)
            activesampler = ActiveSampler(self.fm_active)
            activesamples = activesampler.sample(activereg,self.N,self.selectiontyp,self.optimisation_func)  
            new_activedata = self.df_to_matrix(activesamples)
            if len(new_activedata) == 0:
                logger.log("Experiment failed, no new activedata could be sampled.")
                self.results["metadata"]["stop"] = "no new activedata"
                break 
            activedata = np.concatenate((activedata,new_activedata)).copy()
            activereg = P4Regressor("/tmp/results-debugging")
            try:
                activereg.fit(activedata, self.get_value(activedata), feature_names= list(self.position_map.keys()))    
            except:
                logger.log("Experiment failed, most likly to Lasso feature selection selected no options of interactions.")
                self.results["metadata"]["stop"] = "no Lasso feature selection"
                break
            active_evaluation=self.evaluate(activereg)
            #self.stepssize+=[len(activedata)]
            logger.log("active score", active_evaluation, "dataset size", len(activedata))
            self.results["activescore"]+=[[len(activedata)]+self.evaluate(activereg)]
            if len(activedata) >= 1.3*len(staticdata):
                logger.log("experiment finished, max samples created.")
                self.results["metadata"]["stop"] = "max samples created"
                break
            if i == self.iterations-1:
                logger.log("experiment finished, max iteration reached.")
                self.results["metadata"]["stop"] = "max iteration reached"

    def experiment1b(self):
        self.results["metadata"]["type"] = "1"

        staticssampler = SmartCoverageSampler(self.fm_static)
        pwsamples = staticssampler.sample(2)        
        staticdata = self.df_to_matrix(pwsamples)

        activesampler = SmartCoverageSampler(self.fm_active)
        fwsamples = activesampler.sample(1)        
        activedata = self.df_to_matrix(fwsamples)
        activereg = P4Regressor("/tmp/results-debugging")
        try:
            activereg.fit(activedata, self.get_value(activedata), feature_names= list(self.position_map.keys()))
        except:
                logger.log("Experiment failed, most likly to Lasso feature selection selected no options of interactions.")
                self.results["metadata"]["stop"] = "no Lasso feature selection"
                return

        self.featurewisescore = self.evaluate(activereg)
        logger.log("featurewise score",self.evaluate(activereg), "dataset size", len(activedata))
        self.results["fwscore"] = [len(activedata)]+self.evaluate(activereg)
        self.results["activescore"] = []
        for i in range(self.iterations):
            logger.log("Iteration", i)
            activesampler = ActiveSampler(self.fm_active)
            activesamples = activesampler.sample(activereg,self.N,self.selectiontyp,self.optimisation_func)  
            new_activedata = self.df_to_matrix(activesamples)
            if len(new_activedata) == 0:
                logger.log("Experiment failed, no new activedata could be sampled.")
                self.results["metadata"]["stop"] = "no new activedata"
                break 
            activedata = np.concatenate((activedata,new_activedata)).copy()
            activereg = P4Regressor("/tmp/results-debugging")
            try:
                activereg.fit(activedata, self.get_value(activedata), feature_names= list(self.position_map.keys()))    
            except:
                logger.log("Experiment failed, most likly to Lasso feature selection selected no options of interactions.")
                self.results["metadata"]["stop"] = "no Lasso feature selection"
                break
            active_evaluation=self.evaluate(activereg)
            #self.stepssize+=[len(activedata)]
            logger.log("active score", active_evaluation, "dataset size", len(activedata))
            self.results["activescore"]+=[[len(activedata)]+self.evaluate(activereg)]
            if len(activedata) >= 1.3*len(staticdata):
                logger.log("experiment finished, max samples created.")
                self.results["metadata"]["stop"] = "max samples created"
                break
            if i == self.iterations-1:
                logger.log("experiment finished, max iteration reached.")
                self.results["metadata"]["stop"] = "max iteration reached"

    def experiment1c(self):
        self.results["metadata"]["type"] = "11"

        staticssampler = SmartCoverageSampler(self.fm_static)
        pwsamples = staticssampler.sample(2)        
        staticdata = self.df_to_matrix(pwsamples)

        activesampler = SmartCoverageSampler(self.fm_active)
        fwsamples = activesampler.sample(1)        
        activedata = self.df_to_matrix(fwsamples)
        activereg = P4Regressor("/tmp/results-debugging")
        try:
            activereg.fit(activedata, self.get_value(activedata), feature_names= list(self.position_map.keys()))
        except:
                logger.log("Experiment failed, most likly to Lasso feature selection selected no options of interactions.")
                self.results["metadata"]["stop"] = "no Lasso feature selection"
                return

        self.featurewisescore = self.evaluate(activereg)
        logger.log("featurewise score",self.evaluate(activereg), "dataset size", len(activedata))
        self.results["fwscore"] = [len(activedata)]+self.evaluate(activereg)
        self.results["activescore"] = []
        self.results["influences"] = []
        for i in range(self.iterations):
            logger.log("Iteration", i)
            influences = list(activereg.coef_ci(0.95)["influences"])
            logger.log(influences)
            self.results["influences"] += [influences]
            activesampler = ActiveSampler(self.fm_active)
            activesamples = activesampler.sample(activereg,self.N,self.selectiontyp,self.optimisation_func)  
            new_activedata = self.df_to_matrix(activesamples)
            if len(new_activedata) == 0:
                logger.log("Experiment failed, no new activedata could be sampled.")
                self.results["metadata"]["stop"] = "no new activedata"
                break 
            activedata = np.concatenate((activedata,new_activedata)).copy()
            activereg = P4Regressor("/tmp/results-debugging")
            try:
                activereg.fit(activedata, self.get_value(activedata), feature_names= list(self.position_map.keys()))    
            except:
                logger.log("Experiment failed, most likly to Lasso feature selection selected no options of interactions.")
                self.results["metadata"]["stop"] = "no Lasso feature selection"
                break
            active_evaluation=self.evaluate(activereg)
            #self.stepssize+=[len(activedata)]
            logger.log("active score", active_evaluation, "dataset size", len(activedata))
            self.results["activescore"]+=[[len(activedata)]+self.evaluate(activereg)]
            if len(activedata) >= 1.3*len(staticdata):
                logger.log("experiment finished, max samples created.")
                self.results["metadata"]["stop"] = "max samples created"
                break
            if i == self.iterations-1:
                logger.log("experiment finished, max iteration reached.")
                self.results["metadata"]["stop"] = "max iteration reached"

    def experiment3(self):
        self.results["metadata"]["type"] = "2"
        staticssampler = SmartCoverageSampler(self.fm_static)
        pwsamples = staticssampler.sample(2)        
        staticdata = self.df_to_matrix(pwsamples)
        goal = len(staticdata)
        

        activesampler = SmartCoverageSampler(self.fm_active)
        fwsamples = activesampler.sample(1)        
        activedata = self.df_to_matrix(fwsamples)
        activereg = P4Regressor("/tmp/results-debugging")

        
        try:
            activereg.fit(activedata, self.get_value(activedata), feature_names= list(self.position_map.keys()))
        except:
            logger.log("Experiment failed, most likly to Lasso feature selection selected no options of interactions.")
            self.results["metadata"]["stop"] = "no Lasso feature selection"
            return

        self.featurewisescore = self.evaluate(activereg)
        logger.log("featurewise score",self.evaluate(activereg), "dataset size", len(activedata))
        self.results["fwscore"] = [len(activedata)]+self.evaluate(activereg)
        self.results["activescore"] = []
        for i in range(self.iterations):
            logger.log("Iteration", i)
            activesampler = ActiveSampler(self.fm_active)
            activesamples = activesampler.sample(activereg,self.N,self.selectiontyp,self.optimisation_func)  
            new_activedata = self.df_to_matrix(activesamples)
            if len(new_activedata) == 0:
                logger.log("Experiment failed, no new activedata could be sampled.")
                self.results["metadata"]["stop"] = "no new activedata"
                break 
            
            activedata = np.concatenate((activedata,new_activedata)).copy()
            if len(activedata) > goal:
                activedata = activedata[:goal]
            activereg = P4Regressor("/tmp/results-debugging")
            try:
                activereg.fit(activedata, self.get_value(activedata), feature_names= list(self.position_map.keys()))    
            except:
                logger.log("Experiment failed, most likly to Lasso feature selection selected no options of interactions.")
                self.results["metadata"]["stop"] = "no Lasso feature selection"
                break
            active_evaluation=self.evaluate(activereg)
            #self.stepssize+=[len(activedata)]
            logger.log("active score", active_evaluation, "dataset size", len(activedata))
            self.results["activescore"]+=[[len(activedata)]+self.evaluate(activereg)]
            if len(activedata) == goal:
                logger.log("experiment finished, max samples created.")
                self.results["metadata"]["stop"] = "max samples created"
                break
            if i == self.iterations-1:
                logger.log("experiment finished, max iteration reached.")
                self.results["metadata"]["stop"] = "max iteration reached"

    def experiment4(self):
        self.results["metadata"]["type"] = "3"
        staticssampler = SmartCoverageSampler(self.fm_static)
        pwsamples = staticssampler.sample(2)        
        pwdata = self.df_to_matrix(pwsamples)

        self.rng.shuffle(pwdata)

        pwreg = P4Regressor("/tmp/results-debugging")

        try:
            pwreg.fit(pwdata, self.get_value(pwdata), feature_names= list(self.position_map.keys()))
        except:
            logger.log("Experiment failed, most likly to Lasso feature selection selected no options of interactions.")
            self.results["metadata"]["stop"] = "no Lasso feature selection"
            return

        self.pairwisescore = self.evaluate(pwreg)
        logger.log("pairwise score",self.evaluate(pwreg), "dataset size", len(pwdata))
        self.results["pwscore"] = [len(pwdata)]+self.evaluate(pwreg)
 
    def experiment5(self):
        self.results["metadata"]["type"] = "4"
        
        staticssampler = SmartCoverageSampler(self.fm_static)
        pwsamples = staticssampler.sample(2)        
        pwdata = self.df_to_matrix(pwsamples)

        all_xs, all_ys = self.get_all_data()
        randomdata = all_xs.copy()
        self.rng.shuffle(randomdata)
        randomdata = randomdata[:len(pwdata)]

        randomreg = P4Regressor("/tmp/results-debugging")

        try:
            randomreg.fit(randomdata, self.get_value(randomdata), feature_names= list(self.position_map.keys()))
        except:
            logger.log("Experiment failed, most likly to Lasso feature selection selected no options of interactions.")
            self.results["metadata"]["stop"] = "no Lasso feature selection"
            return

        self.randomscore = self.evaluate(randomreg)
        logger.log("featurewise score",self.evaluate(randomreg), "dataset size", len(randomdata))
        self.results["randomscore"] = [len(randomdata)]+self.evaluate(randomreg)      

    def experiment7(self):
        self.results["metadata"]["type"] = "7"
        staticssampler = SmartCoverageSampler(self.fm_static)
        pwsamples = staticssampler.sample(2)        
        pwdata = self.df_to_matrix(pwsamples)

        goal = len(pwdata)
        staticssampler2 = SmartCoverageSampler(self.fm_active)
        heuristicsamples = staticssampler2.sample(0)  
        staticssampler2 = NewHeuristicSampler(self.fm_active)
        heuristicsamples = staticssampler2.sample(goal)      
        heuristicdata = self.df_to_matrix(heuristicsamples)

        reg = P4Regressor("/tmp/results-debugging")

        try:
            reg.fit(heuristicdata, self.get_value(heuristicdata), feature_names= list(self.position_map.keys()))
        except:
            logger.log("Experiment failed, most likly to Lasso feature selection selected no options of interactions.")
            self.results["metadata"]["stop"] = "no Lasso feature selection"
            return
        self.results["metadata"]["stop"] = "max samples created"


        self.heuristicscore = self.evaluate(reg)
        logger.log("heuristic score",self.evaluate(reg), "dataset size", len(heuristicdata))
        self.results["heuristicscore"] = [len(heuristicdata)]+self.evaluate(reg)
 
    def experiment20(self):
        self.results["metadata"]["type"] = "20"
        t_start = perf_counter()

        activesampler = SmartCoverageSampler(self.fm_active)
        fwsamples = activesampler.sample(2)    
   
        activedata = self.df_to_matrix(fwsamples)
        
        t_end = perf_counter() 
        self.results["times"]["samplings"] += [t_end-t_start]
        
        t_start = perf_counter()
        activereg = P4Regressor("/tmp/results-debugging")
        self.results["times"]["measurments"] +=  self.get_times(activedata).tolist()
        try:
            activereg.fit(activedata, self.get_value(activedata), feature_names= list(self.position_map.keys()))
        except:
            logger.log("Experiment failed, most likly to Lasso feature selection selected no options of interactions.")
            self.results["metadata"]["stop"] = "no Lasso feature selection"
            return
        t_end = perf_counter()
        self.results["times"]["trainings"] += [t_end-t_start]
        self.results["metadata"]["stop"] = "max samples created"
        pw[self.sys_name] += [self.evaluate(activereg)[1]]
        logger.log("pairwise score",self.evaluate(activereg), "dataset size", len(activedata))
        self.results["pwscore"] = [len(activedata)]+self.evaluate(activereg)

    def experiment21(self):
        self.results["metadata"]["type"] = "21"
        t_start = perf_counter()

        activesampler = SmartCoverageSampler(self.fm_active)
        fwsamples = activesampler.sample(1)    
   
        activedata = self.df_to_matrix(fwsamples)
        
        
        t_end = perf_counter() 
        self.results["times"]["samplings"] += [t_end-t_start]
        self.results["times"]["measurments"] +=  self.get_times(activedata).tolist()
        t_start = perf_counter()
        activereg = P4Regressor("/tmp/results-debugging")
        try:
            activereg.fit(activedata, self.get_value(activedata), feature_names= list(self.position_map.keys()))
        except:
            logger.log("Experiment failed, most likly to Lasso feature selection selected no options of interactions.")
            self.results["metadata"]["stop"] = "no Lasso feature selection"
            return
        t_end = perf_counter()
        self.results["times"]["trainings"] += [t_end-t_start]


        self.featurewisescore = self.evaluate(activereg)
        logger.log("featurewise score",self.evaluate(activereg), "dataset size", len(activedata))
        self.results["fwscore"] = [len(activedata)]+self.evaluate(activereg)
        self.results["activescore"] = []
        for i in range(self.iterations):
            logger.log("Iteration", i)
            t_start = perf_counter()
            activesampler = ActiveSampler(self.fm_active)
            activesamples = activesampler.sample(activereg,self.N,self.selectiontyp,self.optimisation_func)  
            new_activedata = self.df_to_matrix(activesamples)
            t_end = perf_counter() 
            self.results["times"]["measurments"] +=  self.get_times(new_activedata).tolist()
            self.results["times"]["samplings"] += [t_end-t_start]
            if len(new_activedata) == 0:
                logger.log("Experiment failed, no new activedata could be sampled.")
                self.results["metadata"]["stop"] = "no new activedata"
                break 
            
            activedata = np.concatenate((activedata,new_activedata)).copy()

            t_start = perf_counter()
            activereg = P4Regressor("/tmp/results-debugging")
            try:
                activereg.fit(activedata, self.get_value(activedata), feature_names= list(self.position_map.keys()))    
            except:
                logger.log("Experiment failed, most likly to Lasso feature selection selected no options of interactions.")
                self.results["metadata"]["stop"] = "no Lasso feature selection"
                break
            t_end = perf_counter()
            self.results["times"]["trainings"] += [t_end-t_start]
            active_evaluation=self.evaluate(activereg)
            if active_evaluation[1] < self.goal:
                logger.log("Experiment finished, goal reached.")
                self.results["metadata"]["stop"] = "goal reached"
                break
            #self.stepssize+=[len(activedata)]
            logger.log("active score", active_evaluation, "dataset size", len(activedata))
            self.results["activescore"]+=[[len(activedata)]+self.evaluate(activereg)]
            if i == self.iterations-1:
                logger.log("experiment finished, max iteration reached.")
                self.results["metadata"]["stop"] = "max iteration reached"

    def datafortable(self):
        print(self.sys_name)
        self.fm = SmartCNFExpression()
        self.fm.from_dimacs('dimacs/'+self.sys_name+'.dimacs')
        print("O",len(self.fm.index_map))



        self.fm = SmartCNFExpression()
        self.fm.from_dimacs('dimacs/'+self.sys_name+'.dimacs')
        staticssampler = SmartCoverageSampler(self.fm)
        pwsamples = staticssampler.sample(1)        
        staticdata = self.df_to_matrix(pwsamples)
        print("FW",len(staticdata))

        self.fm = SmartCNFExpression()
        self.fm.from_dimacs('dimacs/'+self.sys_name+'.dimacs')
        staticssampler = SmartCoverageSampler(self.fm)
        pwsamples = staticssampler.sample(2)        
        staticdata = self.df_to_matrix(pwsamples)

        print("PW",len(staticdata))
        
        xs,_ = self.get_all_data()
        print("C", len(xs))

    def tojson(self):
        #if os.path.exists(os.path.join("messurments",  "results.json")):
        #    os.remove(os.path.join("messurments",  "results.json"))
        with open(os.path.join(self.resultsfolder,  "results.json"), "x") as f:
            json.dump(self.results,f)

    def outputdata(self):
        for i,a,s in zip(self.stepssize,self.active_evaluation,self.static_evaluation):
            print("")
            print(i,s)
            print(i,a)
    
    def outputfile(self,filename = "default"):
        new_samples = self.stepssize
        total_samples = [
            sum(new_samples[:i+1])
            for i in range(len(new_samples))
        ]
        activeA = [i[0]
            for i in self.active_evaluation
        ]
        activeB = [i[1]
            for i in self.active_evaluation
        ]
        activeC = [i[2]
            for i in self.active_evaluation
        ]
        staticA = [i[0]
            for i in self.static_evaluation
        ]
        staticB = [i[1]
            for i in self.static_evaluation
        ]
        staticC = [i[2]
            for i in self.static_evaluation
        ]

        df = pd.DataFrame(zip(new_samples,total_samples,staticA,staticB,staticC,activeA,activeB,activeC), columns = ["new_samples","total_samples","static_rmse","static_mape","static_r_square","active_rmse","active_mape","active_r_square"])
        df.to_csv(path_or_buf=os.path.join("messurments",  filename+".csv"))

def meta_experiment2():
    #system_names =[["LLVM","Performance"],["lrzip", "Performance"],["Polly","Performance"] ,["VP9","Performance"] ,["x264","Performance"]]#["x264"]#["x264","LLVM"]"Dune",                "7z", "BerkeleyDBC",
    system_name_attributes = [["lrzip", "Performance"],["x264","Performance"],["LLVM","Performance"],["VP9","UserTime"],["7z", "Performance"],["BerkeleyDBC", "Performance"],["JavaGC","GC Time"],["Polly","Performance"] ]
    maxiterations = 20
    repeatoffset = 0
    repeats = 1#5

    selectiontypsN = list(itertools.product([ActiveSampler.PAIRWISE],[2,4,8]))# [[ActiveSampler.POWERSET,6]] #list(itertools.product([ActiveSampler.FEATUREWISE],[1,3,5,10]))+ list(itertools.product([ActiveSampler.PAIRWISE],[2,4,8]))+ list(itertools.product( [ActiveSampler.POWERSET],[3,5,6]))
    
    optimisation_funcs = ["unweighted"]#"weighted", 

    startseed = 0

    for (system_name,attribute),(selectiontype,N),repeation,optimisation_func in itertools.product(system_name_attributes,selectiontypsN,range(repeatoffset, repeats),optimisation_funcs):
        logger.log(system_name,attribute,selectiontype,N,maxiterations,startseed,repeation)
        experiment = Experiment(system_name,attribute=attribute, N = N,maxiterations=maxiterations,seed=startseed,selectiontyp=selectiontype,optimisation_func =optimisation_func)
        experiment.execute()
        experiment.outputfile(system_name+";"+attribute+";"+selectiontype+";"+optimisation_func+";"+str(N)+";"+str(startseed))

        startseed +=1

def meta_experiment1():
    #system_name_attributes = [["x264","Performance"],["VP9","UserTime"],["7z", "Performance"],["BerkeleyDBC", "Performance"],["JavaGC","GC Time"],["Polly","Performance"] ]
    system_name_attributes = [["BerkeleyDBC", "Performance"]]
    maxiterations = 100
    repeats = 10

    selectiontypsN = list(itertools.product([ActiveSampler.FEATUREWISE],[16]))# [[ActiveSampler.POWERSET,6]] #list(itertools.product([ActiveSampler.FEATUREWISE],[1,3,5,10]))+ list(itertools.product([ActiveSampler.PAIRWISE],[2,4,8]))+ list(itertools.product( [ActiveSampler.POWERSET],[3,5,6]))
    
    optimisation_funcs = ["weighted","unweighted"]

    seed = 42

    for (system_name,attribute),(selectiontype,N),optimisation_func,_ in itertools.product(system_name_attributes,selectiontypsN,optimisation_funcs,range(repeats)):
        logger.log(system_name,attribute,selectiontype,N,maxiterations,seed)
        experiment = Experiment(system_name,attribute=attribute, N = N,maxiterations=maxiterations,seed=seed,selectiontyp=selectiontype,optimisation_func =optimisation_func)
        experiment.experiment1()
        experiment.tojson()
        #experiment.outputfile(system_name+";"+attribute+";"+selectiontype+";"+optimisation_func+";"+str(N)+";"+str(startseed))

        seed +=1

def meta_experiment1b():
    #system_name_attributes = [["x264","Performance"],["VP9","UserTime"],["7z", "Performance"],["BerkeleyDBC", "Performance"],["JavaGC","GC Time"],["Polly","Performance"] ]
    system_name_attributes = [["BerkeleyDBC", "Performance"]]
    maxiterations = 100
    repeats = 1

    selectiontypsN = list(itertools.product([ActiveSampler.PAIRWISE],[3]))*23 + list(itertools.product([ActiveSampler.PAIRWISE],[4]))*21 + list(itertools.product([ActiveSampler.PAIRWISE],[5]))*35
    selectiontypsN += list(itertools.product([ActiveSampler.POWERSET],[2]))*80 + list(itertools.product([ActiveSampler.POWERSET],[3]))*50 + list(itertools.product([ActiveSampler.POWERSET],[4]))*80
    selectiontypsN += list(itertools.product([ActiveSampler.FEATUREWISE],[1]))*50 + list(itertools.product([ActiveSampler.FEATUREWISE],[2]))*90 + list(itertools.product([ActiveSampler.FEATUREWISE],[4]))*90
    selectiontypsN += list(itertools.product([ActiveSampler.FEATUREWISE],[8]))*90
    optimisation_funcs = ["weighted"]

    seed = 5955435

    for (system_name,attribute),(selectiontype,N),optimisation_func,_ in itertools.product(system_name_attributes,selectiontypsN,optimisation_funcs,range(repeats)):
        logger.log(system_name,attribute,selectiontype,N,maxiterations,seed)
        experiment = Experiment(system_name,attribute=attribute, N = N,maxiterations=maxiterations,seed=seed,selectiontyp=selectiontype,optimisation_func =optimisation_func)
        experiment.experiment1b()
        experiment.tojson()
        #experiment.outputfile(system_name+";"+attribute+";"+selectiontype+";"+optimisation_func+";"+str(N)+";"+str(startseed))
        seed +=1

def meta_experiment11():
    #system_name_attributes = [["x264","Performance"],["VP9","UserTime"],["7z", "Performance"],["BerkeleyDBC", "Performance"],["JavaGC","GC Time"],["Polly","Performance"] ]
    system_name_attributes = [["x264", "Performance"]]
    maxiterations = 100
    repeats = 1


    selectiontypsN = list(itertools.product([ActiveSampler.PAIRWISE],[3]))*40 + list(itertools.product([ActiveSampler.PAIRWISE],[4]))*30 + list(itertools.product([ActiveSampler.PAIRWISE],[5]))*25
    selectiontypsN += list(itertools.product([ActiveSampler.POWERSET],[2]))*30 + list(itertools.product([ActiveSampler.POWERSET],[3]))*20 + list(itertools.product([ActiveSampler.POWERSET],[4]))*30
    selectiontypsN += list(itertools.product([ActiveSampler.FEATUREWISE],[1]))*30 + list(itertools.product([ActiveSampler.FEATUREWISE],[2]))*39 + list(itertools.product([ActiveSampler.FEATUREWISE],[4,8]))*40
    optimisation_funcs = ["weighted"]

    seed = 12055435

    for (system_name,attribute),(selectiontype,N),optimisation_func,_ in itertools.product(system_name_attributes,selectiontypsN,optimisation_funcs,range(repeats)):
        logger.log(system_name,attribute,selectiontype,N,maxiterations,seed)
        experiment = Experiment(system_name,attribute=attribute, N = N,maxiterations=maxiterations,seed=seed,selectiontyp=selectiontype,optimisation_func =optimisation_func)
        experiment.experiment1c()
        experiment.tojson()
        #experiment.outputfile(system_name+";"+attribute+";"+selectiontype+";"+optimisation_func+";"+str(N)+";"+str(startseed))
        seed +=1

def meta_experiment3():
    #system_name_attributes = [["lrzip", "Performance"],["LLVM","Performance"],["VP9","UserTime"],["BerkeleyDBC", "Performance"] ]
    system_name_attributes = [ ["ENERGY-x264_energy", "benchmark-energy"],["ENERGY-LLVM_energy", "energy"]]
    maxiterations = 100
    repeats = 20
    seed = 40043200
    for i in range(repeats):
        
            
        logger.log(i)
        for (system_name,attribute) in system_name_attributes:
            logger.log(i,system_name,attribute,maxiterations,seed)
            if seed < 2000037:
                seed +=1
                continue
            experiment = Experiment(system_name,attribute=attribute, N = 5,maxiterations=maxiterations,seed=seed,selectiontyp=ActiveSampler.PAIRWISE,optimisation_func =ActiveSampler.WEIGHTED)
            experiment.experiment3()
            experiment.tojson()
            
    
            seed +=1

def meta_experiment4():
    #system_name_attributes = [["lrzip", "Performance"],["LLVM","Performance"],["VP9","UserTime"],["BerkeleyDBC", "Performance"] ]
    system_name_attributes = [ ["ENERGY-x264_energy", "benchmark-energy"],["ENERGY-LLVM_energy", "energy"]]
    maxiterations = 100
    repeats = 20
    seed = 30043500
    for i in range(repeats):
        logger.log(i)
        for (system_name,attribute) in system_name_attributes:
            logger.log(i,system_name,attribute,maxiterations,seed)
            experiment = Experiment(system_name,attribute=attribute, N = 5,maxiterations=maxiterations,seed=seed,selectiontyp=ActiveSampler.PAIRWISE,optimisation_func =ActiveSampler.WEIGHTED)
            experiment.experiment4()
            experiment.tojson()
            
    
            seed +=1

            experiment = Experiment(system_name,attribute=attribute, N = 5,maxiterations=maxiterations,seed=seed,selectiontyp=ActiveSampler.PAIRWISE,optimisation_func =ActiveSampler.WEIGHTED)
            experiment.experiment5()
            experiment.tojson()

            seed +=1

def meta_experiment7():
    system_name_attributes = [["x264","Performance"],["7z", "Performance"],["JavaGC","GC Time"],["Polly","Performance"],["lrzip", "Performance"],["LLVM","Performance"],["VP9","UserTime"],["BerkeleyDBC", "Performance"],["ENERGY-x264_energy", "benchmark-energy"],["ENERGY-LLVM_energy", "energy"]]
    maxiterations = 100
    repeats = 20
    seed = 300436565
    for i in range(repeats):
        logger.log(i)
        for (system_name,attribute) in system_name_attributes:
            logger.log(i,system_name,attribute,maxiterations,seed)
            experiment = Experiment(system_name,attribute=attribute, N = 5,maxiterations=maxiterations,seed=seed,selectiontyp=ActiveSampler.PAIRWISE,optimisation_func =ActiveSampler.WEIGHTED)
            experiment.experiment7()
            experiment.tojson()
            
    
            seed +=1

pw = {"LLVM":[],"ENERGY-LLVM_energy":[]}

def meta_experiment20():
    system_name_attributes = [["LLVM","Performance","Performance"],["x264","Performance","Performance"],["ENERGY-LLVM_energy", "energy","performance"]]
    maxiterations = 100
    repeats = 10
    seed = 400436565
    for i in range(repeats):
        logger.log(i)
        for (system_name,attribute,time_attribute) in system_name_attributes:
            logger.log(i,system_name,attribute,maxiterations,seed)
            experiment = Experiment(system_name,attribute=attribute,time_attribute=time_attribute , N = 5,maxiterations=maxiterations,seed=seed,selectiontyp=ActiveSampler.PAIRWISE,optimisation_func =ActiveSampler.WEIGHTED)
            experiment.experiment20()
            experiment.tojson()
            seed +=1

def meta_experiment21():
    system_name_attributes_goals = [["LLVM","Performance","Performance",6.2],["x264","Performance","Performance",2,6],["ENERGY-LLVM_energy", "energy","performance",22.2]]
    maxiterations = 100
    repeats = 10
    seed = 410436565
    for i in range(repeats):
        logger.log(i)
        for (system_name,attribute,time_attribute,goal) in system_name_attributes_goals:
            logger.log(i,system_name,attribute,maxiterations,seed)
            experiment = Experiment(system_name,attribute=attribute,time_attribute=time_attribute , N = 5,maxiterations=maxiterations,seed=seed,selectiontyp=ActiveSampler.PAIRWISE,optimisation_func =ActiveSampler.WEIGHTED,goal=goal)
            experiment.experiment21()
            experiment.tojson()
            seed +=1

#meta_experiment()
#experiment = Experiment(system_name,attribute=attribute, N = N,maxiterations=maxiterations,seed=startseed,selectiontyp=selectiontype,optimisation_func =optimisation_func)
meta_experiment20()
meta_experiment21()
#experiment = Experiment("BerkeleyDBC",attribute="Performance", N = 2,maxiterations=100,seed=76,selectiontyp=ActiveSampler.FEATUREWISE,optimisation_func ="weighted")
#experiment.experiment1()
#experiment.tojson()
#7z, BerkeleyDBC, Dune, Hipacc, JavaGC, LLVM, lrzip, Polly, VP9, x264

#values not found Dune 
# Hipacc [0. 0. 1. 0. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#pairwise sampler is too slow ["7z", ]["JavaGC","GC Time"] ["Polly","Performance"] ["VP9","UserTime"]
#key error VP9 UserTime
#buggy ["JavaGC","GC Time"]
# 