from itertools import combinations
#from pycosa.features import FeatureModel 
from pycosa.sampling import Sampler
from pycosa.sampling import int_to_config
import pycosa.modeling as modeling
from p4 import P4Regressor
import itertools
import more_itertools
import z3
import numpy as np
import pandas as pd

def z3Abs(x):
    return z3.If(x > 0, x, -x)

class SmartCNFExpression(modeling.CNFExpression):
    def __init__(self):
        self.restricted_solutions = []
        super().__init__()

class ActiveSampler(Sampler):
    POWERSET = "powerset"
    FEATUREWISE = "featurewise"
    PAIRWISE = "pairwise"

    UNWEIGHTED = "unweighted"
    WEIGHTED = "weighted"

    def __init__(self, fm):
        super().__init__(fm)

    def opti_func_sum(self,n_options):
        return z3.Sum(
                    [
                        z3.ZeroExt(n_options, z3.Extract(i, i, self.fm.target))
                        for i in range(n_options)
                    ]
        )

    def opti_func_weighted_sum(self,n_options):
        frequencies = [0]+[self.get_frequency([feature])-0.5 for feature in self.fm.feature_map]
        return z3.Sum([
                z3.ZeroExt(n_options, z3.Extract(i, i, self.fm.target))*frequencies[i]
                    for i in range(n_options)
                ])
                

    def sample(self, reg: P4Regressor ,N: int, selectiontyp = "featurewise", optimisation_func = "unweighted", ci: float = 0.95):
        #self.presamples_positionmap = positionmap

        #evaluation Regressor
        coef_ci = reg.coef_ci(ci)["influences"]
        coef_ci_diff = {}
        for k in coef_ci:
            coef_ci_diff[k] = abs(coef_ci[k][1]-coef_ci[k][0])
        if N>len(coef_ci):
            N = len(coef_ci)
        most_uncertain_influences = sorted(coef_ci_diff,key=coef_ci_diff.get)[-N:]
        #most_uncertain_influences = [('ref_5',), ('ref_9',option3), ("option3",)]

        n_options = len(self.fm.feature_map)

        #print(self.fm.bitvec_constraints)
        #print(self._parse_Matrix(pre_solutions,positionmap))

        constraints = self.fm.bitvec_constraints
        #constraints += self._parse_Matrix(pre_solutions,positionmap)
        
        # add non-boolean constraints
        # constraints += self.fm.constraint

        solutions = []

        feature_combinations = []
        if selectiontyp == "featurewise":
             feature_combinations = [[x] for x in most_uncertain_influences]
        if selectiontyp == "pairwise":
             feature_combinations = itertools.combinations_with_replacement(most_uncertain_influences, 2)
        if selectiontyp == "powerset":
             feature_combinations = more_itertools.powerset(most_uncertain_influences)



        for feature_combination in feature_combinations:
            # initialize a new optimizer
            optimizer = z3.Optimize()

            # add feature model clauses
            optimizer.add(constraints)

            # add previous solutions from this and previous iteration as constraints
            for solution in self.fm.restricted_solutions:
                optimizer.add(solution != self.fm.target)

            #print(self.fm.find_optional_options()["mandatory"])
            #add constraints based on most uncertain influences  #inf -> influence
            if selectiontyp == "powerset":
                for influence in most_uncertain_influences:
                    
                        if influence in feature_combination:
                            constraint = self._constrain_enabled_features(influence) == len(influence)
                        else:
                            constraint = self._constrain_enabled_features(influence) < len(influence)
            else:
                for influence in feature_combination:
                    frequency = self.get_frequency(influence)
                    print(influence,frequency)
                    if frequency >0.5:
                        constraint = self._constrain_enabled_features(influence)< len(influence)
                    else:
                        constraint = self._constrain_enabled_features(influence)== len(influence)
                optimizer.add(constraint)

            if optimisation_func == "unweighted":
                optimizer.minimize(self.opti_func_sum(n_options))
            elif optimisation_func == "weighted":
                optimizer.minimize(self.opti_func_weighted_sum(n_options))

            if optimizer.check() == z3.sat:
                solution = optimizer.model()[self.fm.target]
                solutions.append(solution)
                self.fm.restricted_solutions.append(solution)
        
        if solutions == []:
            features = [self.fm.index_map[i] for i in self.fm.index_map]
            return pd.DataFrame([], columns=features)
        solutions = np.vstack(
            [int_to_config(s.as_long(), len(self.fm.index_map)) for s in solutions]
        )[:, 1:]
        features = [self.fm.index_map[i] for i in self.fm.index_map]
        sample = pd.DataFrame(solutions, columns=features)
        return sample

    def get_frequency(self,influence):
        solutions = np.vstack(
            [int_to_config(s.as_long(), len(self.fm.index_map)) for s in self.fm.restricted_solutions]
        )
        
        reversed_index_map = {a:b for b,a in self.fm.index_map.items()}
        features = [reversed_index_map[i] for i in influence]
        #print(self.fm.index_map)
        #print(reversed_index_map)
        #print(influence,features)

        if len(features) == 1:
            v = solutions[:,features[0]]
            return np.sum(v)/len(self.fm.restricted_solutions)
        if len(features) == 2:
            v = np.multiply(solutions[:,features[0]],solutions[:,features[1]])
            return np.sum(v)/len(self.fm.restricted_solutions)
              

    def _constrain_enabled_features(self, features):
        n_options = len(self.fm.feature_map)
        #bitvec_dict = {v: (n_options - k) for v,k in self.fm.feature_map.items()}
        features_indexes = [self.fm.feature_map[feature] for feature in features]
        enabled_features = z3.Sum(
            [
                z3.ZeroExt(n_options, z3.Extract(idx, idx, self.fm.target))
                for idx in features_indexes
            ]
        )
        return enabled_features

#pycosa coverage sampler but it saves solutions
class SmartCoverageSampler(Sampler):

    def __init__(self, fm: SmartCNFExpression, **kwargs):
        super().__init__(fm, **kwargs)

    def sample(self, t: int, negwise: bool = False, include_minimal: bool = False):

        optionals = self.fm.find_optional_options()["optional"]

        n_options = len(self.fm.index_map)
        target = self.fm.target
        constraints = []
        solutions = []
        for interaction in itertools.combinations(optionals, t):

            # initialize a new optimizer
            optimizer = z3.Optimize()

            # add feature model clauses
            optimizer.add(self.fm.bitvec_constraints)

            # add previous solutions as constraints
            for solution in self.fm.restricted_solutions:
                optimizer.add(solution != target)

            
            #for solution in solutions:
            #    optimizer.add(solution != target)

            for opt in interaction:
                if not negwise:
                    constraint = z3.Extract(opt, opt, target) == 1
                else:
                    constraint = z3.Extract(opt, opt, target) == 0

                optimizer.add(constraint)

            # function that counts the number of enabled features
            func = z3.Sum(
                [
                    z3.ZeroExt(n_options, z3.Extract(i, i, target))
                    for i in range(n_options)
                ]
            )

            if not negwise:
                optimizer.minimize(func)
            else:
                optimizer.maximize(func)

            if optimizer.check() == z3.sat:
                solution = optimizer.model()[target]
                constraints.append(solution != target)
                solutions.append(solution)
                self.fm.restricted_solutions.append(solution)
        
        if solutions == []:
            features = [self.fm.index_map[i] for i in self.fm.index_map]
            return pd.DataFrame([], columns=features)
        solutions = np.vstack(
            [int_to_config(s.as_long(), len(self.fm.index_map)) for s in solutions]
        )[:, 1:]
        features = [self.fm.index_map[i] for i in self.fm.index_map]
        sample = pd.DataFrame(solutions, columns=features)
        return sample



