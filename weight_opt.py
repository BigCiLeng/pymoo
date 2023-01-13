import numpy as np
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import evaluate as ev
import effect_fun
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.problems.single import Rastrigin

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.problems import get_problem
from pymoo.operators.sampling.lhs import LHS
class ship_design(Problem):

    def __init__(self, newrank,n_var=5, **kwargs):
        ship_data = np.loadtxt('/share/code/dataset/ship_design/main_five_fun.csv',delimiter=',')
        self.ship_data=ship_data[:10]
        # self.newrank=np.array([5,2,0,6,1,9,3,7,8,4])
        self.newrank=newrank
        super().__init__(n_var=n_var, n_obj=1, n_eq_constr=1,xl=0,xu=1, **kwargs)

    def _evaluate(self, x_input, out, *args, **kwargs):
        f1=[]
        for x in x_input:
            rank=[]
            for data in self.ship_data:
                score=np.dot(data,x)
                rank.append(score)
            index=np.argsort(rank)
            f1.append(np.sum((index-self.newrank)**2))
        out["H"]=x_input[:,0]+x_input[:,1]+x_input[:,2]+x_input[:,3]+x_input[:,4]-1
        out["F"]=f1

def main():
    algorithm1 = PSO()
    algorithm = DE(
        pop_size=100,
        sampling=LHS(),
        variant="DE/rand/1/bin",
        CR=0.3,
        dither="vector",
        jitter=False
    )
    ref_dirs = get_reference_directions("das-dennis",1,n_partitions=12)
    algorithm3 = UNSGA3(ref_dirs,pop_size=100)
    termination = get_termination("n_eval", 25000)
    newrank=np.array([5,2,0,6,1,9,3,7,8,4])
    acc=0
    problem=ship_design(newrank)
    res = minimize(problem,
        algorithm1,
        termination=termination,
        seed=1,
        save_history=True,
        verbose=True)
    F=res.F

if __name__ == '__main__':
    main()
#  array([ 1.43023683e+02,  1.78397740e+01,  1.17887917e+01,  5.01180298e+00,
#         9.23034687e-01,  7.48867766e+01, -9.84186189e-02,  7.58929231e+01,
#        -9.55855153e-01,  8.35859047e-01,  6.70042118e-01,  3.39947785e+00,
#         1.69195577e+01,  7.73702290e+00,  1.59951510e+01,  4.16809119e+00,
#         2.43399352e+00,  2.68081267e+00,  3.17830446e+00,  3.60933471e+01,
#         1.19803191e+01,  1.11257578e+01,  1.11638040e+01,  1.13439022e+01,
#         1.23263745e+01,  1.99328277e+01,  1.79423850e+01,  1.84561500e+01,
#         4.90307889e+02,  6.15106639e+00,  8.62182074e+00])