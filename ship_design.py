import numpy as np

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

    def __init__(self, n_var=31, **kwargs):
        self.ef = effect_fun.EffectFun()
        xl = [143, 16, 11, 5, 0, 71, -0.1, 70, -1, 0, 0, 3, 15, 6, 15, 4, 2, 1, 3, 36, 11, 11, 11, 11, 12, 18, 17, 17, 449, 6, 8]
        xu = [148, 18, 12, 6, 1, 76, 0.1, 76, 0, 1, 1, 5, 17, 8, 16, 5, 3, 3, 5, 41, 14, 13, 13, 12, 14, 20, 19, 19, 565, 7, 9]
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=11,xl=xl,xu=xu, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=15):
        dataset_dir = '/share/code/dataset/ship_design/'
        x = np.loadtxt(dataset_dir+"ori_pareto_set.csv", delimiter=",")
        return x

    def _evaluate(self, x_input, out, *args, **kwargs):
        f1=[]
        g=[]
        for x in x_input:
            temp = [x[0], x[1], x[3]] + \
                list(x[5:10]) + [x[12], x[10], x[11]] + list(x[13:28])
            temp1=[]
            for k in temp:
                temp1.append(round(k, 2))
            R_total, Bales_level, Diameter, Displacement, GM = ev.evaluate(temp1)
            LB = x[0] / x[1]  # 长宽比
            BT = x[1] / x[3]  # 宽度吃水比
            Cb = x[4]  # 方形系数
            Cp = x[9]  # 棱形系数
            DT = x[2] / x[3]  # 型深和吃水比

            f_temp = -(self.ef.bales_levels_effectFun(Bales_level)+self.ef.diameters_effectFun(Diameter) +
                       self.ef.displacements_effectFun(Displacement)+self.ef.gms_effectFun(GM))
            f1.append(f_temp)

            g1=LB-11
            g2=8-LB
            g3=0.4-Cb
            g4=Cb-0.56
            g5=Cp-0.7
            g6=0.4-Cp
            g7=2-DT
            g8=0.6-GM
            g9=GM-1.3
            g10=BT-3.6
            g11=2.8-BT

            g_temp=[g1, g2, g3, g4,g5,g6,g7,g8,g9,g10,g11]
            g.append(g_temp)
        out["F"] = f1
        out["G"] = g
def main():
    problem=ship_design()
    algorithm1 = PSO()
    algorithm = DE(
        pop_size=100,
        sampling=LHS(),
        variant="DE/rand/1/bin",
        CR=0.3,
        dither="vector",
        jitter=False
    )
    res = minimize(problem,
               algorithm,
               seed=1,
               save_history=True,
               verbose=True)

    X = res.X
    F = res.F
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