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

# evaluate mlp
import pytorch_lightning as pl
import torch
import evaluate.evaluate_normalxy as evaluate_normalxy
import numpy as np
import pandas as pd

def PE(result, L=5):
    """PE(Position Encoding)

    Args:
        result (np.array): 操作对象, 每一行将会被PE。
        L (int, optional): 升维的幅度。

    Returns:
        new_result: 升维之后的result。
        
    E.g.
        result.shape = (10, 3)
        -> Set L -> PE
        -> new_result = (10, 3*2L)
    """
    new_result = []
    for idx, proporties in enumerate(result):
        list = []
        for i in range(0, L):
            res = np.sin(2**i*np.pi*proporties)
            list.append(res)
            res = np.cos(2**i*np.pi*proporties)
            list.append(res)
        new_result.append(np.stack(list, axis=0).reshape(1, -1)[0])
    new_result = np.array(new_result)
    return new_result
def data_solve():
    score_list = []
    result = np.loadtxt("../dataset/ship_design/main_five.csv", dtype=float, delimiter=",")
    # 效用函数
    efunc = effect_fun.EffectFun()
    for idx, p in enumerate(result.T):
        res_list = []
        if(idx == 0):
            for item in p:
                res_list.append(efunc.r_tatals_effectFun(item))
        elif(idx == 1):
            for item in p:
                res_list.append(efunc.bales_levels_effectFun(item))
        elif(idx == 2):
            for item in p:
                res_list.append(efunc.diameters_effectFun(item))
        elif(idx == 3):
            for item in p:
                res_list.append(efunc.displacements_effectFun(item))
        else:
            for item in p:
                res_list.append(efunc.gms_effectFun(item))
        score_list.append(res_list)
    result = np.array(score_list).T
    # print('result', result)
    # 升维操作
    result = PE(result, 5)
    # 高斯归一化
    std = result.std(0)
    result = result/std
    result_max=result.max(0)
    result_min=result.min(0)
    return std,result_max,result_min
class ship_design(Problem):

    def __init__(self, n_var=31, **kwargs):
        self.data_1, self.data_2 = evaluate_normalxy.ori_data_std()
        self.rank_1, self.rank_2 = evaluate_normalxy.ori_rank()
        self.model = evaluate_normalxy.MLP_EVALUATE_SYSTEM.load_from_checkpoint("ckpts/best_normalxy.ckpt")
        self.model.eval()
        self.ef = effect_fun.EffectFun()
        self.std,self.max_datas1,self.min_datas1=data_solve()
        xl = [143, 16, 11, 5, 0, 71, -0.1, 70, -1, 0, 0, 3, 15, 6, 15, 4, 2, 1, 3, 36, 11, 11, 11, 11, 12, 18, 17, 17, 449, 6, 8]
        xu = [148, 18, 12, 6, 1, 76, 0.1, 76, 0, 1, 1, 5, 17, 8, 16, 5, 3, 3, 5, 41, 14, 13, 13, 12, 14, 20, 19, 19, 565, 7, 9]
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=11,xl=xl,xu=xu, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=15):
        dataset_dir = '/share/code/dataset/ship_design/'
        x = np.loadtxt(dataset_dir+"ori_pareto_set.csv", delimiter=",")
        return x

    def _evaluate(self, x_input, out, *args, **kwargs):
        temp=evaluate_normalxy.normalize(x_input,self.data_1,self.data_2)
        temp=torch.tensor(temp,dtype=torch.float32)
        with torch.no_grad():
            temp=self.model(temp)
        temp=evaluate_normalxy.denormalize(temp,self.rank_1,self.rank_2)
        results = temp.numpy()
        GM=results.T[4]
        results = [[self.ef.r_tatals_effectFun(i[0]),self.ef.bales_levels_effectFun(i[1])+self.ef.diameters_effectFun(i[2]) +
                    self.ef.displacements_effectFun(i[3])+self.ef.gms_effectFun(i[4])] for i in results]
        results = np.array(results)
        results=PE(results,5)
        results=results/self.std
        results=(results-self.min_datas1)/(self.max_datas1-self.min_datas1)
        new_weights=np.ones(50)*0.02
        rank=new_weights @ results.T
        out["F"] = [i[0] for i in rank]

        x=x_input.T
        x[0]=x[5]+x[6]+x[7]
        LB = x[0] / x[1]  # 长宽比
        BT = x[1] / x[3]  # 宽度吃水比
        Cb = x[4]  # 方形系数
        Cp = x[9]  # 棱形系数
        DT = x[2] / x[3]  # 型深和吃水比
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
        out["G"]=np.array([g1, g2, g3, g4,g5,g6,g7,g8,g9,g10,g11]).T
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