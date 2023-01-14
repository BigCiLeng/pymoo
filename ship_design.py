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
        self.model = evaluate_normalxy.MLP_EVALUATE_SYSTEM.load_from_checkpoint("evaluate/ckpts/best_normalxy.ckpt")
        self.model.eval()
        self.ef = effect_fun.EffectFun()
        self.std,self.max_datas1,self.min_datas1=data_solve()
        xl = [143.01975614083997, 16.370096110303976, 11.241500212251673, 5.638218624397934, 0.456285223288534, 71.50487491510255, 0.0, 70.13268399949628, -0.006216972493419737, 0.6171647367704676, 0.6837863921588061, 3.6272755949699556, 15.778737858945052, 6.024892307900446, 15.305900718209097, 4.447777249459548, 2.765290123707956, 1.8501753056331214, 3.220179596692431, 36.89523353359071, 11.937808784269016, 11.345157764968105, 11.685925046645828, 11.076980795275269, 12.888232471485162, 18.534131127174525, 17.70925703337477, 17.623443146324174, 449.3915571156782, 6.8058150448755335, 8.209141678331397]
        xu = [147.86257262079042, 17.713193383595247, 11.60000896198561, 5.774715502319271, 0.4680546214182675, 75.54954577382784, 0.001, 75.05504836704198, -0.003257296569654667, 0.6375116885357004, 0.7060244948250954, 4.3824662593626, 16.78610060696743, 7.9248086911414335, 15.63252394911592, 4.577298149748174, 2.9130110405167446, 2.0567682869883677, 4.538799883659495, 40.560736904634645, 13.172214731150918, 12.703972511254154, 12.538116825400943, 11.696483618411168, 13.67253738164199, 19.468943869557847, 18.648436269778948, 18.986724967893057, 564.4167610227898, 6.96968291042843, 8.980470252780348]
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
        results = [[self.ef.r_tatals_effectFun(i[0]),self.ef.bales_levels_effectFun(i[1]),self.ef.diameters_effectFun(i[2]),\
                    self.ef.displacements_effectFun(i[3]),self.ef.gms_effectFun(i[4])] for i in results]
        results = np.array(results)
        results=PE(results,5)
        results=results/self.std
        results=(results-self.min_datas1)/(self.max_datas1-self.min_datas1)
        new_weights=np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.92131912e-02,
       1.43934367e-02, 3.40198903e-02, 4.09874951e-03, 1.64354073e-02,
       6.80007701e-02, 7.92597576e-20, 2.47755821e-02, 1.63853570e-02,
       9.06612155e-03, 6.16648374e-03, 0.00000000e+00, 3.37547987e-05,
       3.20041559e-02, 5.28931665e-02, 4.42436067e-01, 0.00000000e+00,
       3.88775611e-04, 0.00000000e+00, 8.33324748e-03, 1.64619008e-02,
       0.00000000e+00, 4.92803962e-20, 3.89725874e-04, 0.00000000e+00,
       0.00000000e+00, 5.19395691e-03, 5.56428772e-04, 1.99419957e-20,
       1.86156297e-20, 4.81774177e-02, 2.04551513e-19, 0.00000000e+00,
       1.72454655e-02, 2.85403552e-05, 4.87936781e-20, 4.95775277e-20,
       4.96199007e-20, 2.70300567e-02, 2.80705993e-03, 3.62976792e-02,
       2.85293162e-05, 0.00000000e+00, 3.70257515e-02, 2.30276059e-02,
       4.19924768e-20, 7.08572440e-03])
        rank=-(new_weights @ results.T)
        out["F"] = rank

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
    termination = get_termination("n_eval", 100000)
    res = minimize(problem,
               algorithm,
               seed=1,
               save_history=True,
               termination=termination,
               verbose=True)

    X = res.X
    F = res.F
    G = res.G
    print(X,F,G)
if __name__ == '__main__':
    main()
# [ 1.44442277e+02  1.64203229e+01  1.15516863e+01  5.66601270e+00
#   4.66964090e-01  7.48778829e+01  2.56097792e-04  7.22587017e+01
#  -5.93711975e-03  6.25470467e-01  6.86008736e-01  3.62754323e+00
#   1.58100508e+01  6.46892134e+00  1.53982211e+01  4.53352044e+00
#   2.77754400e+00  1.91493533e+00  3.50351419e+00  3.99930557e+01
#   1.28187698e+01  1.14063352e+01  1.23435296e+01  1.11007509e+01
#   1.28885294e+01  1.88845474e+01  1.84511430e+01  1.84393214e+01
#   4.56577528e+02  6.95560089e+00  8.80530597e+00] [-0.81024546] [-2.03934548 -0.96065452 -0.06696409 -0.09303591 -0.07452953 -0.22547047
#  -0.03876816 -0.30818776 -0.39181224 -0.70196151 -0.09803849]