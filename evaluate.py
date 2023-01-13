'''
Author: 孙凯威
Date: 2021-11-11 18:54:57
LastEditTime: 2021-11-17 12:50:51
LastEditors: Please set LastEditors
Description: 得分函数，自己设置的，后来需要被动态链接库取代。
FilePath: \ship_design\evaluate.py
'''
import matlab
import PerformanceCalculation

'''
description: 
param {船舶主尺度和型线参数,26个数据的列表等数据格式} ShipHull_parameters
param {航速,实数类型，非可迭代类型} V
return {总阻力（KN）} R_total
return {耐波贝尔斯品级} Bales_level
return {回转直径（m）} Diameter
return {排水量（t）} Displacement
return {初稳性高（m）} GM
'''


def evaluate(ShipHull_parameters, V=20):
    P = PerformanceCalculation.initialize()  # 初始化PerformanceCalculation类
    SH = matlab.double(ShipHull_parameters)  # 将船舶主尺度等参数改为matlab的数组类型
    V = matlab.double([V])  # 同样将V转换成matlab的数据类型
    R_total, Bales_level, Diameter, Displacement, GM = P.PerformanceCalculation(SH, V,
                                                                                nargout=5)  # 调用函数，需要指定输出个数，否则只输出第一个
    return R_total, Bales_level, Diameter, Displacement, GM


if __name__ == "__main__":
    PP = [143, 17, 6.1, 72, 0, 71, 67.9868, 0.724, 15.5, 0.8572, 0.04, 52.85,15.47, 4.4, 2.77, 1.96, 8.34, 35,17.57,16.63, 16.32, 16.2, 16.21, 21.1, 21.93, 18.54]
    V = 20
    out1, out2, out3, out4, out5 = evaluate(PP, V)  # 测试代码
    print(out1, out2, out3, out4, out5)
