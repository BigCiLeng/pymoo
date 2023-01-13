import numpy as np
import math
import matplotlib.pyplot as plt
class EffectFun():

    def __init__(self):
        self.r_tatal_base=382.3685468
        self.bales_level_base=10.32398327
        self.diameter_base=7.005626601
        self.displacement_base=6846.366353
        self.gms_base=1.017701284

        self.r_tatal_max=412.6142178
        self.r_tatal_min=372.6449692
        self.bales_level_max=10.88994216
        self.bales_level_min=9.643966737
        self.diameter_max=7.064171914
        self.diameter_min=6.898748108
        self.displacement_max=7381.217777
        self.displacement_min=6675.766026
        self.gms_max=1.305389037
        self.gms_min=0.869365414

        # self.r_tatal_base=374.0666667
        # self.bales_level_base=8.566666667
        # self.diameter_base=6.51
        # self.displacement_base=6301.666667
        # self.gms_base=1.876666667

        # self.r_tatal_max=self.r_tatal_base*1.2
        # self.r_tatal_min=self.r_tatal_base*0.8
        # self.bales_level_max=self.bales_level_base*1.2
        # self.bales_level_min=self.bales_level_base*0.8
        # self.diameter_max=self.diameter_base*1.2
        # self.diameter_min=self.diameter_base*0.8
        # self.displacement_max=self.displacement_base*1.3
        # self.displacement_min=5000
        # self.gms_max=self.gms_base*1.5
        # self.gms_min=self.gms_base*0.8
        # self.delta=11/18
    def r_tatals_effectFun(self,x):
        if x<self.r_tatal_min:
            return 1
        elif x>=self.r_tatal_min and x<self.r_tatal_base:
            return 0.6+0.4*np.sin((self.r_tatal_base-x)/(self.r_tatal_base-self.r_tatal_min)*np.pi/2)
        elif x>=self.r_tatal_base and x<self.r_tatal_max:
            return 0.6*np.cos((self.r_tatal_max-x)/(self.r_tatal_max-self.r_tatal_base)*np.pi/2-np.pi/2)
        else:
            return 0
    def bales_levels_effectFun(self,x):
        if x<self.bales_level_min:
            return 1
        elif x>=self.bales_level_min and x<self.bales_level_base:
            return 0.6+0.4*np.sin((self.bales_level_base-x)/(self.bales_level_base-self.bales_level_min)*np.pi/2)
        elif x>=self.bales_level_base and x<self.bales_level_max:
            return 0.6*np.cos((self.bales_level_max-x)/(self.bales_level_max-self.bales_level_base)*np.pi/2-np.pi/2)
        else:
            return 0
    def diameters_effectFun(self,x):
        if x<self.diameter_min:
            return 1
        elif x>=self.diameter_min and x<self.diameter_base:
            return 0.6+0.4*np.sin((self.diameter_base-x)/(self.diameter_base-self.diameter_min)*np.pi/2)
        elif x>=self.diameter_base and x<self.bales_level_max:
            return 0.6*np.cos((self.diameter_max-x)/(self.diameter_max-self.diameter_base)*np.pi/2-np.pi/2)
        else:
            return 0
    def displacements_effectFun(self,x):
        if x>=self.displacement_max or x<self.displacement_min:
            return 0
        else:
            return np.sin((x-self.displacement_base+self.displacement_max-self.displacement_min)/(self.displacement_max-self.displacement_min)*np.pi/2)
    def gms_effectFun(self,x):
        if x<=self.gms_min:
            return 0
        elif x>=self.gms_min and x<self.gms_max:
            return 1+np.sin((x-self.gms_min)/(self.gms_max-self.gms_min)*np.pi/2-np.pi/2)
        else:
            return 1
    def plot(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
        plt.rcParams['axes.unicode_minus'] = False
        x1 = np.arange(self.r_tatal_min, self.r_tatal_max, 0.001)
        x2 = np.arange(self.bales_level_min, self.bales_level_max, 0.01)
        x3 = np.arange(self.diameter_min, self.diameter_max, 0.01)
        x4 = np.arange(self.displacement_min, self.displacement_max, 10)
        x5 = np.arange(self.gms_min, self.gms_max, 0.01)
        y1=[]
        y2=[]
        y3=[]
        y4=[]
        y5=[]
        for t in x1:
            y1.append(self.r_tatals_effectFun(t))
        for t in x2:
            y2.append(self.bales_levels_effectFun(t))
        for t in x3:
            y3.append(self.diameters_effectFun(t))
        for t in x4:
            y4.append(self.displacements_effectFun(t))
        for t in x5:
            y5.append(self.gms_effectFun(t))
        fig1=plt.figure(figsize=(16,4.8))
        fig1.canvas.manager.window.wm_geometry('+0+480')
        plt.ion()
        plt.subplot(1,5,1)
        plt.title('单位排水量阻力效用函数')
        plt.plot(x1,y1)
        plt.subplot(1,5,2)
        plt.title('耐波贝尔斯品级效用函数')
        plt.plot(x2,y2)
        plt.subplot(1,5,3)
        plt.title('回转直径效用函数')
        plt.plot(x3,y3)
        plt.subplot(1,5,4)
        plt.title('排水量效用函数')
        plt.plot(x4,y4)
        plt.subplot(1,5,5)
        plt.title('初稳性高效用函数')
        plt.plot(x5,y5)
        plt.ioff()
        fig1.tight_layout()#调整整体空白
        plt.subplots_adjust(wspace =0.2, hspace =0)#调整子图间距
        plt.show()
def main():
    plt=EffectFun()
    plt.plot()
if __name__ == '__main__':
    main()