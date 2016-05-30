'''
Matplotlib学习
http://old.sebug.net/paper/books/scipydoc/matplotlib_intro.html#id5
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import matplotlib
'''
x = np.linspace(0,10,1000)
y = np.sin(x)
z = np.cos(x**2)

plt.figure(figsize=(8,4))                   # 创建一个绘图对象,并设置为当前的绘图对象
plt.plot(x,y,label="$sin(x)$",color='red',linewidth=2)      # label:给曲线设置名字,加上$会调用内置的latex引擎绘制数学公式,最后会在legen中显示linewidth:指定曲线宽度
plt.plot(x,z,"b--",label="$cos(x^2)$")   # b代表蓝色 --表示虚线  输入plt.plot? 可以查看格式化字符串详细配置
plt.xlabel("Time(s)")
plt.ylabel("Volt")
plt.title("Pyplot First Demo")
plt.ylim((-1.2,1.2))                        # y轴取值范围
plt.xlim(0,10)                              # x轴取值范围
plt.legend()                                # 显示图示
plt.show()

print(matplotlib.rcParams["savefig.dpi"])
'''
# 5.1.1 配置属性

x = np.arange(-10,10,0.1)
line , = plt.plot(x,x*x)            # plot返回一个列表，通过line,获取其第一个元素
line.set_antialiased(False)         # 调用Line2D对象的set_*方法设置属性值,调用Line2D对象line的set_antialiased方法，关闭对象的反锯齿效果
lines = plt.plot(x,np.sin(x),x,np.cos(x))   # 同时绘制sin和cos两条曲线，lines是一个有两个Line2D对象的列表
plt.setp(lines,color="r",linewidth=2.0)     # 调用setp函数同时配置多个Line2D对象的多个属性值
plt.xlim(-10,10)
# plt.show()

# print(plt.getp(lines[1]))                  # 不指定属性名,获取曲线的所有属性
print(plt.getp(lines[0],'color'))          # 指定属性名获取属性值
'''
Figure对象有一个axes属性，其值为AxesSubplot对象的列表，
每个AxesSubplot对象代表图表中的一个子图，前面所绘制的图表只包含一个子图，当前子图也可以通过plt.gca获得：
'''
f = plt.gcf()    # matplotlib的整个图表为一个Figure对象，此对象在调用plt.figure函数时返回，我们也可以通过plt.gcf函数获取当前的绘图对象：
print(plt.getp(f,"axes"))
print(plt.gca())

alllines = plt.getp(plt.gca(),'lines')
print(alllines)
print(alllines[0] == line)
plt.show()
