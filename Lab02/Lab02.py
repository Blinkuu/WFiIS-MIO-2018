import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axis3d
from sklearn.metrics import mean_squared_error

# Declare input/output
x1 = ctrl.Antecedent(np.arange(2.5, 3.5, 0.001), 'x1')
x2 = ctrl.Antecedent(np.arange(3.5, 4.5, 0.001), 'x2')
x3 = ctrl.Antecedent(np.arange(2.5, 3.5, 0.001), 'x3')
y = ctrl.Consequent(np.arange(4.2, 5.2, 0.001), 'y')

# Make membership functions
x1['lo'] = fuzz.zmf(x1.universe, 2.5, 3)
x1['md'] = fuzz.gaussmf(x1.universe, 3, 0.1)
x1['hi'] = fuzz.smf(x1.universe, 3, 3.5)

x2['lo'] = fuzz.zmf(x2.universe, 3.5, 4)
x2['md'] = fuzz.gaussmf(x2.universe, 4, 0.1)
x2['hi'] = fuzz.smf(x2.universe, 4, 4.5)

x3['lo'] = fuzz.zmf(x3.universe, 2.5, 3)
x3['md'] = fuzz.gaussmf(x3.universe, 3, 0.1)
x3['hi'] = fuzz.smf(x3.universe, 3, 3.5)

y['lo'] = fuzz.zmf(y.universe, 4.2, 4.7)
y['md'] = fuzz.gaussmf(y.universe, 4.7, 0.1)
y['hi'] = fuzz.smf(y.universe, 4.7, 5.2)

# x1.view()
# x2.view()
# x3.view()
# y.view()

# Make rules
r1 = ctrl.Rule(x1['lo'] & x2['lo'] & x3['lo'] |
               x1['md'] & x2['lo'] & x3['lo'] |
               x1['lo'] & x2['md'] & x3['lo'] |
               x1['lo'] & x2['lo'] & x3['md'],
               y['lo'])
r2 = ctrl.Rule(x1['md'] & x2['md'] & x3['md'] |
               x1['lo'] & x2['md'] & x3['md'] |
               x1['md'] & x2['lo'] & x3['md'] |
               x1['md'] & x2['md'] & x3['lo'] |
               x1['hi'] & x2['md'] & x3['md'] |
               x1['md'] & x2['hi'] & x3['md'] |
               x1['md'] & x2['md'] & x3['hi'],
               y['md'])
r3 = ctrl.Rule(x1['hi'] & x2['hi'] & x3['hi'] |
               x1['md'] & x2['hi'] & x3['hi'] |
               x1['hi'] & x2['md'] & x3['hi'] |
               x1['hi'] & x2['hi'] & x3['md'],
               y['hi'])

myOutput_ctrl = ctrl.ControlSystem([r1, r2, r3])

myOutput = ctrl.ControlSystemSimulation(myOutput_ctrl)

# Read file
tabX1 = []
tabX2 = []
tabX3 = []
tabY = []

file = pandas.read_csv('lab02_data.csv')

for i in file['x1']:
    tabX1.append(float(i))
for i in file['x2']:
    tabX2.append(float(i))
for i in file['x3']:
    tabX3.append(float(i))
for i in file['y']:
    tabY.append(float(i))

'''
fig = plt.figure()
ax = fig.gca(projection='3d')

xValues = tabX1
yValues = tabX2
zValues = tabY

ax.scatter(xValues, yValues, zValues)

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout();

plt.show()
'''

i = 0
n = len(tabY)

mean_error = 0

while i < n:
    myOutput.input['x1'] = tabX1[i]
    myOutput.input['x2'] = tabX2[i]
    myOutput.input['x3'] = tabX3[i]

    myOutput.compute()
    o = myOutput.output['y']
    mean_error += (tabY[i] - o) ** 2
    i += 1
#    y.view(sim=myOutput)

mean_error = mean_error / n
print("MEAN ERROR = ", mean_error)
