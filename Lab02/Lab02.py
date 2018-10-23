import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axis3d
from sklearn.metrics import mean_squared_error

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

# Declare input/output
x1 = ctrl.Antecedent(np.arange(min(tabX1), max(tabX1), 0.001), 'x1')
x2 = ctrl.Antecedent(np.arange(min(tabX2), max(tabX2), 0.001), 'x2')
x3 = ctrl.Antecedent(np.arange(min(tabX3), max(tabX3), 0.001), 'x3')
y = ctrl.Consequent(np.arange(min(tabY), max(tabY), 0.001), 'y')

# Make membership functions
x1['lo'] = fuzz.zmf(x1.universe, min(tabX1), float((max(tabX1) + min(tabX1)) / 2))
x1['md'] = fuzz.gaussmf(x1.universe, float((max(tabX1) + min(tabX1)) / 2), 0.05)
x1['hi'] = fuzz.smf(x1.universe, float((max(tabX1) + min(tabX1)) / 2), max(tabX1))

x2['lo'] = fuzz.zmf(x2.universe, min(tabX2), float((max(tabX2) + min(tabX2)) / 2))
x2['md'] = fuzz.gaussmf(x2.universe, float((max(tabX2) + min(tabX2)) / 2), 0.05)
x2['hi'] = fuzz.smf(x2.universe, float((max(tabX2) + min(tabX2)) / 2), max(tabX2))

x3['lo'] = fuzz.zmf(x3.universe, min(tabX3), float((max(tabX3) + min(tabX3)) / 2))
x3['md'] = fuzz.gaussmf(x3.universe, float((max(tabX3) + min(tabX3)) / 2), 0.05)
x3['hi'] = fuzz.smf(x3.universe, float((max(tabX3) + min(tabX3)) / 2), max(tabX3))

y['lo'] = fuzz.zmf(y.universe, min(tabY), float((max(tabY) + min(tabY)) / 2))
y['md'] = fuzz.gaussmf(y.universe, float((max(tabY) + min(tabY)) / 2), 0.05)
y['hi'] = fuzz.smf(y.universe, float((max(tabY) + min(tabY)) / 2), max(tabY))

x1.view()
x2.view()
x3.view()
y.view()

# Checking how to construct rules (spaghetti code)
x1MF = []

for i in range(255):
    membership_lo = fuzz.interp_membership(x1.universe, x1['lo'].mf, tabX1[i])
    membership_md = fuzz.interp_membership(x1.universe, x1['md'].mf, tabX1[i])
    membership_hi = fuzz.interp_membership(x1.universe, x1['hi'].mf, tabX1[i])
    maximum = max(membership_lo, membership_md, membership_hi)
    if maximum == membership_lo:
        x1MF.append('lo')
    if maximum == membership_md:
        x1MF.append('md')
    if maximum == membership_hi:
        x1MF.append('hi')

x2MF = []

for i in range(255):
    membership_lo = fuzz.interp_membership(x2.universe, x2['lo'].mf, tabX2[i])
    membership_md = fuzz.interp_membership(x2.universe, x2['md'].mf, tabX2[i])
    membership_hi = fuzz.interp_membership(x2.universe, x2['hi'].mf, tabX2[i])
    maximum = max(membership_lo, membership_md, membership_hi)
    if maximum == membership_lo:
        x2MF.append('lo')
    if maximum == membership_md:
        x2MF.append('md')
    if maximum == membership_hi:
        x2MF.append('hi')

x3MF = []

for i in range(255):
    membership_lo = fuzz.interp_membership(x3.universe, x3['lo'].mf, tabX3[i])
    membership_md = fuzz.interp_membership(x3.universe, x3['md'].mf, tabX3[i])
    membership_hi = fuzz.interp_membership(x3.universe, x3['hi'].mf, tabX3[i])
    maximum = max(membership_lo, membership_md, membership_hi)
    if maximum == membership_lo:
        x3MF.append('lo')
    if maximum == membership_md:
        x3MF.append('md')
    if maximum == membership_hi:
        x3MF.append('hi')

yMF = []

for i in range(255):
    membership_lo = fuzz.interp_membership(y.universe, y['lo'].mf, tabY[i])
    membership_md = fuzz.interp_membership(y.universe, y['md'].mf, tabY[i])
    membership_hi = fuzz.interp_membership(y.universe, y['hi'].mf, tabY[i])
    maximum = max(membership_lo, membership_md, membership_hi)
    if maximum == membership_lo:
        yMF.append('lo')
    if maximum == membership_md:
        yMF.append('md')
    if maximum == membership_hi:
        yMF.append('hi')

support = ['lo', 'md', 'hi']

for i in range(3):
    for j in range(3):
        for k in range(3):
            lo_c = 0
            md_c = 0
            hi_c = 0
            for x in range(255):
                if x1MF[x] == support[i] and x2MF[x] == support[j] and x3MF[x] == support[k]:
                    if yMF[x] == 'lo':
                        lo_c += 1
                    if yMF[x] == 'md':
                        md_c += 1
                    if yMF[x] == 'hi':
                        hi_c += 1
            print(support[i], " ", support[j], " ", support[k])
            maks = max(lo_c, md_c, hi_c)
            print("    ", maks)
            if maks == lo_c:
                print("LOW!\n")
            if maks == md_c:
                print("MID!\n")
            if maks == hi_c:
                print("HIGH!\n")

# Make rules
r1 = ctrl.Rule(x1['lo'] & x2['lo'] & x3['lo'] |
               x1['lo'] & x2['lo'] & x3['md'] |
               x1['lo'] & x2['md'] & x3['md'],
               y['lo'])
r2 = ctrl.Rule(x1['lo'] & x2['md'] & x3['hi'] |
               x1['lo'] & x2['hi'] & x3['lo'] |
               x1['md'] & x2['md'] & x3['lo'] |
               x1['md'] & x2['md'] & x3['md'] |
               x1['md'] & x2['md'] & x3['hi'],
               y['md'])
r3 = ctrl.Rule(x1['hi'] & x2['hi'] & x3['hi'] |
               x1['hi'] & x2['md'] & x3['hi'] |
               x1['hi'] & x2['md'] & x3['md'] |
               x1['md'] & x2['hi'] & x3['hi'] |
               x1['md'] & x2['hi'] & x3['md'] |
               x1['hi'] & x2['lo'] & x3['md'],
               y['hi'])

myOutput_ctrl = ctrl.ControlSystem([r1, r2, r3])

myOutput = ctrl.ControlSystemSimulation(myOutput_ctrl)

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
