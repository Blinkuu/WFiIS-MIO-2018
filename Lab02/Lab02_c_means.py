from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import pandas

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

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

centers = [[3, 4.7],
           [4, 4.7],
           [3, 4.7]]

sigmas = [[np.std(tabX1), np.std(tabY)],
          [np.std(tabX2), np.std(tabY)],
          [np.std(tabX3), np.std(tabY)]]

# Set up the loop and plot


xpts = tabX1[0]
ypts = tabY[0]
labels = np.zeros(1)

tab = [tabX1, tabX2, tabX3]

for i in range(3):
    xpts = np.hstack((xpts, tab[i]))
    ypts = np.hstack((ypts, tabY))
    labels = np.hstack((labels, np.ones(255) * i))

fig0, ax0 = plt.subplots()
for label in range(3):
    ax0.plot(xpts[labels == label], ypts[labels == label], '.', color=colors[label], label='X' + str(label + 1))
ax0.set_title('My data')
ax0.set_xlabel("Universe")
ax0.set_ylabel("y")
ax0.legend()

plt.show()

# Set up the loop and plot
fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
alldata = np.vstack((xpts, ypts))
fpcs = []

for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

    # print("Center[" + str(ncenters) +"]", cntr)
    # Store fpc values for later
    fpcs.append(fpc)

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)
    for j in range(ncenters):
        ax.plot(xpts[cluster_membership == j],
                ypts[cluster_membership == j], '.', color=colors[j])

    # Mark the center of each fuzzy cluster
    for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')

    ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    ax.axis('off')

fig1.tight_layout()

plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(np.r_[2:11], fpcs)
ax2.set_xlabel("Number of centers")
ax2.set_ylabel("Fuzzy partition coefficient")

plt.show()

# Using c-means algorithm to get better results

center, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, 3, 2, error=0.005, maxiter=1000)

# Declare input/output
x1 = ctrl.Antecedent(np.arange(min(tabX1), max(tabX1), 0.001), 'x1')
x2 = ctrl.Antecedent(np.arange(min(tabX2), max(tabX2), 0.001), 'x2')
x3 = ctrl.Antecedent(np.arange(min(tabX3), max(tabX3), 0.001), 'x3')
y = ctrl.Consequent(np.arange(min(tabY), max(tabY), 0.001), 'y')

print("center = ", center)
print("center[0][0] = ", center[0][0])

# Make membership functions
x1['lo'] = fuzz.zmf(x1.universe, min(tabX1), center[0][0])
x1['md'] = fuzz.gaussmf(x1.universe, center[0][0], 0.1)
x1['hi'] = fuzz.smf(x1.universe, center[0][0], max(tabX1))

x1.view()

x2['lo'] = fuzz.zmf(x2.universe, min(tabX2), center[1][0])
x2['md'] = fuzz.gaussmf(x2.universe, center[1][0], 0.1)
x2['hi'] = fuzz.smf(x2.universe, center[1][0], max(tabX2))

x2.view()

x3['lo'] = fuzz.zmf(x3.universe, min(tabX3), center[2][0])
x3['md'] = fuzz.gaussmf(x3.universe, center[2][0], 0.1)
x3['hi'] = fuzz.smf(x3.universe, center[2][0], max(tabX3))

x3.view()

y['lo'] = fuzz.zmf(y.universe, 4.2, 4.7)
y['md'] = fuzz.gaussmf(y.universe, 4.7, 0.1)
y['hi'] = fuzz.smf(y.universe, 4.7, 5.2)

# Rules
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
#   y.view(sim=myOutput)

mean_error = mean_error / n
print("MEAN ERROR = ", mean_error)
