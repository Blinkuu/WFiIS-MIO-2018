import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# New Antecedent/Consequent objects hold universe variables and membership
# functions
free_time = ctrl.Antecedent(np.arange(0, 11, 1), 'free_time')
material_amount = ctrl.Antecedent(np.arange(0, 11, 1), 'material_amount')
study_time = ctrl.Consequent(np.arange(0, 5, 1), 'study_time')

free_time['very low'] = fuzz.trimf(free_time.universe, [0, 0, 2])
free_time['low'] = fuzz.trimf(free_time.universe, [1, 3, 5])
free_time['medium'] = fuzz.trimf(free_time.universe, [3, 5, 7])
free_time['high'] = fuzz.trimf(free_time.universe, [5, 7, 9])
free_time['very high'] = fuzz.trimf(free_time.universe, [8, 10, 10])

material_amount['very low'] = fuzz.trimf(material_amount.universe, [0, 0, 2])
material_amount['low'] = fuzz.trimf(material_amount.universe, [1, 3, 5])
material_amount['medium'] = fuzz.trimf(material_amount.universe, [3, 5, 7])
material_amount['high'] = fuzz.trimf(material_amount.universe, [5, 7, 9])
material_amount['very high'] = fuzz.trimf(material_amount.universe, [8, 10, 10])

study_time['very low'] = fuzz.trimf(study_time.universe, [0, 0, 1])
study_time['low'] = fuzz.trimf(study_time.universe, [0, 1, 2])
study_time['medium'] = fuzz.trimf(study_time.universe, [1, 2, 3])
study_time['high'] = fuzz.trimf(study_time.universe, [2, 3, 4])
study_time['very high'] = fuzz.trimf(study_time.universe, [3, 4, 5])

# You can see how these look with .view()

free_time.view()
material_amount.view()
study_time.view()

r0 = ctrl.Rule(free_time['very low'] | material_amount['very low'], study_time['very low'])
r1 = ctrl.Rule(free_time['very low'] | material_amount['low'], study_time['very low'])
r2 = ctrl.Rule(free_time['low'] | material_amount['very low'], study_time['very low'])
r3 = ctrl.Rule(free_time['low'] | material_amount['low'], study_time['low'])
r4 = ctrl.Rule(free_time['very low'] | material_amount['medium'], study_time['low'])
r5 = ctrl.Rule(free_time['medium'], study_time['medium'])
r6 = ctrl.Rule(free_time['very high'] | material_amount['medium'], study_time['high'])
r7 = ctrl.Rule(free_time['high'] | material_amount['very high'], study_time['high'])
r8 = ctrl.Rule(free_time['very high'] | material_amount['high'], study_time['very high'])
r9 = ctrl.Rule(free_time['very high'] | material_amount['very high'], study_time['very high'])

study_time_ctrl = ctrl.ControlSystem([r1,r2,r3,r4,r5,r6,r7,r8,r9])

studying = ctrl.ControlSystemSimulation(study_time_ctrl)

# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
studying.input['free_time'] = 6.2
studying.input['material_amount'] = 3.8

# Crunch the numbers
studying.compute()

print (studying.output['study_time'])
study_time.view(sim=studying)
