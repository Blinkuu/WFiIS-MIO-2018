import numpy as np
import skfuzzy.control as ctrl
import skfuzzy as fuzz

universe = np.arange(0, 11, 1)

# Create the three fuzzy variables - two inputs, one output
free_time = ctrl.Antecedent(universe, 'free_time')
material_amount = ctrl.Antecedent(universe, 'material_amount')
study_time = ctrl.Consequent(np.arange(0, 5, 0.5), 'study_time')

# Here we use the convenience `automf` to populate the fuzzy variables with
# terms. The optional kwarg `names=` lets us specify the names of our Terms.
names = ['vvl', 'vl', 'l', 'm', 'h','vh','vvh']

'''
free_time['vvl'] = fuzz.trimf(free_time.universe, [0, 0, 2])
free_time['vl'] = fuzz.trimf(free_time.universe, [0, 2, 4])
free_time['l'] = fuzz.trimf(free_time.universe, [2, 4, 6])
free_time['m'] = fuzz.trimf(free_time.universe, [4, 5, 6])
free_time['h'] = fuzz.trimf(free_time.universe, [4, 6, 8])
free_time['vh'] = fuzz.trimf(free_time.universe, [6, 8, 10])
free_time['vvh'] = fuzz.trimf(free_time.universe, [8, 10, 10])

free_time.view()
'''
material_amount['vvl'] = fuzz.trimf(material_amount.universe, [0, 0, 2])
material_amount['vl'] = fuzz.trimf(material_amount.universe, [0, 2, 4])
material_amount['l'] = fuzz.trimf(material_amount.universe, [2, 4, 6])
material_amount['m'] = fuzz.trimf(material_amount.universe, [4, 5, 6])
material_amount['h'] = fuzz.trimf(material_amount.universe, [4, 6, 8])
material_amount['vh'] = fuzz.trimf(material_amount.universe, [6, 8, 10])
material_amount['vvh'] = fuzz.trimf(material_amount.universe, [8, 10, 10])

material_amount.view()

study_time['vvl'] = fuzz.trimf(study_time.universe, [0, 0.5, 1])
study_time['vl'] = fuzz.trimf(study_time.universe, [0.5, 1, 1.5])
study_time['l'] = fuzz.trimf(study_time.universe, [1, 1.5, 2])
study_time['m'] = fuzz.trimf(study_time.universe, [1.5, 2, 2.5])
study_time['h'] = fuzz.trimf(study_time.universe, [2, 2.5, 3])
study_time['vh'] = fuzz.trimf(study_time.universe, [2.5, 3, 3.5])
study_time['vvh'] = fuzz.trimf(study_time.universe, [3, 3.5, 4])

study_time.view()


free_time.automf(names=names)
#material_amount.automf(names=names)
#study_time.automf(names=names)


r1 = ctrl.Rule(antecedent=( (free_time['vvl'] & material_amount['vvl']) |
                            (free_time['vvl'] & material_amount['vl']) |
                            (free_time['vl'] & material_amount['vvl'])),
                  consequent=study_time['vvl'], label='rule vvl')

r2 = ctrl.Rule(antecedent=( (free_time['vl'] & material_amount['vl']) |
                            (free_time['vl'] & material_amount['l']) |
                            (free_time['l'] & material_amount['vl'])),
                  consequent=study_time['vl'], label='rule vl')

r3 = ctrl.Rule(antecedent=( (free_time['l'] & material_amount['l']) |
                            (free_time['l'] & material_amount['m']) |
                            (free_time['m'] & material_amount['l'])),
                  consequent=study_time['l'], label='rule l')

r4 = ctrl.Rule(antecedent=( (free_time['m'] & material_amount['m']) |
                            (free_time['l'] & material_amount['m']) |
                            (free_time['m'] & material_amount['l'])),
                  consequent=study_time['m'], label='rule m')

r5 = ctrl.Rule(antecedent=( (free_time['h'] & material_amount['h']) |
                            (free_time['vh'] & material_amount['h']) |
                            (free_time['h'] & material_amount['vh'])),
                  consequent=study_time['h'], label='rule h')

r6 = ctrl.Rule(antecedent=( (free_time['vh'] & material_amount['vh']) |
                            (free_time['vh'] & material_amount['vvh']) |
                            (free_time['vvh'] & material_amount['vh'])),
                  consequent=study_time['vh'], label='rule vh')

r7 = ctrl.Rule(antecedent=( (free_time['vvh'] & material_amount['vvh'])),
                  consequent=study_time['vvh'], label='rule vvh')

system = ctrl.ControlSystem(rules=[r1,r2,r3,r4,r5,r6,r7])

sim = ctrl.ControlSystemSimulation(system)

# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
sim.input['free_time'] = 6.2
sim.input['material_amount'] = 3.8

# Crunch the numbers
sim.compute()

print (sim.output['study_time'])
study_time.view(sim=sim)