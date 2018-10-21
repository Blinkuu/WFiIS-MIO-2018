import numpy as np 
import skfuzzy as fuzz
import matplotlib.pyplot as plt

#Setting up input/output variables

x_free_time = np.arange(0,11,1)
x_material_amount = np.arange(0,11,1)
x_study_time = np.arange(0,5,1)

free_time_lo = fuzz.trimf(x_free_time,[0,0,5])
free_time_md = fuzz.trimf(x_free_time,[0,5,10])
free_time_hi = fuzz.trimf(x_free_time,[5,10,10])

material_amount_lo = fuzz.trimf(x_material_amount,[0,0,5])
material_amount_md = fuzz.trimf(x_material_amount,[0,5,10])
material_amount_hi = fuzz.trimf(x_material_amount,[5,10,10])

study_time_lo = fuzz.trimf(x_study_time,[0,0,2])
study_time_md = fuzz.trimf(x_study_time,[0,2,4])
study_time_hi = fuzz.trimf(x_study_time,[2,4,4])

#Making plots

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(9, 9))

ax0.plot(x_free_time, free_time_lo, 'b', linewidth=1.5, label='Low')
ax0.plot(x_free_time, free_time_md, 'g', linewidth=1.5, label='Mid')
ax0.plot(x_free_time, free_time_hi, 'r', linewidth=1.5, label='High')
ax0.set_title('Free time')
ax0.legend()

ax1.plot(x_material_amount, material_amount_lo, 'b', linewidth=1.5, label='Few')
ax1.plot(x_material_amount, material_amount_md, 'g', linewidth=1.5, label='Acceptable')
ax1.plot(x_material_amount, material_amount_hi, 'r', linewidth=1.5, label='A lot')
ax1.set_title('Material amount')
ax1.legend()

ax2.plot(x_study_time, study_time_lo, 'b', linewidth=1.5, label='Low')
ax2.plot(x_study_time, study_time_md, 'g', linewidth=1.5, label='Mid')
ax2.plot(x_study_time, study_time_hi, 'r', linewidth=1.5, label='High')
ax2.set_title('Optimal study time')
ax2.legend()

for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

#plt.show()

#Non int values

#print('Enter your free time (0-10)')
#free_time_input = input()
#print('Enter your material amount (0-10)')
#material_amount_input = input()

free_time_input = 6.2
material_amount_input = 3.8

free_time_level_lo = fuzz.interp_membership(x_free_time, free_time_lo, free_time_input)
free_time_level_md = fuzz.interp_membership(x_free_time, free_time_md, free_time_input)
free_time_level_hi = fuzz.interp_membership(x_free_time, free_time_hi, free_time_input)

material_amount_level_lo = fuzz.interp_membership(x_material_amount, material_amount_lo, material_amount_input)
material_amount_level_md = fuzz.interp_membership(x_material_amount, material_amount_md, material_amount_input)
material_amount_level_hi = fuzz.interp_membership(x_material_amount, material_amount_hi, material_amount_input)


#Activating rules

#rule 1: IF AMOUNT OF TIME LOW AND MATERIAL AMOUNT LOW => OPTIMAL STUDY TIME LOW
activate_rule1 = np.fmax(free_time_level_lo, material_amount_level_lo)

study_time_activation_lo = np.fmin(activate_rule1,study_time_lo)

#rule 2: IF MATERIAL AMOUNT IS MID => OPTIMAL STUDY TIME MID
study_time_activation_md = np.fmin(material_amount_level_md,study_time_md)

#rule 3: IF FREE TIME IS HIGH AND MATERIAL AMOUNT IS HIGH => OPTIMAL STUDY TIME HIGH
activate_rule3 = np.fmax(free_time_level_hi,material_amount_level_hi)

study_time_activation_hi = np.fmin(activate_rule3,study_time_hi)

study_time0 = np.zeros_like(x_study_time)

#Visualize the output membership activity

fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(x_study_time, study_time0, study_time_activation_lo, facecolor='b', alpha=0.7)
ax0.plot(x_study_time, study_time_lo, 'b', linewidth=0.5, linestyle='--', )

ax0.fill_between(x_study_time, study_time0, study_time_activation_md, facecolor='g', alpha=0.7)
ax0.plot(x_study_time, study_time_md, 'g', linewidth=0.5, linestyle='--')

ax0.fill_between(x_study_time, study_time0, study_time_activation_hi, facecolor='r', alpha=0.7)
ax0.plot(x_study_time, study_time_hi, 'r', linewidth=0.5, linestyle='--')

ax0.set_title('Output membership activity')

plt.tight_layout()

#plt.show()

aggregated = np.fmax(study_time_activation_lo, np.fmax(study_time_activation_md, study_time_activation_hi))

study_time = fuzz.defuzz(x_study_time,aggregated,'centroid')
study_time_activation = fuzz.interp_membership(x_study_time,aggregated,study_time) #This is for plot

fig, ax0 = plt.subplots(figsize=(8,3))

ax0.plot(x_study_time, study_time_lo, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(x_study_time, study_time_md, 'g', linewidth=0.5, linestyle='--')
ax0.plot(x_study_time, study_time_hi, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(x_study_time, study_time0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([study_time, study_time], [0, study_time_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated membership and result (line)')

plt.tight_layout()


print(study_time)
plt.show()