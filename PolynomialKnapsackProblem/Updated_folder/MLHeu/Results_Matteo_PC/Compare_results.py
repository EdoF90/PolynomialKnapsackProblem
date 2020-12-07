import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dictHeu010={}
heu_1="./resultsMLHeu_1.txt"
with open(heu_1) as f:
	for line in f:
		name=line.split(",")[0]
		obj_heu=line.split(",")[1]
		comp_time_heu=line.split(",")[2]
		dictHeu010[name]=((float(obj_heu.strip()),float(comp_time_heu.strip())))

dictHeu015={}
heu_1="./resultsMLHeu_2.txt"
with open(heu_1) as f:
	for line in f:
		name=line.split(",")[0]
		obj_heu=line.split(",")[1]
		comp_time_heu=line.split(",")[2]
		dictHeu015[name]=((float(obj_heu.strip()),float(comp_time_heu.strip())))

dictHeu020={}
heu_1="./resultsMLHeu_3.txt"
with open(heu_1) as f:
	for line in f:
		name=line.split(",")[0]
		obj_heu=line.split(",")[1]
		comp_time_heu=line.split(",")[2]
		dictHeu020[name]=((float(obj_heu.strip()),float(comp_time_heu.strip())))

dfs = pd.read_excel("../Comparison_withML.xlsx",header=0,skiprows=1)
names=dfs["Filename"]
objFunModel=dfs["Result "] 

timesModel=dfs["Time"]
dictMod={}
for it in range(len(names)):
	name=names[it]
	dictMod[name]=((float(objFunModel[it]),float(timesModel[it])))

gap_list_010=[]
ratio_list010=[]
count_over_threshold010=0
gap_list_015=[]
ratio_list015=[]
count_over_threshold015=0
gap_list_020=[]
ratio_list020=[]
count_over_threshold020=0

for key in dictMod.keys():
	solMod,timeMod=dictMod[key]
	solHeu,timeHeu=dictHeu010[key]
	gap=(solMod-solHeu)*100/solMod
	ratio_time=timeHeu/timeMod
	gap_list_010.append(gap)
	ratio_list010.append(ratio_time)
	#print(f"For file {key}:\n\t- Gap:\t{gap}\n\t- Ratio:\t{ratio_time}")
	if gap>=5:
		count_over_threshold010+=1
	solHeu,timeHeu=dictHeu015[key]
	gap=(solMod-solHeu)*100/solMod
	ratio_time=timeHeu/timeMod
	gap_list_015.append(gap)
	ratio_list015.append(ratio_time)
	#print(f"For file {key}:\n\t- Gap:\t{gap}\n\t- Ratio:\t{ratio_time}")
	if gap>=5:
		count_over_threshold015+=1
	solHeu,timeHeu=dictHeu020[key]
	gap=(solMod-solHeu)*100/solMod
	ratio_time=timeHeu/timeMod
	gap_list_020.append(gap)
	ratio_list020.append(ratio_time)
	#print(f"For file {key}:\n\t- Gap:\t{gap}\n\t- Ratio:\t{ratio_time}")
	if gap>=5:
		count_over_threshold020+=1
print()
print()
print(f"With 0.10 the Heuristic generated {count_over_threshold010} over-threshold out of {len(dictMod)} attempts")
print(f"Furthermore:\n\t- Average gap: {np.mean(gap_list_010)}\n\t- St_dev gap: {np.std(gap_list_010)}\
		\n\t- Average ratio: {np.mean(ratio_list010)}\n\t- St_dev ratio: {np.std(ratio_list010)}")
print(f"\nWith 0.15 the Heuristic generated {count_over_threshold015} over-threshold out of {len(dictMod)} attempts")
print(f"Furthermore:\n\t- Average gap: {np.mean(gap_list_015)}\n\t- St_dev gap: {np.std(gap_list_015)}\
		\n\t- Average ratio: {np.mean(ratio_list015)}\n\t- St_dev ratio: {np.std(ratio_list015)}")
print(f"\nWith 0.20 the Heuristic generated {count_over_threshold020} over-threshold out of {len(dictMod)} attempts")
print(f"Furthermore:\n\t- Average gap: {np.mean(gap_list_020)}\n\t- St_dev gap: {np.std(gap_list_020)}\
		\n\t- Average ratio: {np.mean(ratio_list020)}\n\t- St_dev ratio: {np.std(ratio_list020)}")
print()
print()
fig, axs = plt.subplots(3,2)
axs[0,0].boxplot(gap_list_010,patch_artist=True)
axs[0,0].plot(np.arange(3),np.ones(3)*5,"r")
axs[0,1].boxplot(ratio_list010,patch_artist=True)
axs[0,1].plot(np.arange(3),np.ones(3)*1,"r")
axs[0,0].set_title("Gap fixing 90%",fontsize=10)
axs[0,1].set_title("Ratio_time fixing 90%",fontsize=10)
axs[1,0].boxplot(gap_list_015,patch_artist=True)
axs[1,0].plot(np.arange(3),np.ones(3)*5,"r")
axs[1,1].boxplot(ratio_list015,patch_artist=True)
axs[1,1].plot(np.arange(3),np.ones(3)*1,"r")
axs[1,0].set_title("Gap fixing 85%",fontsize=10)
axs[1,1].set_title("Ratio_time fixing 85%",fontsize=10)
axs[2,0].boxplot(gap_list_020,patch_artist=True)
axs[2,0].plot(np.arange(3),np.ones(3)*5,"r")
axs[2,1].boxplot(ratio_list020,patch_artist=True)
axs[2,1].plot(np.arange(3),np.ones(3)*1,"r")
axs[2,0].set_title("Gap fixing 80%",fontsize=10)
axs[2,1].set_title("Ratio_time fixing 80%",fontsize=10)
for ax in axs.flat:
    ax.set_xticks([])
fig.suptitle("Boxplots showing outliers",fontsize=14)
plt.show()

fig, axs = plt.subplots(3,2)
axs[0,0].boxplot(gap_list_010,patch_artist=True,showfliers=False)
axs[0,0].plot(np.arange(3),np.ones(3)*5,"r")
axs[0,1].boxplot(ratio_list010,patch_artist=True,showfliers=False)
axs[0,1].plot(np.arange(3),np.ones(3)*1,"r")
axs[0,0].set_title("Gap fixing 90%",fontsize=10)
axs[0,1].set_title("Ratio_time fixing 90%",fontsize=10)
axs[1,0].boxplot(gap_list_015,patch_artist=True,showfliers=False)
axs[1,0].plot(np.arange(3),np.ones(3)*5,"r")
axs[1,1].boxplot(ratio_list015,patch_artist=True,showfliers=False)
axs[1,1].plot(np.arange(3),np.ones(3)*1,"r")
axs[1,0].set_title("Gap fixing 85%",fontsize=10)
axs[1,1].set_title("Ratio_time fixing 85%",fontsize=10)
axs[2,0].boxplot(gap_list_020,patch_artist=True,showfliers=False)
axs[2,0].plot(np.arange(3),np.ones(3)*5,"r")
axs[2,1].boxplot(ratio_list020,patch_artist=True,showfliers=False)
axs[2,1].plot(np.arange(3),np.ones(3)*1,"r")
axs[2,0].set_title("Gap fixing 80%",fontsize=10)
axs[2,1].set_title("Ratio_time fixing 80%",fontsize=10)
for ax in axs.flat:
    ax.set_xticks([])
fig.suptitle("Boxplots not showing outliers",fontsize=14)
plt.show()
