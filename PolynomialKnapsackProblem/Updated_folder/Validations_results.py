import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
import time as t
import math

with open("Results/Validation/validation_more_complete_2.json") as f:
	d=json.load(f)

dfs = pd.read_excel("Results/Model_results/new_results(config_final).xlsx", sheet_name="Modello",header=0,skiprows=1)
names=[name.split(".")[0] for name in dfs["name_file"]]
objFunModel=dfs["objfun"] 
timesModel=dfs["comp_time"]
#EXTRACT ALSO THE DATA GENERATED BY THE 'OLD GENETIC' -> OBTAINED FIXING n_chromosomes=100 AND NO SCALING
objFunOldHeu=dfs["objfun.1"] 
timesOldHeu=dfs["comp_time.1"]
dict_model={}
dict_old_heu={}
for it in range(len(names)):
	dict_model[names[it]]=(objFunModel[it],timesModel[it])
	dict_old_heu[names[it]]=(objFunOldHeu[it],timesOldHeu[it])

dict_result={}

for it in range(len(list(d.keys()))): #CYCLING ON THE NAME
	name=list(d.keys())[it]
	file=d[list(d.keys())[it]]
	numerosity=file["70"] #ALREADY EXTRACTING THE NUMEROSIY (FIXED)
	for it2 in range(len(list(numerosity.keys()))): #CYCLING ON THE PENALIZATION
		penalization=list(numerosity.keys())[it2]
		if penalization not in dict_result.keys():
			dict_result[penalization]={}
		for it3 in range(len(list(numerosity[penalization].keys()))): #CYCLING ON THE WEIGHT
			weight=list(numerosity[penalization].keys())[it3]
			if weight not in dict_result[penalization].keys():
				dict_result[penalization][weight]={
				"gap_result":[],
				"gap_time":[],
				"counter_no_result":0,
				"counter_model_no_result":0
				}
			#print(f"For file {name} and penalization {penalization} and weight {weight} the result is {numerosity[penalization][weight][0]}")
			#Per ciascuna penalità e peso, salva gap in tempo e percentuale e soluzioni =-1
			resultModel=numerosity[penalization][weight][0][0]
			resultHeu=numerosity[penalization][weight][0][1]
			timeHeu=numerosity[penalization][weight][1]
			timeModel=dict_model[name][1]
			#print(name,resultModel,resultHeu,timeModel,timeHeu)
			
			#print((timeModel-timeHeu)/timeModel)
			#exit()
			if int(name.split("_")[1])>50:
				if not math.isnan((timeModel-timeHeu)/timeModel) and resultModel!=-1 and resultHeu!=-1:
					#print(name,timeModel,timeHeu)
					#print(resultModel,resultHeu)
					dict_result[penalization][weight]["gap_time"].append(((timeHeu)/timeModel)*100)
					dict_result[penalization][weight]["gap_result"].append(((resultModel-resultHeu)/resultModel)*100)
					if ((resultModel-resultHeu)/resultModel)*100>10:
						print(name,((timeHeu)/timeModel)*100,((resultModel-resultHeu)/resultModel)*100)
					#exit()
				elif resultModel==-1 and not math.isnan((timeModel-timeHeu)/timeModel) and resultHeu!=-1:
					dict_result[penalization][weight]["counter_model_no_result"]+=1
				elif resultModel!=-1 and not math.isnan((timeModel-timeHeu)/timeModel) and resultHeu==-1:
					dict_result[penalization][weight]["counter_no_result"]+=1

#fig, (ax1, ax2) = plt.subplots(1,2)
fig, (ax2) = plt.subplots(1,figsize=(30,30))
fig.suptitle('Boxplots grid search')
stats_dic={}
it=0
step=0
x_ax_label=[]
for penalization in dict_result.keys():
	stats_dic[penalization]={}
	for weight in dict_result[penalization].keys():
		stats_dic[penalization][weight]={
		"avg_gap_results":np.mean(dict_result[penalization][weight]["gap_result"]),
		"std_gap_results":np.std(dict_result[penalization][weight]["gap_result"]),
		"avg_gap_times":np.mean(dict_result[penalization][weight]["gap_time"]),
		"std_gap_times":np.std(dict_result[penalization][weight]["gap_time"]),
		"counter_no_result":dict_result[penalization][weight]["counter_no_result"],
		"counter_model_no_result":dict_result[penalization][weight]["counter_model_no_result"],
		"min_result":np.min(dict_result[penalization][weight]["gap_result"]),
		"max_result":np.max(dict_result[penalization][weight]["gap_result"])
		}
		#print(stats_dic[penalization][weight]['avg_gap_results'])
		print(f"\nFor pen:{penalization} and weight:{weight} we have:")
		print(f"\tAVG gap result:{stats_dic[penalization][weight]['avg_gap_results']}")
		print(f"\tSTD gap result:{stats_dic[penalization][weight]['std_gap_results']}")
		print(f"\tAVG gap time:{stats_dic[penalization][weight]['avg_gap_times']}")
		print(f"\tSTD gap time:{stats_dic[penalization][weight]['std_gap_times']}")
		print(f"\tNumber of missed results:{stats_dic[penalization][weight]['counter_no_result']}")
		it+=1
		x_ax_label.append(weight)
		#ax1.boxplot(dict_result[penalization][weight]["gap_result"],labels=[f"p={penalization};w={weight}"],positions=[it])

	x_ax=np.arange(step,step+len(stats_dic[penalization].keys()))

	#ax1.plot(x_ax,[stats_dic[penalization][weight]["avg_gap_results"] for weight in stats_dic[penalization].keys()],label=f"p={penalization}")
	ax2.plot(x_ax,[stats_dic[penalization][weight]["avg_gap_times"] for weight in stats_dic[penalization].keys()],label=f"p={penalization}")
	#ax1.scatter(x_ax,[stats_dic[penalization][weight]["avg_gap_results"] for weight in stats_dic[penalization].keys()])
	ax2.scatter(x_ax,[stats_dic[penalization][weight]["avg_gap_times"] for weight in stats_dic[penalization].keys()])	
	#ax1.fill_between(x_ax,[stats_dic[penalization][weight]["max_result"] for weight in stats_dic[penalization].keys()],\
	#	[stats_dic[penalization][weight]["min_result"] for weight in stats_dic[penalization].keys()],alpha=0.5)
	
	ax2.fill_between(x_ax,[stats_dic[penalization][weight]["avg_gap_times"]-\
		2*stats_dic[penalization][weight]["std_gap_times"] for weight in stats_dic[penalization].keys()],\
		[stats_dic[penalization][weight]["avg_gap_times"]+\
		2*stats_dic[penalization][weight]["std_gap_times"] for weight in stats_dic[penalization].keys()],
		alpha=0.5)

	"""
	ax2.fill_between(x_ax,[stats_dic[penalization][weight]["avg_gap_times"]+\
		2*stats_dic[penalization][weight]["std_gap_times"] for weight in stats_dic[penalization].keys()],\
		alpha=0.5)
	ax2.fill_between(x_ax,[stats_dic[penalization][weight]["avg_gap_times"]-\
		2*stats_dic[penalization][weight]["std_gap_times"] for weight in stats_dic[penalization].keys()],\
		alpha=0.5)
	"""
	step=it

#ax1.set_title("Relative residuals")
ax2.set_title("Times taken")
#ax1.legend()
#plt.sca(ax1)
#plt.xticks(range(it), x_ax_label,rotation=45,fontsize=8)
#print(it)
#print(x_ax_label)
plt.sca(ax2)
plt.xticks(range(it), x_ax_label,rotation=45,fontsize=8)
plt.ylim(0)
#ax1.set_xticks(np.arange(it),x_ax_label)
#ax1.set_xticklabels(x_ax_label)
ax2.legend()
#ax2.set_xticks(np.arange(it),x_ax_label)
#ax2.set_xticklabels(x_ax_label)
plt.show()	
