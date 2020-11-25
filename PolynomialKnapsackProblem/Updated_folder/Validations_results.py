import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
import time as t

with open("Results/Validation/validation_most_numerous.json") as f:
	d=json.load(f)

dfs = pd.read_excel("Results/Model_results/new_results(config_final).xlsx", sheet_name="Modello",header=0,skiprows=1)
names=[name.split(".")[0] for name in dfs["name_file"]]

objFunPerc=dfs["%OF"]
timesOldHeu=dfs["comp_time.1"]
dict_model={}
for it in range(len(names)):
	dict_model[names[it]]=(objFunPerc[it],timesOldHeu[it])
#X = np.arange(len(list(d.keys())))

sum={
	"50":0,
	"60":0,
	"70":0,
	"80":0,
	"90":0,
	"100":0
}

times={
	"50":0,
	"60":0,
	"70":0,
	"80":0,
	"90":0,
	"100":0
}

sumList={
	"50":[],
	"60":[],
	"70":[],
	"80":[],
	"90":[],
	"100":[],
	"100*":[]
}

timesList={
	"50":[],
	"60":[],
	"70":[],
	"80":[],
	"90":[],
	"100":[],
	"100*":[]
}

plots={
	"50":[],
	"60":[],
	"70":[],
	"80":[],
	"90":[],
	"100":[],
}

counter=0
for it in range(len(list(d.keys()))):
	file=d[list(d.keys())[it]]
	#print()
	#barWidth=0.0
	for it2 in range(len(list(file.keys()))):
		n_chrom=list(file.keys())[it2]
		value=file[n_chrom]
		difference=(value[0][0]-value[0][1])
		differencePercentage=difference/value[0][0]
		if differencePercentage<1:
			#print("For file {} and n_chomosomes {} we had: {}".format(it,n_chrom,value))
			plots[str(n_chrom)].append(differencePercentage)
			sum[str(n_chrom)]+=differencePercentage
			sumList[str(n_chrom)].append(differencePercentage)
			time=value[1]
			times[str(n_chrom)]+=time
			timesList[str(n_chrom)].append(time)
			counter+=1
	
	infoOld=dict_model[list(d.keys())[it]]
	if infoOld[0]/100<1:
		timesList["100*"].append(infoOld[1])
		sumList["100*"].append(infoOld[0]/100)
	#print(infoOld[1],infoOld[0]/100)
	#t.sleep(5)

print("The number of file analysed is {}".format(counter))

plt.figure()
#colours=["r","b","k","y","g","orange"]
labels=["50","60","70","80","90","100"]
#l=[str(i) for i in [np.arange(len(list(d.keys())))]]
for key in plots.keys():
	X=np.arange(len(plots[key]))
	plt.plot(X,plots[key])
	#plt.scatter(X,plots[key])

plt.title("Difference percentage varying n* of initial chromosomes")
plt.legend(labels=labels)
plt.plot(X,np.ones(len(X))*0.05)
plt.ylabel("Percentage residual")
#plt.ylim(0,0.05)
plt.xlabel("File Runned")
plt.show()

fig, axs = plt.subplots(2)
fig.suptitle('Statistics per number of chromosomes')
axs[0].plot(sum.keys(),sum.values())
axs[0].scatter(sum.keys(),sum.values())
axs[0].set_title("Sum of relative residuals")
axs[1].plot(times.keys(),times.values())
axs[1].scatter(times.keys(),times.values())
axs[1].set_title("Sum of times taken")
plt.show()

"""
plt.figure()
plt.boxplot(sumList.values(),labels=labels,patch_artist=True)
plt.ylabel('Percentage residual')
plt.xlabel('N° of chromosomes')
plt.xticks(rotation = 45)
plt.grid()
plt.title("Residuals varying number of chromosomes")
plt.show()
"""
labels=["50","60","70","80","90","100","100*"]
fig, (ax1, ax2) = plt.subplots(1,2)
fig.suptitle('Boxplots per number of chromosomes')
ax1.boxplot(sumList.values(),labels=labels,patch_artist=True)
ax1.set_title("Relative residuals")
ax2.boxplot(timesList.values(),labels=labels,patch_artist=True)
ax2.set_title("Times taken")
plt.show()
