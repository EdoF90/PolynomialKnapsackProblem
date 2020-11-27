import matplotlib.pyplot as plt
import pandas as pd
import json

file_json="Results/Model_results/model_first_configs.json"
heu_1="Results/Genetic_results/First_configs/results_GAHeu_modified.txt"
heu_2="Results/Genetic_results/First_configs/results_GAHeu_non_modified.txt"

with open(file_json) as f:
	model_results=json.load(f)

dictHeuMod={}

with open(heu_1) as f:
	for line in f:
		name=line.split(",")[0]
		obj_heu=line.split(",")[1]
		comp_time_heu=line.split(",")[2]
		dictHeuMod[name]=((float(obj_heu.strip()),float(comp_time_heu.strip())))

dictHeuNonMod={}

with open(heu_2) as f:
	for line in f:
		name=line.split(",")[0]
		obj_heu=line.split(",")[1]
		comp_time_heu=line.split(",")[2]
		dictHeuNonMod[name]=((float(obj_heu.strip()),float(comp_time_heu.strip())))

excel_file={
	"Name":[],
	"Result_Model":[],
	"Result_Heu":[],
	"Result_Heu2":[],
	"Time_Model":[],
	"Time_Heu":[],
	"Time_Heu2":[],
	"Diff_Time_Model_Heu":[],
	"Diff_Time_Model_Heu2":[],
	"PercentageResHeu1":[],
	"PercentageResHeu2":[]
}


for name in model_results:
	excel_file["Name"].append(name)
	excel_file["Result_Model"].append(model_results[name][0])
	excel_file["Result_Heu"].append(dictHeuNonMod[name][0])
	excel_file["Result_Heu2"].append(dictHeuMod[name][0])
	excel_file["Time_Model"].append(model_results[name][1])
	excel_file["Time_Heu"].append(dictHeuNonMod[name][1])
	excel_file["Time_Heu2"].append(dictHeuMod[name][1])
	excel_file["Diff_Time_Model_Heu"].append(round(model_results[name][1]-dictHeuNonMod[name][1],3))
	excel_file["Diff_Time_Model_Heu2"].append(round(model_results[name][1]-dictHeuMod[name][1],3))
	excel_file["PercentageResHeu1"].append((model_results[name][0]-dictHeuNonMod[name][0])/model_results[name][0])
	excel_file["PercentageResHeu2"].append((model_results[name][0]-dictHeuMod[name][0])/model_results[name][0])

df2=pd.DataFrame(excel_file,columns=["Name","Result_Model","Result_Heu","Result_Heu2",\
	"PercentageResHeu1","PercentageResHeu2","Time_Model","Time_Heu","Time_Heu2",\
	"Diff_Time_Model_Heu","Diff_Time_Model_Heu2"])
df2.to_excel("Results/Comparisons/First_configs_comparison.xlsx")