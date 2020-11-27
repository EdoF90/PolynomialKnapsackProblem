dictHeuMod={}
heu_1="./largerFiles.txt"
with open(heu_1) as f:
	for line in f:
		name=line.split(",")[0]
		obj_heu=line.split(",")[1]
		comp_time_heu=line.split(",")[2]
		dictHeuMod[name]=((float(obj_heu.strip()),float(comp_time_heu.strip())))
dictHeuNonMod={}
heu_2="./largerFiles_no_pen.txt"
with open(heu_2) as f:
	for line in f:
		name=line.split(",")[0]
		obj_heu=line.split(",")[1]
		comp_time_heu=line.split(",")[2]
		dictHeuNonMod[name]=((float(obj_heu.strip()),float(comp_time_heu.strip())))

counter_non_mod=0
counter_mod=0
for name in dictHeuNonMod.keys():
	if dictHeuNonMod[name][0]==-1:
		counter_non_mod+=1
for name in dictHeuMod.keys():
	if dictHeuMod[name][0]==-1:
		counter_mod+=1

print(f"N° of times the heuristic without modifications gave a -1 output: {counter_non_mod} out of {len(dictHeuMod.keys())} results")
print(f"N° of times the heuristic with modifications gave a -1 output: {counter_mod} out of {len(dictHeuNonMod.keys())} results")



