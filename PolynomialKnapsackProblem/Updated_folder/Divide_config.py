import os
import shutil

new_folder="config_matteo/" #THE FOLDER IN WHICH THE ASSIGNED CONFIG WILL BE SAVED -> CHANGE THE NAME!
try:
    os.mkdir(new_folder)
except OSError:
    print ("Creation of the directory %s failed" % new_folder)
else:
    print ("Successfully created the directory %s " % new_folder)

list_of_files = os.listdir("config_final_2") #LIST THE CONFIG FILES

#EACH MEMBER HAS TO CREATE ITS OWN FOLDER SO THAT THE SCRIPT KNOWS WHICH FILES ARE TO BE AVOIDED
#assignedToOthers = os.listdir("configs_of_...") 

#CHECK WHICH ARE THE FILES ALREADY RUNNED
alreadyRunned=[]
with open("Results/Model_results/model_second_configs.json") as f:
	for line in f:
		name=line.split(",")[0]
		alreadyRunned.append(name)

#THE SCRIPT WILL TRY TO ASSIGN A CERTAIN NUMBER OF CONFIG
n_config=200
moved=0

for element in list_of_files:
	if element not in alreadyRunned: #and element not in assignedToOthers: #DECOMMENT TO HAVE ALSO THE SECOND CONSTRAINT
		if moved<n_config:
			shutil.copyfile("config_final_2/"+element,new_folder+element)
			moved+=1
		else:
			break

print(f"Successfully moved {moved} configs!")