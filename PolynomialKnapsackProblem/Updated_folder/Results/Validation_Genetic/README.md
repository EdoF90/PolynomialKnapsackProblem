# Validation_Genetic
This is the folder containing the material to validate the Genetic Heuristic. 
At the moment it contains:

### config_validation
Folder which contains the large configs used for the validation 

### count-1-py
A simple script that counts the number of times in which -1 was emitted as result of the function

### GAHeu_tuning.py
Script which runs the Genetic Heuristic varying some crucial parameters in order to make it more flexible on threating non-feasible solutions from the continuous relaxation

### largerFiles_no_pen.txt & largerFiles.txt
Results containing the objective functions values of the validation

### validation_more_complete.json
Results containing the objective functions and times of the tuning script varying the parameters 'penalization'; 'n_chromosomes'; weight

### Validations_results.py
Script to analyse the json above and obtain useful stats
