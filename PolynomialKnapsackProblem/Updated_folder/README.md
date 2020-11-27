# PolynomialKnapsackProblem - UPDATED
This is the updated folder. 
At the moment it contains:

### polynomial_knapsack.py:
script to launch the model (call functions of sub-folder 'solver' and generate an instance through 'Instance.py'; needs also the 'logs' folder to output logs.). At the moment is appending results on a txt file (Results/Model_results); CHECK BEFORE RUNNING!

### GAHeuristic.py:
the genetic heuristic. At the moment is appending results on a txt file; it requires as input both the input-folder (config) and the output one -> CHECK BEFORE RUNNING!

### DataCreatorNew.py:
script to generate new configs; at the moment is set to produce 200 configs for each numerosity (from 100 to 1500) and there are different policies to handle the synergies (Exponential - Square Root - Linear)

### config_final
Set of config used to validate (both GAHeuristic and MLHeuristic). We have availability for the model's results for each one of the config belonging to this folder

### config_final_2:
Set of config to display results of Model and Heuristic and to perform comparisons between them. Generated according to the specifications of the DataCreatorNew. At the moment Alessandro ran over 400 config on the Model

### Divide_config.py
Simple script to divide the configs to be runned; avoid re-running or run things assigned to other members.
