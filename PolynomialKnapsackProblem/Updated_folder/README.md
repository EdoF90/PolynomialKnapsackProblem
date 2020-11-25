# PolynomialKnapsackProblem - UPDATED
This is the updated folder. 
At the moment it contains:

- polynomial_knapsack.py -> script to launch the model (call functions of sub-folder 'solver' and generate an instance through 'Instance.py'; needs also the 'logs' folder to output logs.). At the moment is appending results on a txt file (Results/Model_results); CHECK BEFORE RUNNING!

- GAHeuristic.py -> the genetic heuristic. At the moment is appending results on a txt file (Results/Genetic_results/Second_configs); CHECK BEFORE RUNNING!

- GAHeu_tuning.py -> tool used to tune the hyper-parameters of the Genetic heu.

- DataCreatorNew.py -> script to generate new configs; at the moment is set to produce 200 configs for each numerosity (from 100 to 1500) and there are different policies to handle the synergies (Exponential - Square Root - Linear)

- Unify_model_heu.py -> used to unify the genetic's outputs with the model's one in the same excel file

- Validations_results.py -> used to plot and display results of the genetic's validation

- config_final -> first set of config launched; not generated according to the current policy of the DataCreatorNew; have model results untill number 758.

- config_final_2 -> second set of config launched; most of them are still to be runned but some has already been. At the moment Alessandro is running some of them.

- Divide_config.py -> Simple script to divide the configs to be runned; avoid re-running or run things assigned to other members.
