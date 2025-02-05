# Experiments T6.3.2 - D6.3. Topology-based optimization of robot fleet behavior. Detection of stable topological patterns using persistent entropy

This repository contains data and experiments associated to "D6.3. Topology-based optimization of robot fleet behavior" performed for the European Project REXASI-PRO (REliable & eXplainable Swarm Intelligence for People with Reduced mObility) (HORIZON-CL4-HUMAN-01 programme under grant agreement nº101070028), in concrete to subtask 6.3.2 and line 1. It has been created by the CIMAgroup research team at the University of Seville, Spain.

## Repository structure

- **ExploratoryAnalysisWithouTopology Folder**: It contains an exploratory analysis for behavior comparation.
- **RealScenarios Folder**: It contains a realistic simulation environment for different scenarios: Cross and corridor.
- **T6-3-2-Experiments Folder**: It contains the experiments that have developed for the specific purpose. It contains a folder for **Cross Scenario** and another one for **Corridor Scenario**. Each of them contains different notebooks, for different behaviors, comparing them, and for predicting collisions. In this folder, it is also a folder called **IlustrationDeliverable**, where images for the deliverable have been created.
- **TrajectoryAnalysis Folder**: It contains experiments analyzing robot trajectories and their relation with collisions.
- **Two_or_more_type_agents**: It contains a simulation example with two type of agents.
- **function.py**: Contains some functions that are useful and will be used in the rest of files.
- 
## Usage
1) Clone this repository and create a virtual enviromment:

```bash
python3 -m venv entorno python=3.10.12
```

2) Activate the virtual enviromment:

```bash
source entorno/bin/activate
```

3) Install Navground (we have used version 0.3.3, so using other version may vary or produces error):

```bash
pip install navground[all]
```

4) Install the necessary dependencies:

```bash
pip install jupyter notebook matplotlib scipy multiprocess gudhi plotly scikit-learn pandas ripser seaborn tqdm
```

Ok, you can now run experiments! :)

**Note:** In case of issues doing that in WSL, reinstall using the following distribution and reinstall Navground with the steps mentioned earlier:

1. Remove the entire Ubuntu distribution: wsl --unregister distribution

2. Install the WSL distribution: wsl --install -d Ubuntu-22.04

3. sudo apt-get update

4. wsl --set-default-version 2

5. 
## Citation  and reference

If you want to use our code for your experiments or for analyzing it, please cite our paper as follows:

Toscano-Duran V, Perera-Lago J, Torras-Casas A, Gonzalez-Diaz, R. An in-depth topology analysis and optimization of robot fleet behavior. Open Research Europe (Submitted, pending of acceptance and publication)

For further information, please contact us at: vtoscano@us.es


