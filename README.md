# Trajectory Misalignment Analysis

This repository contains the code and partial results used for the **misalignment analysis** presented in the main body and appendix of our research paper. The scripts provided here generate key results related to action sequences, hierarchical structures, and state-space graph analysis. Note that **user trajectory data is not included** due to confidentiality.

## Repository Structure

- **`action_sequence_parser.py`**: Parses and processes action sequences for analysis.
- **`classify.py`**: Classifies data based on predefined criteria for misalignment detection.
- **`degree_dist_calculator.py`**: Computes degree distributions of graphs to analyze structural properties.
- **`degree_dist.zip`**: Precomputed degree distribution data used in the analysis.
- **`grapher_with_abstract_actions.py`**: Generates graphs incorporating abstract actions.
- **`hierarchical_analysis.ipynb`**: Jupyter Notebook containing hierarchical analysis of the state-space.
- **`state_space_graph_maker_with_abstract_actions.py`**: Creates state-space graphs using abstract actions.
- **`state_space_graphs_with_abstract_actions.zip`**: Precomputed state-space graphs for further analysis.
- **`trajectory_visualization.py`**: Visualizes action trajectories to examine misalignment patterns.
