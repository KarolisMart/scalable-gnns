# Scalable Graph Networks for Particle Simulations

PyTorch implementation for the paper **[Scalable Graph Networks for Particle Simulations](https://arxiv.org/abs/2010.06948)**

You can create a conda environment with all of the required packages:  
`conda env create -f environment.yml`  
`conda activate scalable-graph-networks`

In the provided Jupyter notebook `notebooks/example.ipynb` you can see how to build a dataset or train and evaluate a model.  

We provide the [datasets](https://polybox.ethz.ch/index.php/s/BU1YYXi40DgWTng) we used with up to 1000 particles as well as the best [runs](https://polybox.ethz.ch/index.php/s/NpQ7TTCkCy4R74G) of the models trained on them. 
[Download](https://polybox.ethz.ch/index.php/s/goapswgs2LCdWki) the respective folders and put them in the project's main directory. In the Jupyter notebook, you can see how to evaluate those models on the provided datasets and visualise the results. 

Note that the hierarchical models only work with `graph_type='*_level_hierarchical'`, while DeltaGN and HOGN models only work with `graph_type='*_nn'` or `graph_type='fully_connected'`.

The project is structured as follows:  
* `data.py` contains the simulator and the PyTorch dataset class. At the bottom of this file, you can find the commands used to generate all of the datasets used in our experiments. Running this file from the command line will generate gravitational 20 particle dataset, as used in our experiments. To generate other datasets uncomment the corresponding lines.
* `model.py` contains the models (DeltaGN, HOGN, HierarchicalDeltaGN, HierarchicalHOGN).
* `train.py` is the training script. It can be run from the command line. Use the `-h` flag to see all of the available options for model training.
* `eval.py` is the evaluation script. It can also be run from the command line. Use the `-h` flag to see all of the available options.
* `visualize.py` contains the plotting functions.
* `util.py` contains various helper functions such as total energy calculation or folder removal, as well as the torch implementations of the graph construction that are used during model evaluation.

Please cite our paper if you use this code or our method in your work:
```
@article{martinkus2020scalable,
  title={Scalable Graph Networks for Particle Simulations},
  author={Martinkus, Karolis and Lucchi, Aurelien and Perraudin, Nathana{\"e}l},
  journal={arXiv preprint arXiv:2010.06948},
  year={2020}
}
```

If you have any questions don't hesitate to reach out: [martinkus@ethz.ch](mailto:martinkus@ethz.ch)
