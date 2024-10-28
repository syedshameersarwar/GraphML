# Exercise 1

## Running the homework

To replicate the results of this exercise, please run all cells in the `exercise1.ipynb` notebook. If you are running it locally, ensure you have installed the following libraries using your preferred dependency manager:

- pytorch
- torch_geometric
- ogb
- scikit-learn

## Questions

### Q1. Describe the datasets in your own words. Also talk about its features and statistical properties of the graphs and labels.

### Dataset description

The `ogbg-molhiv` dataset is part of the Open Graph Benchmark (OGB), Stanford's collection of datasets for machine learning on graphs. This dataset is designed for molecular property prediction, specifically to determine whether a molecule inhibits HIV virus replication, thereby indicating its activity against HIV. More details about the dataset can be found on [OGB's website](https://ogb.stanford.edu/docs/graphprop/#ogbg-mol).

#### Graph Structure and Features

The dataset contains information about molecules and their bonds, where each node respresents an atom and each edge a bond. Both atoms and bonds have specific features, which are described in detail in the [OGB GitHub repository](https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py). The list of features for the nodes and edges are:

- **Node features**: 9-dimensional, including:
  - Atomic number
  - Chirality type
  - Degree - number of bonds
  - Formal charge
  - Number of hydrogen atoms attached
  - Number of radical electrons
  - Hybridization type
  - Aromaticity
  - Atomic mass
- **Edge features**: 3-dimensional, including:

  - Bond Type
  - Bond Stereochemistry
  - Conjugation

- **Labels**: 1 or 0 depending on whether a molecule inhibits HIV virus replication or not

### Statistical properties

There are 41,127 molecules in the dataset and the average cluster coefficient is 0.002 (from [huggingface](https://huggingface.co/datasets/OGB/ogbg-molhiv)), which indicates very sparse graphs.

After installing the ogb library, the following code can be used to get some statistics about the data.

```python3
molHIV = PygGraphPropPredDataset(name = "ogbg-molhiv")
molHIV.print_summary()
```

|            | #nodes | #edges |
| ---------- | ------ | ------ |
| mean       | 25.5   | 54.9   |
| std        | 12.1   | 26.4   |
| min        | 2      | 2      |
| quantile25 | 18     | 40     |
| median     | 23     | 50     |
| quantile75 | 29     | 64     |
| max        | 222    | 502    |

By examining the median and mean of the table, we observe that most of the graphs (or molecules) are similar in their number of nodes and edges. However, there are outlier molecules with significantly higher numbers of nodes.

## Notes about the implementation

- We used one-hot encoding for atomic number feature.
- Updated implmentation to load data in batches by utilizing pytorch Dataset efficiently.
- We added "self-loops," meaning that each node uses its own features along with its neighbours when calculating the hidden state. In other words, we treat a node and its neighbors identically.
- Early stopping was implemented to prevent overfitting, with a patience of 10 epochs. This means that if the validation ROC does not exceed its previous maximum after 10 epochs, training stops, and we revert to the best model saved up to that point.
- We set the seed for the random libraries used to get replicable results.
- We initially applied Xavier initialization, but later switched to the default initialization, which provided a slight improvement in the test ROC.
- We used ChatGPT and Claude to suggest optimizations and improvements for code, and clarify language for documentation.
