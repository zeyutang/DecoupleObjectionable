# Procedural Fairness Through Decoupling Objectionable Data Generating Components

### Abstract

We reveal and address the frequently overlooked yet important issue of _disguised procedural unfairness_, namely, the potentially inadvertent alterations on the behavior of neutral (i.e., not problematic) aspects of data generating process, and/or the lack of procedural assurance of the greatest benefit of the least advantaged individuals. Inspired by John Rawls's advocacy for _pure procedural justice_ \citep{rawls1971theory,rawls2001justice}, we view automated decision-making as a microcosm of social institutions, and consider how the data generating process itself can satisfy the requirements of procedural fairness. We propose a framework that decouples the objectionable data generating components from the neutral ones by utilizing reference points and the associated value instantiation rule. Our findings highlight the necessity of preventing _disguised procedural unfairness_, drawing attention not only to the objectionable data generating components that we aim to mitigate, but also more importantly, to the neutral components that we intend to keep unaffected.

---

### Running Environment Requirements

- `python >= 3.9.6`
- `numpy >= 1.22.0`
- `scikit-learn >= 1.0.0`
- `torch >= 1.5.0`

### To Run the Code

Open `uci_adult.ipynb` in [Jupyter Notebook](https://jupyter.org/), and click `Run All` to execute the notebook containing the implementation of our framework.
