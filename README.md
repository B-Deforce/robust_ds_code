# Robust and Reproducible Data Science Code

Welcome to the **Robust and Reproducible Data Science Code** repository! This project is designed as part of a class on writing clean, maintainable, and reproducible data science code. The repository demonstrates best practices for structuring data science projects, implementing reusable components, and ensuring reproducibility.

## Overview

The goal of this project is to teach foundational concepts in writing robust data science code. The repository includes examples and tools to help you learn and apply these concepts in your own projects.

Key topics covered include:
- Writing clean and maintainable code using object-oriented programming (OOP).
- Designing reusable components with abstract base classes (ABC).
- Ensuring type safety with tools like `beartype`.
- Using configuration-driven workflows with Hydra.
- Setting up reproducible environments with `hatch`.

## Repository Structure

```bash
.
├── src/
│   ├── postgrad_class/
│   │   ├── conf/               # Configuration files for Hydra
│   │   ├── model/              # Model implementations (e.g., LinearModel, SimpleNNModel)
│   │   ├── notebooks/          # Jupyter notebooks for examples and exercises
│   │   ├── __about__.py        # Project metadata
│   │   ├── __init__.py         # Package initialization
│   │   └── main.py             # Entry point for running models
├── tests/                      # Unit tests for the project
├── pyproject.toml              # Project configuration for Hatch
├── LICENSE                     # License information
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore rules
```


## Getting Started

### Prerequisites

To run this project, you need:
- Python 3.8 or later
- `hatch` for managing the project environment and dependencies ([installation isntructions](https://hatch.pypa.io/1.13/install/#macos) for Hatch).
   - If you're used to `conda`, an easy way to install `hatch` is to create a separate `conda` env with your desired `python` version and install `hatch` there. You can then run the `hatch` commands below inside this environment.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BojeDeforce/postgrad-class.git
   cd postgrad-class

2. Set up the environment using `hatch`:
    ```bash
    hatch env create
    hatch shell
    ```
## Examples
The `src/postgrad_class/notebooks/example.py` notebook (in `py:percent` format) introduces key concepts and provides hands-on examples. Open it in Jupyter Notebook.
### Running Jupyter Notebooks on a Notebook Server with Hatch

Jupyter notebooks run on a *notebook server*, which is a Python process that provides a web interface to interactively write, run, and visualize code. This server handles kernel management, file I/O, and communication between the front-end (UI) and the back-end (kernel) over HTTP. The actual computation happens in a kernel — typically a Python interpreter — that executes code sent from the notebook interface.

#### Launching a Jupyter Notebook Server with Hatch

You can launch a notebook server within your active Hatch environment (see step 2 above) by running:
```bash
jupyter notebook --no-browser
```

This will start the Jupyter server and print a URL with a token, for example:

```bash
http://127.0.0.1:8888/?token=your-token-here
```
You have now two main options:

1. In your browser
Simply paste the provided URL into your browser to access the classic Jupyter notebook interface.

2. In VSCode or another IDE

- Open the Command Palette and select:  
  **"Jupyter: Specify local or remote Jupyter server"**
- Paste the URL with the token.
- Now, when you open a `.ipynb` file, VSCode will connect to the running kernel on that server.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

Created by Boje Deforce. For questions or feedback, feel free to reach out via [GitHub](https://github.com/BojeDeforce).
