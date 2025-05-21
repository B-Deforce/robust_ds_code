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
- `hatch` for managing the project environment and dependencies

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

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

Created by Boje Deforce. For questions or feedback, feel free to reach out via [GitHub](https://github.com/BojeDeforce).
