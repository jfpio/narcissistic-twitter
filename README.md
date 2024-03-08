# Narcisissitic Twitter

## Project goals

Create a model that can predict the narcissism of the person based on their tweets.

### Project structure
```
├── configs
├── lib
├── notebooks
├── scripts
├── tests
```

### Lib
**The main library of the project.** Contains the code for models, metrics, datasets, etc. 
The library is divided into modules, each of which is responsible for a different part of the project, including:
- `text_cleaners` - contains functions for cleaning text data, including lemmatization and removing URLs, emojis, etc. It is used by scripts for creating datasets.
- `source_processing` - contains functions for matching tweets with articles based on various criteria and functions for loading articles and tweets from files as dataframes. It is used by scripts for creating datasets.
- `google_api` - contains functions for using the Google API to save and download files from Google Drive.

### Configs
Contains configuration files for experiments. We use [Hydra](https://hydra.cc/docs/intro/) to manage them.

### Tests
Contains unit tests for the project, including tests for models, metrics, datasets, etc. Each model has its own test file, which contains a test class with tests for the model, which can be inherited from other test classes.

### Scripts
Contains the main scripts for training and testing models (in general python programs that you run).

### Notebooks
Contains various notebooks.

## Project setup

### Python environment
Using Poetry to set up the environment is the preferred option. 
To start using Poetry, you can [install](https://python-poetry.org/docs/#installing-with-the-official-installer) it.

If the Python version is mismatched one can use 
```sh
poetry env use [full_path_to_the_python_interpreter]
```
This ensures that your project uses the desired Python version.

Once you have your environment set up, you can install the project's dependencies from the lock file by running:
```sh
poetry install
```
This command reads the `poetry.lock` file and installs the exact versions of the dependencies specified in it
To add a new dependency one needs to type
```sh
poetry add [package_name]
```
This command will install the specified package and update both the `pyproject.toml` file and the `poetry.lock` file with the new dependency information.
To update or make a new lock file we can type in:
```sh
poetry lock [--no-update] [--check]
```
By default, running poetry lock without any options will update the lock file based on the latest `pyproject.toml` information. However, you can use the `--no-update` option to only refresh the lock file without modifying it based on the project's configuration. Additionally, the `--check` option allows you to verify whether the lock file is consistent with the `pyproject.toml` file.

Poetry allows us to use groups with specific dependencies. As Poetry is super concise with its documentation, to modify, add or make optional one should refer [here](https://python-poetry.org/docs/master/managing-dependencies/).

### Environment file .env
Create your environment file from template `.env_template`. Change its name to `.env` file and paste your credentials.

#### Twitter credentials
Generate your bearer token [here](https://developer.twitter.com/en/portal/dashboard) and add it to your `.env` file.

#### Load credentials
You can use:
In the application, you may also use [python-dotenv](https://pypi.org/project/python-dotenv/):
```python
from dotenv import load_dotenv

load_dotenv()
```
in notebooks:
```jupyter
%load_ext dotenv
%dotenv
```

### Git hooks
We use [pre-commit](https://pre-commit.com/) to manage git hooks. Unfortunately, the custom hook doesn't work with Windows.
Windows users can comment `strip-notebooks` hook on `.pre-commit-config.yaml`.
To configure this extension one has to run
```sh
chmod +x .hooks
pre-commit install
```

## Running tests
Run tests for a given directory or file:
```
python3 -m pytest tests -rA
```
Add `-k 'test_name or expression'` to run just selected test(s):
```
python3 -m pytest tests/model -k 'test_metrics' -rA
```

### Linting
We use `flake8` and `black`. They are running automatically as git hooks.
Configuration for flake8 is stored as a `.flake8` file and black configuration is stored in `pyproject.toml`.

#### VS Code
To enable linting you have to have configured interpreter 
```Python: Select Interpreter``` (Ctrl+Shift+P) and choose your conda environment. Next, you have to choose ```Python: Select Linter``` and pick flake8.  

## Additional Tools
### Hydra
Hydra is a framework for elegantly configuring complex applications. It is very useful for managing experiments config and for multijobs.
It could be used for example to run multiple experiments with different parameters like learning rate.

It is recommended to read [the article written by Hydra author](https://medium.com/pytorch/hydra-a-fresh-look-at-configuration-for-machine-learning-projects-50583186b710).