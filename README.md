# Concrete-REBAR with Hamiltonian Monte Carlo

## Setup
### Installing dependencies
First create a virtual environment in the current directory.
```bash
python -m venv env
```
Activate the virtual environment.
```bash
source env/bin/activate
```

Install the dependencies using the following command.
```bash
pip install -r requirements.txt
```

### Adding dependencies
If you add any new dependencies, make sure to update the `requirements.txt` file.
```bash
pip freeze > requirements.txt
```