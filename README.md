# Jaguar Re-Identification
ML project to Re-Identify Jaguars as part of HPI Project Seminar.

## Setup

1. Create the environment:
```bash
   conda env create -f "environment.yml"
```
2. Activate it:
```bash
   conda activate jid
```

Use the provided Makefile to run the tests, linter, formatter or MyPy type checker.  
```bash
make all
```

### Update Environment
In the unlikely case that someone changed the environment.yml file, you can update the environment with the following command: (environment should be active)
```bash
conda env update --file environment.yml --prune
```

## Authors
- Mehdi Gouasmi (https://github.com/D-i-n-o)
- Philipp Kolbe (https://github.com/philippkolbe)
- Supervisor: Antonio Rueda-Toicen (https://github.com/andandandand)
