## Setting up Python Environment
This program requires Python >= 3.9 to run. I used Anaconda Python on Windows.

First, create and activate a python virtual environment. From an Anaconda terminal, or a terminal with Python on the PATH.
```
python -m venv cs4372
.\cs4372\scripts\activate
#./cs4372/bin/active on Linux
```

Next, install all the program dependencies.

`pip install -r requirements.txt`

Also, Graphviz needs to be installed.
For Windows, download an executable install package from https://graphviz.org/download/. When installing, make sure to add Graphviz to the System PATH (all users).

For Linux (Debian/Ubuntu), use `sudo apt install graphviz`
For MacOS, use `brew install graphviz`

## Running the program
The program is a Juypter notebook.

## Cleaning up
To clean up, deactivate the virutal environment and delete its folder.

```
deactivate
rd /s /q cs4372  # If on Windows
# rm -rf cs4372  # If using *nix
```
