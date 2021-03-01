x = [
    "numpy",
    "pandas",
    "matplotlib",
    "os",
    "tarfile",
    "six.moves",
    "joblib",
    "sklearn",
    "pyplot",
    "tensorflow",
]
for lib in x:
    try:
        __import__(lib)
        print(lib, " is installed")
    except ModuleNotFoundError:
        print(lib, " is not installed")
