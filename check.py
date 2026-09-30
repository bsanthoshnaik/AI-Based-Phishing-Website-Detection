import importlib.metadata

# 🔧 Add your package names here
packages = ["numpy", "pandas", "requests", "matplotlib","flask","catboost","tensorflow","re","tldextract"]

for pkg in packages:
    try:
        version = importlib.metadata.version(pkg)
        print(f"{pkg}=={version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg} is not installed.")
