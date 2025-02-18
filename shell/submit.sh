# To create the environment. Here we are assuming that it has already been done
#env_name="devtools_scicomp"
#conda create --name $env_name python=3.9
#conda $env_name
#python -m pip install pytest

# To clone the repo (which we assume has already been done)
#git clone git@github.com:lorenzotomada/devtools_scicomp_project_2025.git
#cd devtools_scicomp_project_2025/

# Touch README.md and push
touch README.md
git add README.md
git commit -m "first commit"
git push origin HEAD:main

# To create all the folders that are needed
mkdir -p src/pyclassify shell scripts experiments test
cd src/pyclassify/
touch __init__.py utils.py
cd ../..
touch scripts/run.py
cd shell/
touch submit.sbatch
touch submit.sh
cd ..
touch experiments/config.yaml
touch test/test_.py

# To finish structuring the package
python -m pip freeze > requirements.txt
wget https://raw.githubusercontent.com/lorenzotomada/template_files/main/pyproject.toml

# Chose names to substitute in the pyproject.toml
project_name="pyclassify"
project_description="Source code for the project of the course in Development Tools for Scientific Computing"
my_name="Lorenzo Tomada"
my_gmail="lorenzotomada2000"

sed -i "s/PROJECT_NAME/${project_name}/g" pyproject.toml
sed -i "s/PROJECT_DESCRIPTION/${project_description}/g" pyproject.toml
sed -i "s/MY_NAME/${my_name}/g" pyproject.toml
sed -i "s/MY_GMAIL/${my_gmail}/g" pyproject.toml

echo "ignore .dat and .data" >> .gitignore
echo "*.dat" >> .gitignore
echo "*.data" >> .gitignore

# Push the changes
git add .
git commit -m "structuring the package"
git push origin HEAD:main
