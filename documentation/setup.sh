find ../$1/. -type d -exec touch {}/__init__.py \;
#Se ejecuta en la carpeta documentation
sphinx-apidoc -o source ../$1/
python setup/setup.py $1