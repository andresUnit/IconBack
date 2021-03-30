import config
import sys
from os import listdir

mypath = "../" + sys.argv[1] + "/"
folders = [f for f in listdir(mypath) if "." not in f and "__" not in f]

template = config.template.split("\n")

file = open("source/index.rst", "w")

for k in template:
    file.write(k + "\n")
    if ":caption: Contents:" in k.strip():
        file.write("\n")
        for folder in folders:
            file.write("   " + sys.argv[1] + "." + folder + "\n")
        file.write("   " + "modules"+"\n")

file.close()
