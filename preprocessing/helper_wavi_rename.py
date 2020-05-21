import os
import shutil

# bad old anonymizing script, but works as example, obviously there are easier ways to script this, depending
# on how your raw data is structured
# but I find that usually these are unique to each study and should be pretty easy to clean

for fname in os.listdir():
    if "BANNANTINE" in fname:
        shutil.move(fname, fname.replace("BDOE", "116"))
    elif "BARKER" in fname:
        shutil.move(fname, fname.replace("BDOE", "115"))
    elif "CARTER" in fname:
        shutil.move(fname, fname.replace("CDOE", "204"))
    elif "CHAMBERS" in fname:
        shutil.move(fname, fname.replace("CDOE", "115"))
    elif "COTTRELL" in fname:
        shutil.move(fname, fname.replace("CJOHNSON", "114"))
    elif "FERRACUTI" in fname:
        shutil.move(fname, fname.replace("FJAMESON", "100"))
    elif "FOSSE" in fname:
        shutil.move(fname, fname.replace("FJIM", "203"))
    elif "GERWICK" in fname:
        shutil.move(fname, fname.replace("GVON", "200"))
    elif "GERVASI" in fname:
        shutil.move(fname, fname.replace("GJANE", "113"))
    elif "GIES" in fname:
        shutil.move(fname, fname.replace("GALE", "112"))
    elif "MITCHELL" in fname:
        shutil.move(fname, fname.replace("MOM", "111"))
    elif "NEUBAUER" in fname:
        shutil.move(fname, fname.replace("NIFTY", "110"))
    elif "OAK" in fname:
        shutil.move(fname, fname.replace("OAK", "202"))
    elif "OLIVAS" in fname:
        shutil.move(fname, fname.replace("OLIVE", "109"))
    elif "ROTH" in fname:
        shutil.move(fname, fname.replace("ROSS", "108"))
    elif "ROWE" in fname:
        shutil.move(fname, fname.replace("ROZ", "107"))
    elif "SIMONE" in fname:
        shutil.move(fname, fname.replace("SIMON", "106"))
    elif "SORBO" in fname:
        shutil.move(fname, fname.replace("SORBET", "202"))
    elif "STEVENS" in fname:
        shutil.move(fname, fname.replace("STEVE", "104"))
    elif "WAGNER" in fname:
        shutil.move(fname, fname.replace("WALLE", "201"))
    elif "PALOSAARI" in fname:
        shutil.move(fname, fname.replace("PAOLO", "103"))
    elif "KRAMER" in fname:
        shutil.move(fname, fname.replace("KRAMER", "101"))
    elif "WILLIAMS" in fname:
        shutil.move(fname, fname.replace("WILL", "102"))
