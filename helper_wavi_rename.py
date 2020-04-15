import os
import shutil

for fname in os.listdir():
    if "BANNANTINE" in fname:
        shutil.move(fname, fname.replace("BANNANTINE", "116"))
    elif "BARKER" in fname:
        shutil.move(fname, fname.replace("BARKER", "115"))
    elif "CARTER" in fname:
        shutil.move(fname, fname.replace("CARTER", "204"))
    elif "CHAMBERS" in fname:
        shutil.move(fname, fname.replace("CHAMBERS", "115"))
    elif "COTTRELL" in fname:
        shutil.move(fname, fname.replace("COTTRELL", "114"))
    elif "FERRACUTI" in fname:
        shutil.move(fname, fname.replace("FERRACUTI", "100"))
    elif "FOSSE" in fname:
        shutil.move(fname, fname.replace("FOSSE", "203"))
    elif "GERWICK" in fname:
        shutil.move(fname, fname.replace("GERWICK", "200"))
    elif "GERVASI" in fname:
        shutil.move(fname, fname.replace("GERVASI", "113"))
    elif "GIES" in fname:
        shutil.move(fname, fname.replace("GIES", "112"))
    elif "MITCHELL" in fname:
        shutil.move(fname, fname.replace("MITCHELL", "111"))
    elif "NEUBAUER" in fname:
        shutil.move(fname, fname.replace("NEUBAUER", "110"))
    elif "OAK" in fname:
        shutil.move(fname, fname.replace("OAK", "202"))
    elif "OLIVAS" in fname:
        shutil.move(fname, fname.replace("OLIVAS", "109"))
    elif "ROTH" in fname:
        shutil.move(fname, fname.replace("ROTH", "108"))
    elif "ROWE" in fname:
        shutil.move(fname, fname.replace("ROWE", "107"))
    elif "SIMONE" in fname:
        shutil.move(fname, fname.replace("SIMONE", "106"))
    elif "SORBO" in fname:
        shutil.move(fname, fname.replace("SORBO", "202"))
    elif "STEVENS" in fname:
        shutil.move(fname, fname.replace("STEVENS", "104"))
    elif "WAGNER" in fname:
        shutil.move(fname, fname.replace("WAGNER", "201"))
    elif "PALOSAARI" in fname:
        shutil.move(fname, fname.replace("PALOSAARI", "103"))
    elif "KRAMER" in fname:
        shutil.move(fname, fname.replace("KRAMER", "101"))
    elif "WILLIAMS" in fname:
        shutil.move(fname, fname.replace("WILLIAMS", "102"))
