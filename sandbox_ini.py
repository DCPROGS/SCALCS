from dcpyps import dcio
from pprint import pprint

filename = "./dcpyps/samples/etc/HJCFIT1.INI"

ini = dcio.ini_HJCFIT_read(filename, verbose=True)

pprint(ini)
