#! /bin/bash

find ./ -type f  \( -name "conf.py" -o -name "configure.in" -o -name "setup.py" \) -exec sed -i 's/'$1'/'$2'/' {} \;
