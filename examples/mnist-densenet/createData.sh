#! /bin/bash   
set -m

cd ..
mkdir valid
mkdir test

cd valid 
mkdir corona/ external/ internal/ noise

cd ../test
mkdir corona/ external/ internal/ noise

cd ../train

#In train folder create and shuffle data
a=$(find 'corona'/ -type f | shuf -n 120)
mv $a ../valid/corona/ 
b=$(find 'corona'/ -type f | shuf -n 5)
mv $b ../test/corona/ 

c=$(find 'external'/ -type f | shuf -n 60)
mv $c ../valid/external/ 
d=$(find 'external'/ -type f | shuf -n 5)
mv $d ../test/external/ 

e=$(find 'internal'/ -type f | shuf -n 60)
mv $e ../valid/internal/ 
f=$(find 'internal'/ -type f | shuf -n 5)
mv $f ../test/internal/ 

g=$(find 'noise'/ -type f | shuf -n 120)
mv $g ../valid/noise/ 
h=$(find 'noise'/ -type f | shuf -n 5)
mv $h ../test/noise/ 

