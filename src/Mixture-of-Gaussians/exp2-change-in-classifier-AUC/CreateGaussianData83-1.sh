#!/bin/bash

#----------USAGE--------------------------
# python CreateGaussianData-G83-woMultiprocessing.py <num_iteration>
#-----------------------------------------

#python CreateGaussianData-G83-woMultiprocessing.py 4

for i in {1..200}
do
	echo Loop $i
        python CreateGaussianData-G83-woMultiprocessing.py $i
done

