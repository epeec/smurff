#!/bin/sh


for file in $*
do
    echo $file ...
    vim -e - $file <<@@@
g/macau_prior_config_item/d
g/\[side_info_/d
wq
@@@

done
