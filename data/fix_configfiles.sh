#!/bin/sh

files=`find . -name '*ini'`

for file in $files
do
    echo $file ...
    vim -e - $file <<@@@
g/macau_prior_config_item/d
g/side_info_\d_\d/d
%s/macau_prior_config_/side_info_/g
wq
@@@

done
