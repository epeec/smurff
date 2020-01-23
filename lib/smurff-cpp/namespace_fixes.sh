#!/bin/sh

FILES=$(find . -name '*.cpp' -type f)

for F in $FILES
do
	if grep -q "using namespace smurff" $F 
	then
		gsed -i -e "
		s/using namespace smurff;/namespace smurff {/g;
		s/smurff:://g;
		\$a} // end namespace smurff
		" $F
	fi
done

