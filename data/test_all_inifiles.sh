files=`find . -name '*ini'`
for ini in $files
do
   cd `dirname $ini`
   f=`basename $ini`
   pwd
   echo smurff --ini $f --burnin 1 --nsamples 1 --verbose 0
   smurff --ini $f --burnin 0 --nsamples 1 --verbose 0
   cd -
done
