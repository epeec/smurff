SRC_FILES=$(find SmurffCpp -name '*.cpp' -o -name '*.hpp' -o -name '*.h')

UTILS_FILES="Error.h
 InvNormCdf.cpp
 InvNormCdf.h
 StringUtils.cpp
 StringUtils.h
 ThreadVector.hpp
 TruncNorm.cpp
 TruncNorm.h
 counters.cpp
 counters.h
 linop.h
 omp_util.cpp
 omp_util.h"

for F in $UTILS_FILES
do
       	gsed -i -e "s@SmurffCpp/Utils/$F@Utils/$F@g;" $SRC_FILES
done

