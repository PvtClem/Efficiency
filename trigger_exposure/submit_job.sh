#!/bin/sh

thisdir=/current/your/directory
exe=${thisdir}/exe
input=${thisdir}/COREAS-AN_sim_data_directory.txt

mkdir -p ${exe}

for n in `cat ${input}`
do
    num=`echo $n | cut -c 103-106`
    cat ${thisdir}/job.sh | sed 's/filenumber/'$num'/g' > ${exe}/exe-$num.sh
    sbatch ${exe}/exe-$num.sh
done
