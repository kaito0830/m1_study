#PBS -q large-md
#PBS -l nodes=2
#PBS -l walltime=23:50:00
#PBS -N disk

cd ${PBS_O_WORKDIR}

aprun -n 64 ./athena -i athinput.disk_sph > output-00.txt
