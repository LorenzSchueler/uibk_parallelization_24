module load openmpi/3.1.6-gcc-12.2.0-d2gmn55

cd build
make
cd ..

rm -r output/
mkdir output

for tasks in 4 ; do #2 4 16 64 96 ; do
    sbatch --output=output/main-full-code-parallel-$tasks.out --ntasks=$tasks parallel.sh
done