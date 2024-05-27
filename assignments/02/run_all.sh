module load openmpi/3.1.6-gcc-12.2.0-d2gmn55
#make clean
#make all

rm -r output/
mkdir output

for tasks in 6 ; do #2 4 16 64 96 ; do
    sbatch --output=output/mandelbrot-mpi-$tasks.out --ntasks=$tasks mandelbrot_mpi.sh
    sbatch --output=output/mandelbrot-seq-$tasks.out --ntasks=1 mandelbrot_seq.sh
done