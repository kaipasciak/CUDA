#!/bin/sh
# Specify a partition
#SBATCH --partition=dggpu
# Request nodes
#SBATCH --nodes=1
# Request some processor cores
#SBATCH --ntasks=4
# Request GPUs
#SBATCH --gres=gpu:1
# Request memory 
#SBATCH --mem=1G
# Run for some minutes
#SBATCH --time=00:04:00
# Name job
#SBATCH --job-name=my-program
# Name output file
#SBATCH --output=%x_%j.out
# Set email address (for user with email "usr1234@uvm.edu")
#SBATCH --mail-user=walter.clay@uvm.edu

# Request email to be sent at begin and end, and if fails;
# options are NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-type=FAIL 
#SBATCH --mail-type=BEGIN 
#SBATCH --mail-type=END 

# change to the directory where you submitted this script
cd ${SLURM_SUBMIT_DIR}

# print pwd
echo "current directory is $(pwd)"
# Executable section: echoing some Slurm data
echo "Starting sbatch script sample-job.slurm at:`date`"
echo "Running host:    ${SLURMD_NODENAME}"
echo "Assigned nodes:  ${SLURM_JOB_NODELIST}"
echo "Job ID:          ${SLURM_JOBID}"
echo "GPU(s):          ${CUDA_VISIBLE_DEVICES}"

# change this if necessary to match the name and location
# of your compiled CUDA program
./a.out

# here is documentation:
# https://www.uvm.edu/vacc/kb/knowledge-base/write-submit-job-deepgreen/

