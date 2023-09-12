# Multinode training - JAX

In this example we describe an example of multinode training in JAX on a toy network with MSE loss.

## Training
We used slurm to perform training with the follwing script from Moritz Wolter [here](https://www.wolter.tech/?p=577).
The sample script on JUWELS Booster is given below

```
#!/bin/bash
#
#SBATCH -A TODO:enter-your-project-here
#SBATCH --nodes=3
#SBATCH --job-name=test_multi_node
#SBATCH --output=test_multi_node-%j.out
#SBATCH --error=test_multi_node-%j.err
#SBATCH --time=00:20:00
#SBATCH --gres gpu:4
#SBATCH --partition develbooster

echo "Got nodes:"
echo $SLURM_JOB_NODELIST
echo "Jobs per node:"
echo $SLURM_JOB_NUM_NODES 

module load Python
ml CUDA/.12.0

export PYTHONPATH=.

source TODO: Path to your virtual environment

srun --nodes=3 --gres gpu:4 python src/train.py

```

## Acknowledgements
We would like to thank [Moritz Wolter](https://www.wolter.tech) for helping me make this work.