#! /bin/bash
###### Part 1 ######
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --account=higgsgpu
#SBATCH --job-name=my_pfn 
#SBATCH --ntasks=4
#SBATCH --output=/afs/ihep.ac.cn/users/w/wuzuofei/PFN/my_script_DDP/my_log/pfn_DP1.log
#SBATCH --mem-per-cpu=20000
#SBATCH --gres=gpu:v100:2

###### Part 2 ######

 ulimit -d unlimited
 ulimit -f unlimited
 ulimit -l unlimited
 ulimit -n unlimited
 ulimit -s unlimited
 ulimit -t unlimited
 srun -l hostname

 /usr/bin/nvidia-smi -L

 echo "Allocate GPU cards : ${CUDA_VISIBLE_DEVICES}"
 nvidia-smi
 # >>> conda initialize >>>
 # !! Contents within this block are managed by 'conda init' !!
 __conda_setup="$('/hpcfs/cepc/higgsgpu/wuzuofei/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
 if [ $? -eq 0 ]; then
     eval "$__conda_setup"
 else    
     if [ -f "/hpcfs/cepc/higgsgpu/wuzuofei/miniconda3/etc/profile.d/conda.sh" ]; then
         . "/hpcfs/cepc/higgsgpu/wuzuofei/miniconda3/etc/profile.d/conda.sh"
     else   
         export PATH="/hpcfs/cepc/higgsgpu/wuzuofei/miniconda3/bin:$PATH"
     fi      
 fi
 unset __conda_setup
 # <<< conda initialize <<<
 
 conda activate weaver

 which python 

python -u my_train_DDP.py DDP_test
