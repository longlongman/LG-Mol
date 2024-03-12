data_path= # replace to your data path
save_dir=  # replace to your save path
n_gpu=1
MASTER_PORT=10086
dict_name="dict.txt"
weight_path=  # replace to your ckpt path
task_name="hiv"  # molecular property prediction task name 
task_num=1
loss_func="shape4classify_finetune_multi_task_BCE"
lr=5e-5
batch_size=256
epoch=5
dropout=0.2
warmup=0.1
local_batch_size=16
only_polar=-1
conf_size=11
seed=0
unimol_dir= # replace to your unimol path

if [ "$task_name" == "qm7dft" ] || [ "$task_name" == "qm8dft" ] || [ "$task_name" == "qm9dft" ]; then
	metric="valid_agg_mae"
elif [ "$task_name" == "esol" ] || [ "$task_name" == "freesolv" ] || [ "$task_name" == "lipo" ]; then
    metric="valid_agg_rmse"
else 
    metric="valid_agg_auc"
fi

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
update_freq=`expr $batch_size / $local_batch_size`
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --task-name $task_name --user-dir $unimol_dir --train-subset train --valid-subset valid,test \
       --conf-size $conf_size \
       --num-workers 8 --ddp-backend=c10d \
       --dict-name $dict_name \
       --task shape4classify --loss $loss_func --arch shape4classify  \
       --classification-head-name $task_name --num-classes $task_num \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $local_batch_size --pooler-dropout $dropout\
       --update-freq $update_freq --seed $seed \
       --log-interval 100 --log-format simple \
       --validate-interval 1 \
       --finetune-from-model $weight_path \
       --best-checkpoint-metric $metric --patience 20 \
       --save-dir $save_dir --only-polar $only_polar\
       --maximize-best-checkpoint-metric \
       --points-num 100
