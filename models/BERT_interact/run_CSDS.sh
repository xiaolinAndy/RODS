export CUDA_VISIBLE_DEVICES=2

#bert both for CSDS
#python preprocess.py -log_file logs/log -data_name CSDS

# python train.py -task abs -mode train -bert_data_path data/CSDS/bert/ -dec_dropout 0.2  -model_path output/both -sep_optim true -lr_bert 0.002 -lr_dec 0.02 -save_checkpoint_steps 400 -batch_size 1 -train_steps 4000 -report_every 50 -accum_count 15 -use_bert_emb true -use_interval true -warmup_steps_bert 1000 -warmup_steps_dec 1000 -max_pos 512 -visible_gpus 1  -log_file logs/bert_both_train.log -finetune_bert True -merge=gate -role_weight=0.5 -kl_weight=0
# python train.py -task abs -mode validate -batch_size 10 -test_batch_size 10 -bert_data_path data/CSDS/bert/ -log_file logs/bert_both_val.log -model_path output/both -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 10 -result_path logs/bert_both_val.txt -temp_dir temp/ -test_all=True -merge=gate
# finetune for KL loss
# python train.py -task abs -mode train -bert_data_path data/CSDS/bert/ -dec_dropout 0.2 -train_from output/both/xxx.pt -model_path output/both_ft -sep_optim true -lr_bert 0.002 -lr_dec 0.02 -save_checkpoint_steps 200 -batch_size 1 -train_steps xxx -report_every 50 -accum_count 15 -use_bert_emb true -use_interval true -warmup_steps_bert 0 -warmup_steps_dec 0 -max_pos 512 -visible_gpus 1  -log_file logs/bert_both_ft_train.log -finetune_bert True -merge=gate -role_weight=0.5 -kl_weight=0.25
# python train.py -task abs -mode validate -batch_size 10 -test_batch_size 10 -bert_data_path data/CSDS/bert/ -log_file logs/bert_both_ft_val.log -model_path output/both_ft -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 15 -result_path logs/bert_both_ft_val.txt -temp_dir temp/ -test_all=True -merge=gate
#run on testset
python train.py -task abs -mode test -batch_size 1 -test_batch_size 1 -bert_data_path data/CSDS/bert/ -log_file logs/bert_both_ft_test.log -test_from output/both/model_step_2600.pt -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 15 -result_path logs/bert_both_ft_test_ -temp_dir temp/ -merge=gate