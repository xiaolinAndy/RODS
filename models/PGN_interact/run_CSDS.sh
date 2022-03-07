export CUDA_VISIBLE_DEVICES=0

# PGN both for CSDS
python main.py --do_train --gpu_id=0 --epochs=40 --context_mode=both --save_path=output/both/ --no_final --data_name=CSDS --kl_loss_weight=0 --role_weight=0.5
python main.py --do_ft --context_mode=both --gpu_id=0 --coverage=True --epochs=10 --save_path=output/both/ --val_freq=500 --test_first --no_final --data_name=CSDS  --kl_loss_weight=1 --role_weight=0.5
python main.py --do_eval --context_mode=both --gpu_id=0 --coverage=True --no_final --best_model_pth=output/both/xxx --data_name=CSDS


