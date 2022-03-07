export CUDA_VISIBLE_DEVICES=0

# PGN both for MC
python main.py --do_train --gpu_id=0 --epochs=30 --context_mode=both --save_path=output/both/ --no_final --data_name=MC --kl_loss_weight=0.5 --role_weight=0.2
python main.py --do_eval --context_mode=both --gpu_id=0 --coverage=True --no_final --best_model_pth=output/both/xxx --data_name=MC
