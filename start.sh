#-----------------------------resnet50 cifar100---------------------------------------
# python3 -u ./centralized/resnet50_cifar100.py --config_path ./config/resnet50_cifar100/dir_5_seed_16_client_30/centralized.yaml
# ---------------------- dir 0.5 ---------------------
# ------------------ seed 16 ---------------------------
python3 -u ./data/generate_data.py --config_path ./config/resnet50_cifar100/dir_5_seed_16_client_30/data.yaml
# python3 -u ./server/fedavg.py --config_path ./config/resnet50_cifar100/dir_5_seed_16_client_30/fedavg.yaml
python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_5_seed_16_client_30/loss_fixed_batch_global_loss5.yaml
python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_5_seed_16_client_30/loss_fixed_batch_global_loss4.yaml
python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_5_seed_16_client_30/loss_fixed_batch_global_loss3.yaml
python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_5_seed_16_client_30/loss_fixed_batch_global_loss2.yaml
python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_5_seed_16_client_30/loss_fixed_batch_global_loss1.yaml