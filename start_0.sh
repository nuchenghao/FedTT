#===============================================================================================================================================================
#-----------------------------resnet34 cinic10---------------------------------------
#===============================================================================================================================================================
# ---------------------- dir 0.3 ---------------------
# ------------------ seed 16 ---------------------------

# python3 -u ./centralized/resnet34_cinic10.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/centralized.yaml

python3 -u ./data/generate_data.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/data.yaml

# python3 -u ./server/fedavg.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedavg.yaml

# python3 -u ./server/fedavg.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedavg_H.yaml

# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/loss_dynamic_batch_global_loss_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/loss_dynamic_batch_global_loss_r3_s1.yaml

# python3 -u ./server/fedbalancer.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedbalancer.yaml

# python3 -u ./server/fedsampling.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedsampling.yaml

# python3 -u ./server/ODE.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/ode.yaml

# python3 -u ./server/fedcase.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedcase.yaml





# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r1_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r2_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r5_s1.yaml

# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/loss_dynamic_batch_global_loss_r1_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/loss_dynamic_batch_global_loss_r2_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/loss_dynamic_batch_global_loss_r5_s1.yaml





# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r3_s05.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r3_s15.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r3_s2.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/loss_dynamic_batch_global_loss_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/loss_dynamic_batch_global_loss_r3_s05.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/loss_dynamic_batch_global_loss_r3_s15.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/loss_dynamic_batch_global_loss_r3_s2.yaml





# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_wo_weights_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_wo_weights_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/loss_dynamic_batch_global_loss_wo_weights_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/loss_dynamic_batch_global_loss_wo_weights_r4_s1.yaml






# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r1_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r2_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r4_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r5_s0.yaml





# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/loss_dynamic_batch_global_loss_r1_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/loss_dynamic_batch_global_loss_r2_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/loss_dynamic_batch_global_loss_r4_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/loss_dynamic_batch_global_loss_r5_s0.yaml



# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_wo_weights_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/loss_dynamic_batch_global_loss_wo_weights_r3_s0.yaml





# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/woparallel_128.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r3_s0_128.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/woparallel_160.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r3_s0_160.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/woparallel_192.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r3_s0_192.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/woparallel_224.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r3_s0_224.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/woparallel_256.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r3_s0_256.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/woparallel_512.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r3_s0_512.yaml

# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/random_select.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/random_select400.yaml




# python3 -u ./server/fedprox.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedprox.yaml
# python3 -u ./server/fedprox.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedprox_my.yaml






# python3 -u ./server/fednova.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fednova.yaml
# python3 -u ./server/fednova.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fednova_my.yaml





# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_SD_r3_s0.yaml






# python3 -u ./server/fedavg.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedavg_128.yaml
# python3 -u ./server/fedavg.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedavg_256.yaml
# python3 -u ./server/fedavg.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedavg_512.yaml


# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r3_s0_128.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r3_s0_256.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/classify_dynamic_batch_r3_s0_512.yaml




# python3 -u ./server/fedbalancer.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedbalancer_128.yaml
# python3 -u ./server/fedbalancer.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedbalancer_256.yaml
# python3 -u ./server/fedbalancer.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedbalancer_512.yaml


python3 -u ./server/fedsampling.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedsampling_128.yaml
python3 -u ./server/fedsampling.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedsampling_256.yaml
python3 -u ./server/fedsampling.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedsampling_512.yaml

# python3 -u ./server/fedbalancer.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedbalancer_128.yaml
# python3 -u ./server/fedbalancer.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedbalancer_256.yaml
# python3 -u ./server/fedbalancer.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedbalancer_512.yaml

python3 -u ./server/fedcase.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedcase_128.yaml
python3 -u ./server/fedcase.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedcase_256.yaml
python3 -u ./server/fedcase.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/fedcase_512.yaml



# python3 -u ./server/ODE.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/ode_128.yaml
# python3 -u ./server/ODE.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/ode_256.yaml
# python3 -u ./server/ODE.py --config_path ./config/resnet34_cinic10/dir_3_seed_16_client_100/ode_512.yaml

# # ------------------ seed 166 ---------------------------
# python3 -u ./centralized/resnet34_cinic10.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/centralized.yaml

# python3 -u ./data/generate_data.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/data.yaml

# python3 -u ./server/fedavg.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/fedavg.yaml

# python3 -u ./server/fedavg.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/fedavg_H.yaml


# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/classify_dynamic_batch_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/loss_dynamic_batch_global_loss_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/classify_dynamic_batch_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/loss_dynamic_batch_global_loss_r3_s1.yaml

# python3 -u ./server/fedbalancer.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/fedbalancer.yaml

# python3 -u ./server/fedsampling.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/fedsampling.yaml

# python3 -u ./server/ODE.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/ode.yaml

# python3 -u ./server/fedcase.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/fedcase.yaml



# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/classify_dynamic_batch_r1_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/classify_dynamic_batch_r2_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/classify_dynamic_batch_r5_s1.yaml

# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/loss_dynamic_batch_global_loss_r1_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/loss_dynamic_batch_global_loss_r2_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/loss_dynamic_batch_global_loss_r5_s1.yaml





# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/classify_dynamic_batch_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/classify_dynamic_batch_r3_s05.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/classify_dynamic_batch_r3_s15.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/classify_dynamic_batch_r3_s2.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/loss_dynamic_batch_global_loss_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/loss_dynamic_batch_global_loss_r3_s05.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/loss_dynamic_batch_global_loss_r3_s15.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/loss_dynamic_batch_global_loss_r3_s2.yaml





# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/classify_dynamic_batch_wo_weights_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/classify_dynamic_batch_wo_weights_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/loss_dynamic_batch_global_loss_wo_weights_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166_client_100/loss_dynamic_batch_global_loss_wo_weights_r4_s1.yaml





# # ------------------ seed 1666 ---------------------------
# python3 -u ./centralized/resnet34_cinic10.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/centralized.yaml

# python3 -u ./data/generate_data.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/data.yaml

# python3 -u ./server/fedavg.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/fedavg.yaml

# python3 -u ./server/fedavg.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/fedavg_H.yaml

# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/classify_dynamic_batch_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/loss_dynamic_batch_global_loss_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/classify_dynamic_batch_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/loss_dynamic_batch_global_loss_r3_s1.yaml

# python3 -u ./server/fedbalancer.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/fedbalancer.yaml

# python3 -u ./server/fedsampling.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/fedsampling.yaml

# python3 -u ./server/ODE.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/ode.yaml

# python3 -u ./server/fedcase.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/fedcase.yaml



# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/classify_dynamic_batch_r1_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/classify_dynamic_batch_r2_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/classify_dynamic_batch_r5_s1.yaml

# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/loss_dynamic_batch_global_loss_r1_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/loss_dynamic_batch_global_loss_r2_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/loss_dynamic_batch_global_loss_r5_s1.yaml





# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/classify_dynamic_batch_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/classify_dynamic_batch_r3_s05.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/classify_dynamic_batch_r3_s15.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/classify_dynamic_batch_r3_s2.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/loss_dynamic_batch_global_loss_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/loss_dynamic_batch_global_loss_r3_s05.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/loss_dynamic_batch_global_loss_r3_s15.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/loss_dynamic_batch_global_loss_r3_s2.yaml





# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/classify_dynamic_batch_wo_weights_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/classify_dynamic_batch_wo_weights_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/loss_dynamic_batch_global_loss_wo_weights_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_1666_client_100/loss_dynamic_batch_global_loss_wo_weights_r4_s1.yaml





# # ------------------ seed 16666 ---------------------------
# python3 -u ./centralized/resnet34_cinic10.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/centralized.yaml

# python3 -u ./data/generate_data.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/data.yaml

# python3 -u ./server/fedavg.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/fedavg.yaml

# python3 -u ./server/fedavg.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/fedavg_H.yaml

# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/classify_dynamic_batch_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/loss_dynamic_batch_global_loss_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/classify_dynamic_batch_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/loss_dynamic_batch_global_loss_r3_s1.yaml

# python3 -u ./server/fedbalancer.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/fedbalancer.yaml

# python3 -u ./server/fedsampling.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/fedsampling.yaml

# python3 -u ./server/ODE.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/ode.yaml

# python3 -u ./server/fedcase.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/fedcase.yaml



# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/classify_dynamic_batch_r1_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/classify_dynamic_batch_r2_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/classify_dynamic_batch_r5_s1.yaml

# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/loss_dynamic_batch_global_loss_r1_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/loss_dynamic_batch_global_loss_r2_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/loss_dynamic_batch_global_loss_r5_s1.yaml






# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/classify_dynamic_batch_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/classify_dynamic_batch_r3_s05.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/classify_dynamic_batch_r3_s15.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/classify_dynamic_batch_r3_s2.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/loss_dynamic_batch_global_loss_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/loss_dynamic_batch_global_loss_r3_s05.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/loss_dynamic_batch_global_loss_r3_s15.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/loss_dynamic_batch_global_loss_r3_s2.yaml





# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/classify_dynamic_batch_wo_weights_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/classify_dynamic_batch_wo_weights_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/loss_dynamic_batch_global_loss_wo_weights_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_16666_client_100/loss_dynamic_batch_global_loss_wo_weights_r4_s1.yaml






# # ------------------ seed 166666 ---------------------------
# python3 -u ./centralized/resnet34_cinic10.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/centralized.yaml

# python3 -u ./data/generate_data.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/data.yaml

# python3 -u ./server/fedavg.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/fedavg.yaml

# python3 -u ./server/fedavg.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/fedavg_H.yaml

# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/classify_dynamic_batch_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/loss_dynamic_batch_global_loss_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/classify_dynamic_batch_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/loss_dynamic_batch_global_loss_r3_s1.yaml

# python3 -u ./server/fedbalancer.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/fedbalancer.yaml

# python3 -u ./server/fedsampling.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/fedsampling.yaml

# python3 -u ./server/ODE.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/ode.yaml

# python3 -u ./server/fedcase.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/fedcase.yaml



# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/classify_dynamic_batch_r1_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/classify_dynamic_batch_r2_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/classify_dynamic_batch_r5_s1.yaml

# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/loss_dynamic_batch_global_loss_r1_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/loss_dynamic_batch_global_loss_r2_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/loss_dynamic_batch_global_loss_r5_s1.yaml






# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/classify_dynamic_batch_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/classify_dynamic_batch_r3_s05.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/classify_dynamic_batch_r3_s15.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/classify_dynamic_batch_r3_s2.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/loss_dynamic_batch_global_loss_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/loss_dynamic_batch_global_loss_r3_s05.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/loss_dynamic_batch_global_loss_r3_s15.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/loss_dynamic_batch_global_loss_r3_s2.yaml





# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/classify_dynamic_batch_wo_weights_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/classify_dynamic_batch_wo_weights_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/loss_dynamic_batch_global_loss_wo_weights_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet34_cinic10/dir_3_seed_166666_client_100/loss_dynamic_batch_global_loss_wo_weights_r4_s1.yaml



















#===============================================================================================================================================================
#-----------------------------resnet50 cifar100---------------------------------------
#===============================================================================================================================================================
# ---------------------- dir 0.3 ---------------------
# ------------------ seed 16 ---------------------------
# python3 -u ./centralized/resnet50_cifar100.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/centralized.yaml

# python3 -u ./data/generate_data.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/data.yaml

# python3 -u ./server/fedavg.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/fedavg.yaml

# python3 -u ./server/fedavg.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/fedavg_H.yaml

# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/classify_dynamic_batch_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/loss_dynamic_batch_global_loss_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/classify_dynamic_batch_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/loss_dynamic_batch_global_loss_r3_s1.yaml

# python3 -u ./server/fedbalancer.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/fedbalancer.yaml

# python3 -u ./server/fedsampling.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/fedsampling.yaml

# python3 -u ./server/ODE.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/ode.yaml

# python3 -u ./server/fedcase.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/fedcase.yaml





# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/classify_dynamic_batch_wo_weights_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/classify_dynamic_batch_wo_weights_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/loss_dynamic_batch_global_loss_wo_weights_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/loss_dynamic_batch_global_loss_wo_weights_r3_s1.yaml





# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/classify_dynamic_batch_r1_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/classify_dynamic_batch_r2_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/classify_dynamic_batch_r5_s1.yaml

# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/loss_dynamic_batch_global_loss_r1_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/loss_dynamic_batch_global_loss_r2_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/loss_dynamic_batch_global_loss_r5_s1.yaml





# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/classify_dynamic_batch_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/classify_dynamic_batch_r3_s05.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/classify_dynamic_batch_r3_s15.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/classify_dynamic_batch_r3_s2.yaml

# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/loss_dynamic_batch_global_loss_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/loss_dynamic_batch_global_loss_r3_s05.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/loss_dynamic_batch_global_loss_r3_s15.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/loss_dynamic_batch_global_loss_r3_s2.yaml








# ------------------ seed 166 ---------------------------
# python3 -u ./centralized/resnet50_cifar100.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/centralized.yaml

# python3 -u ./data/generate_data.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/data.yaml

# python3 -u ./server/fedavg.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/fedavg.yaml

# python3 -u ./server/fedavg.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/fedavg_H.yaml

# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/loss_dynamic_batch_global_loss_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/loss_dynamic_batch_global_loss_r3_s1.yaml

# python3 -u ./server/fedbalancer.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/fedbalancer.yaml

# python3 -u ./server/fedsampling.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/fedsampling.yaml

# python3 -u ./server/ODE.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/ode.yaml

# python3 -u ./server/fedcase.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/fedcase.yaml





# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_wo_weights_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/loss_dynamic_batch_global_loss_wo_weights_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_wo_weights_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/loss_dynamic_batch_global_loss_wo_weights_r3_s1.yaml





# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r1_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r2_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r5_s1.yaml

# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/loss_dynamic_batch_global_loss_r1_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/loss_dynamic_batch_global_loss_r2_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/loss_dynamic_batch_global_loss_r5_s1.yaml





# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r3_s05.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r3_s15.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r3_s2.yaml

# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/loss_dynamic_batch_global_loss_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/loss_dynamic_batch_global_loss_r3_s05.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/loss_dynamic_batch_global_loss_r3_s15.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/loss_dynamic_batch_global_loss_r3_s2.yaml






# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r1_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r2_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r4_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r5_s0.yaml





# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/loss_dynamic_batch_global_loss_r1_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/loss_dynamic_batch_global_loss_r2_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/loss_dynamic_batch_global_loss_r4_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/loss_dynamic_batch_global_loss_r5_s0.yaml





# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_wo_weights_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/loss_dynamic_batch_global_loss_wo_weights_r3_s0.yaml



# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r3_s0_96.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/woparallel_96.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r3_s0_128.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/woparallel_128.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r3_s0_160.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/woparallel_160.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r3_s0_192.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/woparallel_192.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r3_s0_256.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/woparallel_256.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r3_s0_512.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/woparallel_512.yaml


# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/random_select.yaml






# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_SD_r3_s0.yaml







# python3 -u ./server/fedavg.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/fedavg_512.yaml
# python3 -u ./server/fedavg.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/fedavg_96.yaml
# python3 -u ./server/fedavg.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/fedavg_256.yaml


# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r3_s0_96.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r3_s0_256.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/classify_dynamic_batch_r3_s0_512.yaml



# python3 -u ./server/fedbalancer.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/fedbalancer_96.yaml
# python3 -u ./server/fedbalancer.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/fedbalancer_256.yaml
# python3 -u ./server/fedbalancer.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/fedbalancer_512.yaml


# python3 -u ./server/fedsampling.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/fedsampling_96.yaml
# python3 -u ./server/fedsampling.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/fedsampling_256.yaml
# python3 -u ./server/fedsampling.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/fedsampling_512.yaml


# python3 -u ./server/fedcase.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/fedcase_96.yaml
# python3 -u ./server/fedcase.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/fedcase_256.yaml
# python3 -u ./server/fedcase.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/fedcase_512.yaml



# python3 -u ./server/ODE.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/ode_96.yaml
# python3 -u ./server/ODE.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/ode_256.yaml
# python3 -u ./server/ODE.py --config_path ./config/resnet50_cifar100/dir_3_seed_166_client_30/ode_512.yaml



# ------------------ seed 1666 ---------------------------
# python3 -u ./centralized/resnet50_cifar100.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/centralized.yaml

# python3 -u ./data/generate_data.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/data.yaml

# python3 -u ./server/fedavg.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/fedavg.yaml

# python3 -u ./server/fedavg.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/fedavg_H.yaml

# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/classify_dynamic_batch_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/loss_dynamic_batch_global_loss_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/classify_dynamic_batch_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/loss_dynamic_batch_global_loss_r3_s1.yaml

# python3 -u ./server/fedbalancer.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/fedbalancer.yaml

# python3 -u ./server/fedsampling.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/fedsampling.yaml

# python3 -u ./server/fedcase.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/fedcase.yaml

# python3 -u ./server/ODE.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/ode.yaml





# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/classify_dynamic_batch_wo_weights_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/loss_dynamic_batch_global_loss_wo_weights_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/classify_dynamic_batch_wo_weights_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/loss_dynamic_batch_global_loss_wo_weights_r3_s1.yaml





# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/classify_dynamic_batch_r1_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/classify_dynamic_batch_r2_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/classify_dynamic_batch_r5_s1.yaml

# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/loss_dynamic_batch_global_loss_r1_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/loss_dynamic_batch_global_loss_r2_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/loss_dynamic_batch_global_loss_r5_s1.yaml





# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/classify_dynamic_batch_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/classify_dynamic_batch_r3_s05.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/classify_dynamic_batch_r3_s15.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/classify_dynamic_batch_r3_s2.yaml

# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/loss_dynamic_batch_global_loss_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/loss_dynamic_batch_global_loss_r3_s05.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/loss_dynamic_batch_global_loss_r3_s15.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_1666_client_30/loss_dynamic_batch_global_loss_r3_s2.yaml





# ------------------ seed 16666 ---------------------------
# python3 -u ./centralized/resnet50_cifar100.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/centralized.yaml

# python3 -u ./data/generate_data.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/data.yaml

# python3 -u ./server/fedavg.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/fedavg.yaml

# python3 -u ./server/fedavg.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/fedavg_H.yaml

# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/classify_dynamic_batch_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/loss_dynamic_batch_global_loss_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/classify_dynamic_batch_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/loss_dynamic_batch_global_loss_r3_s1.yaml

# python3 -u ./server/fedbalancer.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/fedbalancer.yaml

# python3 -u ./server/fedsampling.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/fedsampling.yaml

# python3 -u ./server/ODE.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/ode.yaml

# python3 -u ./server/fedcase.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/fedcase.yaml





# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/classify_dynamic_batch_wo_weights_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/loss_dynamic_batch_global_loss_wo_weights_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/classify_dynamic_batch_wo_weights_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/loss_dynamic_batch_global_loss_wo_weights_r3_s1.yaml





# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/classify_dynamic_batch_r1_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/classify_dynamic_batch_r2_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/classify_dynamic_batch_r5_s1.yaml

# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/loss_dynamic_batch_global_loss_r1_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/loss_dynamic_batch_global_loss_r2_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/loss_dynamic_batch_global_loss_r5_s1.yaml





# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/classify_dynamic_batch_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/classify_dynamic_batch_r3_s05.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/classify_dynamic_batch_r3_s15.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/classify_dynamic_batch_r3_s2.yaml

# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/loss_dynamic_batch_global_loss_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/loss_dynamic_batch_global_loss_r3_s05.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/loss_dynamic_batch_global_loss_r3_s15.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_16666_client_30/loss_dynamic_batch_global_loss_r3_s2.yaml




# ------------------ seed 166666 ---------------------------
# python3 -u ./centralized/resnet50_cifar100.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/centralized.yaml

# python3 -u ./data/generate_data.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/data.yaml

# python3 -u ./server/fedavg.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/fedavg.yaml

# python3 -u ./server/fedavg.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/fedavg_H.yaml

# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/classify_dynamic_batch_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/loss_dynamic_batch_global_loss_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/classify_dynamic_batch_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/loss_dynamic_batch_global_loss_r3_s1.yaml

# python3 -u ./server/fedbalancer.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/fedbalancer.yaml

# python3 -u ./server/fedsampling.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/fedsampling.yaml

# python3 -u ./server/ODE.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/ode.yaml

# python3 -u ./server/fedcase.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/fedcase.yaml





# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/classify_dynamic_batch_wo_weights_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/loss_dynamic_batch_global_loss_wo_weights_r4_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/classify_dynamic_batch_wo_weights_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/loss_dynamic_batch_global_loss_wo_weights_r3_s1.yaml





# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/classify_dynamic_batch_r1_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/classify_dynamic_batch_r2_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/classify_dynamic_batch_r5_s1.yaml

# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/loss_dynamic_batch_global_loss_r1_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/loss_dynamic_batch_global_loss_r2_s1.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/loss_dynamic_batch_global_loss_r5_s1.yaml





# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/classify_dynamic_batch_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/classify_dynamic_batch_r3_s05.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/classify_dynamic_batch_r3_s15.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/classify_dynamic_batch_r3_s2.yaml

# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/loss_dynamic_batch_global_loss_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/loss_dynamic_batch_global_loss_r3_s05.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/loss_dynamic_batch_global_loss_r3_s15.yaml
# python3 -u ./server/my.py --config_path ./config/resnet50_cifar100/dir_3_seed_166666_client_30/loss_dynamic_batch_global_loss_r3_s2.yaml



























#===============================================================================================================================================================
#-----------------------------rnn snli---------------------------------------
#===============================================================================================================================================================
# ---------------------- dir 0.3 ---------------------
# ------------------ seed 16 ---------------------------

# python3 -u ./centralized/biRNN_snli.py --config_path ./config/rnn_snli/dir_3_seed_16_client_100/centralized.yaml

# python3 -u ./data/generate_data.py --config_path ./config/rnn_snli/dir_3_seed_16_client_100/data.yaml

# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_16_client_100/classify_dynamic_batch_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_16_client_100/classify_dynamic_batch_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_16_client_100/loss_dynamic_batch_global_loss_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_16_client_100/loss_dynamic_batch_global_loss_r3_s0.yaml

# python3 -u ./server/fedavg.py --config_path ./config/rnn_snli/dir_3_seed_16_client_100/fedavg.yaml

# python3 -u ./server/fedavg.py --config_path ./config/rnn_snli/dir_3_seed_16_client_100/fedavg_H.yaml

# python3 -u ./server/fedbalancer.py --config_path ./config/rnn_snli/dir_3_seed_16_client_100/fedbalancer.yaml

# python3 -u ./server/fedcase.py --config_path ./config/rnn_snli/dir_3_seed_16_client_100/fedcase.yaml

# python3 -u ./server/fedsampling.py --config_path ./config/rnn_snli/dir_3_seed_16_client_100/fedsampling.yaml

# python3 -u ./server/ODE.py --config_path ./config/rnn_snli/dir_3_seed_16_client_100/ode.yaml








# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_16_client_100/loss_dynamic_batch_global_loss_wo_weights_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_16_client_100/classify_dynamic_batch_wo_weights_r3_s1.yaml










# ------------------ seed 166 ---------------------------
# python3 -u ./centralized/biRNN_snli.py --config_path ./config/rnn_snli/dir_3_seed_166_client_100/centralized.yaml

# python3 -u ./data/generate_data.py --config_path ./config/rnn_snli/dir_3_seed_166_client_100/data.yaml

# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_166_client_100/classify_dynamic_batch_r3_s1.yaml
# python3 -u ./server/my.py --config_path  ./config/rnn_snli/dir_3_seed_166_client_100/classify_dynamic_batch_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_166_client_100/loss_dynamic_batch_global_loss_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_166_client_100/loss_dynamic_batch_global_loss_r3_s0.yaml

# python3 -u ./server/fedavg.py --config_path ./config/rnn_snli/dir_3_seed_166_client_100/fedavg.yaml

# python3 -u ./server/fedavg.py --config_path ./config/rnn_snli/dir_3_seed_166_client_100/fedavg_H.yaml

# python3 -u ./server/fedbalancer.py --config_path ./config/rnn_snli/dir_3_seed_166_client_100/fedbalancer.yaml

# python3 -u ./server/fedcase.py --config_path ./config/rnn_snli/dir_3_seed_166_client_100/fedcase.yaml

# python3 -u ./server/fedsampling.py --config_path ./config/rnn_snli/dir_3_seed_166_client_100/fedsampling.yaml

# python3 -u ./server/ODE.py --config_path ./config/rnn_snli/dir_3_seed_166_client_100/ode.yaml








# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_166_client_100/loss_dynamic_batch_global_loss_wo_weights_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_166_client_100/classify_dynamic_batch_wo_weights_r3_s1.yaml






# ------------------ seed 1666 ---------------------------
# python3 -u ./centralized/biRNN_snli.py --config_path ./config/rnn_snli/dir_3_seed_1666_client_100/centralized.yaml

# python3 -u ./data/generate_data.py --config_path ./config/rnn_snli/dir_3_seed_1666_client_100/data.yaml

# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_1666_client_100/classify_dynamic_batch_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_1666_client_100/classify_dynamic_batch_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_1666_client_100/loss_dynamic_batch_global_loss_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_1666_client_100/loss_dynamic_batch_global_loss_r3_s0.yaml

# python3 -u ./server/fedavg.py --config_path ./config/rnn_snli/dir_3_seed_1666_client_100/fedavg.yaml

# python3 -u ./server/fedavg.py --config_path ./config/rnn_snli/dir_3_seed_1666_client_100/fedavg_H.yaml

# python3 -u ./server/fedbalancer.py --config_path ./config/rnn_snli/dir_3_seed_1666_client_100/fedbalancer.yaml

# python3 -u ./server/fedcase.py --config_path ./config/rnn_snli/dir_3_seed_1666_client_100/fedcase.yaml

# python3 -u ./server/fedsampling.py --config_path ./config/rnn_snli/dir_3_seed_1666_client_100/fedsampling.yaml

# python3 -u ./server/ODE.py --config_path ./config/rnn_snli/dir_3_seed_1666_client_100/ode.yaml








# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_1666_client_100/loss_dynamic_batch_global_loss_wo_weights_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_1666_client_100/classify_dynamic_batch_wo_weights_r3_s1.yaml









# ------------------ seed 16666 ---------------------------
# python3 -u ./centralized/biRNN_snli.py --config_path ./config/rnn_snli/dir_3_seed_16666_client_100/centralized.yaml

# python3 -u ./data/generate_data.py --config_path ./config/rnn_snli/dir_3_seed_16666_client_100/data.yaml

# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_16666_client_100/classify_dynamic_batch_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_16666_client_100/classify_dynamic_batch_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_16666_client_100/loss_dynamic_batch_global_loss_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_16666_client_100/loss_dynamic_batch_global_loss_r3_s0.yaml


# python3 -u ./server/fedavg.py --config_path ./config/rnn_snli/dir_3_seed_16666_client_100/fedavg.yaml

# python3 -u ./server/fedavg.py --config_path ./config/rnn_snli/dir_3_seed_16666_client_100/fedavg_H.yaml

# python3 -u ./server/fedbalancer.py --config_path ./config/rnn_snli/dir_3_seed_16666_client_100/fedbalancer.yaml

# python3 -u ./server/fedcase.py --config_path ./config/rnn_snli/dir_3_seed_16666_client_100/fedcase.yaml

# python3 -u ./server/fedsampling.py --config_path ./config/rnn_snli/dir_3_seed_16666_client_100/fedsampling.yaml

# python3 -u ./server/ODE.py --config_path ./config/rnn_snli/dir_3_seed_16666_client_100/ode.yaml








# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_16666_client_100/loss_dynamic_batch_global_loss_wo_weights_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_16666_client_100/classify_dynamic_batch_wo_weights_r3_s1.yaml






# ------------------ seed 166666 ---------------------------
# python3 -u ./centralized/biRNN_snli.py --config_path ./config/rnn_snli/dir_3_seed_166666_client_100/centralized.yaml

# python3 -u ./data/generate_data.py --config_path ./config/rnn_snli/dir_3_seed_166666_client_100/data.yaml

# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_166666_client_100/classify_dynamic_batch_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_166666_client_100/classify_dynamic_batch_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_166666_client_100/loss_dynamic_batch_global_loss_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_166666_client_100/loss_dynamic_batch_global_loss_r3_s0.yaml


# python3 -u ./server/fedavg.py --config_path ./config/rnn_snli/dir_3_seed_166666_client_100/fedavg.yaml

# python3 -u ./server/fedavg.py --config_path ./config/rnn_snli/dir_3_seed_166666_client_100/fedavg_H.yaml

# python3 -u ./server/fedbalancer.py --config_path ./config/rnn_snli/dir_3_seed_166666_client_100/fedbalancer.yaml

# python3 -u ./server/fedcase.py --config_path ./config/rnn_snli/dir_3_seed_166666_client_100/fedcase.yaml

# python3 -u ./server/fedsampling.py --config_path ./config/rnn_snli/dir_3_seed_166666_client_100/fedsampling.yaml

# python3 -u ./server/ODE.py --config_path ./config/rnn_snli/dir_3_seed_166666_client_100/ode.yaml








# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_166666_client_100/loss_dynamic_batch_global_loss_wo_weights_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/rnn_snli/dir_3_seed_166666_client_100/classify_dynamic_batch_wo_weights_r3_s1.yaml








































#===============================================================================================================================================================
#-----------------------------vit domainnet---------------------------------------
#===============================================================================================================================================================
# ---------------------- dir 0.3 ---------------------
# ------------------ seed 16 ---------------------------

# python3 -u ./centralized/vit_domainnet.py --config_path ./config/vit_domainnet/dir_1_seed_16_client_30/centralized.yaml


# python3 -u ./data/generate_data.py --config_path ./config/vit_domainnet/dir_1_seed_16_client_30/data.yaml

# python3 -u ./server/fedavg.py --config_path ./config/vit_domainnet/dir_1_seed_16_client_30/fedavg.yaml

# python3 -u ./server/fedavg.py --config_path ./config/vit_domainnet/dir_1_seed_16_client_30/fedavg_H.yaml

# python3 -u ./server/my.py --config_path ./config/vit_domainnet/dir_1_seed_16_client_30/classify_dynamic_batch_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/vit_domainnet/dir_1_seed_16_client_30/classify_dynamic_batch_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/vit_domainnet/dir_1_seed_16_client_30/loss_dynamic_batch_global_loss_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/vit_domainnet/dir_1_seed_16_client_30/loss_dynamic_batch_global_loss_r3_s0.yaml


# python3 -u ./server/fedbalancer.py --config_path ./config/vit_domainnet/dir_1_seed_16_client_30/fedbalancer.yaml


# python3 -u ./server/fedcase.py --config_path ./config/vit_domainnet/dir_1_seed_16_client_30/fedcase.yaml


# python3 -u ./server/fedsampling.py --config_path ./config/vit_domainnet/dir_1_seed_16_client_30/fedsampling.yaml


# python3 -u ./server/ODE.py --config_path ./config/vit_domainnet/dir_1_seed_16_client_30/ode.yaml







# ------------------ seed 166 ---------------------------

# python3 -u ./centralized/vit_domainnet.py --config_path ./config/vit_domainnet/dir_1_seed_166_client_30/centralized.yaml

# python3 -u ./data/generate_data.py --config_path ./config/vit_domainnet/dir_1_seed_166_client_30/data.yaml

# python3 -u ./server/fedavg.py --config_path ./config/vit_domainnet/dir_1_seed_166_client_30/fedavg.yaml

# python3 -u ./server/fedavg.py --config_path ./config/vit_domainnet/dir_1_seed_166_client_30/fedavg_H.yaml


# python3 -u ./server/my.py --config_path ./config/vit_domainnet/dir_1_seed_166_client_30/classify_dynamic_batch_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/vit_domainnet/dir_1_seed_166_client_30/classify_dynamic_batch_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/vit_domainnet/dir_1_seed_166_client_30/loss_dynamic_batch_global_loss_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/vit_domainnet/dir_1_seed_166_client_30/loss_dynamic_batch_global_loss_r3_s0.yaml

# python3 -u ./server/fedbalancer.py --config_path ./config/vit_domainnet/dir_1_seed_166_client_30/fedbalancer.yaml


# python3 -u ./server/fedcase.py --config_path ./config/vit_domainnet/dir_1_seed_166_client_30/fedcase.yaml


# python3 -u ./server/fedsampling.py --config_path ./config/vit_domainnet/dir_1_seed_166_client_30/fedsampling.yaml


# python3 -u ./server/ODE.py --config_path ./config/vit_domainnet/dir_1_seed_166_client_30/ode.yaml






# ------------------ seed 1666 ---------------------------
# python3 -u ./centralized/vit_domainnet.py --config_path ./config/vit_domainnet/dir_1_seed_1666_client_30/centralized.yaml

# python3 -u ./data/generate_data.py --config_path ./config/vit_domainnet/dir_1_seed_1666_client_30/data.yaml

# python3 -u ./server/fedavg.py --config_path ./config/vit_domainnet/dir_1_seed_1666_client_30/fedavg.yaml

# python3 -u ./server/fedavg.py --config_path ./config/vit_domainnet/dir_1_seed_1666_client_30/fedavg_H.yaml

# python3 -u ./server/my.py --config_path ./config/vit_domainnet/dir_1_seed_1666_client_30/classify_dynamic_batch_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/vit_domainnet/dir_1_seed_1666_client_30/classify_dynamic_batch_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/vit_domainnet/dir_1_seed_1666_client_30/loss_dynamic_batch_global_loss_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/vit_domainnet/dir_1_seed_1666_client_30/loss_dynamic_batch_global_loss_r3_s0.yaml

# python3 -u ./server/fedbalancer.py --config_path ./config/vit_domainnet/dir_1_seed_1666_client_30/fedbalancer.yaml


# python3 -u ./server/fedcase.py --config_path ./config/vit_domainnet/dir_1_seed_1666_client_30/fedcase.yaml


# python3 -u ./server/fedsampling.py --config_path ./config/vit_domainnet/dir_1_seed_1666_client_30/fedsampling.yaml


# python3 -u ./server/ODE.py --config_path ./config/vit_domainnet/dir_1_seed_1666_client_30/ode.yaml






# ------------------ seed 16666 ---------------------------
# python3 -u ./centralized/vit_domainnet.py --config_path ./config/vit_domainnet/dir_1_seed_16666_client_30/centralized.yaml

# python3 -u ./data/generate_data.py --config_path ./config/vit_domainnet/dir_1_seed_16666_client_30/data.yaml

# python3 -u ./server/fedavg.py --config_path ./config/vit_domainnet/dir_1_seed_16666_client_30/fedavg.yaml

# python3 -u ./server/fedavg.py --config_path ./config/vit_domainnet/dir_1_seed_16666_client_30/fedavg_H.yaml

# python3 -u ./server/my.py --config_path ./config/vit_domainnet/dir_1_seed_16666_client_30/classify_dynamic_batch_r3_s1.yaml
# python3 -u ./server/my.py --config_path ./config/vit_domainnet/dir_1_seed_16666_client_30/classify_dynamic_batch_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/vit_domainnet/dir_1_seed_16666_client_30/loss_dynamic_batch_global_loss_r3_s1.yaml


# python3 -u ./server/fedbalancer.py --config_path ./config/vit_domainnet/dir_1_seed_16666_client_30/fedbalancer.yaml


# python3 -u ./server/fedcase.py --config_path ./config/vit_domainnet/dir_1_seed_16666_client_30/fedcase.yaml


# python3 -u ./server/fedsampling.py --config_path ./config/vit_domainnet/dir_1_seed_16666_client_30/fedsampling.yaml


# python3 -u ./server/ODE.py --config_path ./config/vit_domainnet/dir_1_seed_16666_client_30/ode.yaml







# ------------------ seed 166666 ---------------------------
# python3 -u ./centralized/vit_domainnet.py --config_path ./config/vit_domainnet/dir_1_seed_166666_client_30/centralized.yaml

# python3 -u ./data/generate_data.py --config_path ./config/vit_domainnet/dir_1_seed_166666_client_30/data.yaml

# python3 -u ./server/fedavg.py --config_path ./config/vit_domainnet/dir_1_seed_166666_client_30/fedavg.yaml

# python3 -u ./server/fedavg.py --config_path ./config/vit_domainnet/dir_1_seed_166666_client_30/fedavg_H.yaml


# python3 -u ./server/my.py --config_path ./config/vit_domainnet/dir_1_seed_166666_client_30/classify_dynamic_batch_r3_s1.yaml
# python3 -u ./server/my.py --config_path  ./config/vit_domainnet/dir_1_seed_166666_client_30/classify_dynamic_batch_r3_s0.yaml
# python3 -u ./server/my.py --config_path ./config/vit_domainnet/dir_1_seed_166666_client_30/loss_dynamic_batch_global_loss_r3_s1.yaml


# python3 -u ./server/fedbalancer.py --config_path ./config/vit_domainnet/dir_1_seed_166666_client_30/fedbalancer.yaml


# python3 -u ./server/fedcase.py --config_path ./config/vit_domainnet/dir_1_seed_166666_client_30/fedcase.yaml


# python3 -u ./server/fedsampling.py --config_path ./config/vit_domainnet/dir_1_seed_166666_client_30/fedsampling.yaml


# python3 -u ./server/ODE.py --config_path ./config/vit_domainnet/dir_1_seed_166666_client_30/ode.yaml