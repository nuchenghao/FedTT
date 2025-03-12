python3 -u ./data/generate_data.py --config_path ./config/resnet50_cifar100/dir_3_seed_16_client_30/data.yaml



# python3 -u ./on_device/fedavg_server.py --config_path ./on_device/config/fedavg.yaml

# python3 -u ./on_device/fedavg_client.py --config_path ./on_device/config/fedavg.yaml --name AGX_1

# python3 -u ./on_device/fedavg_client.py --config_path ./on_device/config/fedavg.yaml --name AGX_2

# python3 -u ./on_device/fedavg_client.py --config_path ./on_device/config/fedavg.yaml --name ORIN_1









# python3 -u ./on_device/my_server.py --config_path ./on_device/config/classify_dynamic_batch_r3_s1.yaml

# python3 -u ./on_device/my_client.py --config_path ./on_device/config/classify_dynamic_batch_r3_s1.yaml --name AGX_1

# python3 -u ./on_device/my_client.py --config_path ./on_device/config/classify_dynamic_batch_r3_s1.yaml --name AGX_2

# python3 -u ./on_device/my_client.py --config_path ./on_device/config/classify_dynamic_batch_r3_s1.yaml --name ORIN_1







python3 -u ./on_device/my_server.py --config_path ./on_device/config/loss_dynamic_batch_global_loss_r3_s1.yaml

# python3 -u ./on_device/my_client.py --config_path ./on_device/config/loss_dynamic_batch_global_loss_r3_s1.yaml --name AGX_1

# python3 -u ./on_device/my_client.py --config_path ./on_device/config/loss_dynamic_batch_global_loss_r3_s1.yaml --name AGX_2

# python3 -u ./on_device/my_client.py --config_path ./on_device/config/loss_dynamic_batch_global_loss_r3_s1.yaml --name ORIN_1









# python3 -u ./on_device/fedbalancer_server.py --config_path ./on_device/config/fedbalancer.yaml

# python3 -u ./on_device/fedbalancer_client.py --config_path ./on_device/config/fedbalancer.yaml --name AGX_1

# python3 -u ./on_device/fedbalancer_client.py --config_path ./on_device/config/fedbalancer.yaml --name AGX_2

# python3 -u ./on_device/fedbalancer_client.py --config_path ./on_device/config/fedbalancer.yaml --name ORIN_1










# python3 -u ./on_device/fedsampling_server.py --config_path ./on_device/config/fedsampling.yaml

# python3 -u ./on_device/fedsampling_client.py --config_path ./on_device/config/fedsampling.yaml --name AGX_1

# python3 -u ./on_device/fedsampling_client.py --config_path ./on_device/config/fedsampling.yaml --name AGX_2

# python3 -u ./on_device/fedsampling_client.py --config_path ./on_device/config/fedsampling.yaml --name ORIN_1











# python3 -u ./on_device/fedcase_server.py --config_path ./on_device/config/fedcase.yaml

# python3 -u ./on_device/fedcase_client.py --config_path ./on_device/config/fedcase.yaml --name AGX_1

# python3 -u ./on_device/fedcase_client.py --config_path ./on_device/config/fedcase.yaml --name AGX_2

# python3 -u ./on_device/fedcase_client.py --config_path ./on_device/config/fedcase.yaml --name ORIN_1