# Reporte de experimentos

| strategy_id | experiment_name | changed_component | status | gender_accuracy | gender_f1 | age_mae | age_rmse | age_r2 | trainable_parameters | training_seconds | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | classical_base | ninguno | COMPLETADO | 0.8010 | 0.8002 | 11.4897 | 15.2479 | 0.3890 | - | 15.2990 |  |
| E1 | classical_pca_50 | PCA=50 componentes | COMPLETADO | 0.7832 | 0.7821 | 11.3484 | 15.0982 | 0.4009 | - | 11.7551 |  |
| E1 | classical_pca_200 | PCA=200 componentes | COMPLETADO | 0.7830 | 0.7830 | 11.6537 | 15.4351 | 0.3739 | - | 22.7762 |  |
| E2 | mlp_base | ninguno | COMPLETADO | 0.8465 | 0.8465 | 12.2991 | 16.8009 | 0.2582 | 77136899 | 133.8626 |  |
| E2 | mlp_no_dropout | dropout=0.0 | COMPLETADO | 0.8577 | 0.8577 | 11.7485 | 16.5388 | 0.2812 | 77136899 | 132.2027 |  |
| E2 | mlp_lambda_low | lambda_age=0.001 | COMPLETADO | 0.8431 | 0.8431 | 16.9758 | 23.7452 | -0.4818 | 77136899 | 133.6641 |  |
| E2 | mlp_lambda_high | lambda_age=0.1 | COMPLETADO | 0.8406 | 0.8403 | 10.6029 | 14.8253 | 0.4224 | 77136899 | 134.3023 |  |
| E3 | cnn_base | ninguno | COMPLETADO | 0.8746 | 0.8744 | 9.6607 | 13.2543 | 0.5383 | 286755 | 133.0638 |  |
| E3 | cnn_no_augmentation | sin aumentacion | COMPLETADO | 0.8766 | 0.8762 | 9.6720 | 13.5238 | 0.5194 | 286755 | 85.4907 |  |
| E3 | cnn_no_dropout | dropout=0.0 | COMPLETADO | 0.8811 | 0.8808 | 9.4201 | 12.9366 | 0.5602 | 286755 | 137.5448 |  |
| E3 | cnn_lambda_low | lambda_age=0.001 | COMPLETADO | 0.8760 | 0.8757 | 12.8291 | 18.3405 | 0.1160 | 286755 | 133.9898 |  |
| E3 | cnn_lambda_high | lambda_age=0.1 | COMPLETADO | 0.8521 | 0.8519 | 9.3570 | 12.9573 | 0.5588 | 286755 | 136.5507 |  |
| E4 | resnet_frozen_base | ninguno | COMPLETADO | 0.8367 | 0.8363 | 10.5745 | 14.0454 | 0.4816 | 1539 | 131.6568 |  |
| E4 | resnet_frozen_no_augmentation | sin aumentacion | COMPLETADO | 0.8310 | 0.8307 | 10.5627 | 14.0809 | 0.4789 | 1539 | 82.3529 |  |
| E4 | resnet_frozen_lambda_low | lambda_age=0.001 | COMPLETADO | 0.8389 | 0.8389 | 11.2315 | 15.2017 | 0.3927 | 1539 | 138.9502 |  |
| E4 | resnet_frozen_lambda_high | lambda_age=0.1 | COMPLETADO | 0.8367 | 0.8363 | 10.5218 | 13.9497 | 0.4886 | 1539 | 137.7559 |  |
| E5 | resnet_finetuning_base | layer4 descongelada | COMPLETADO | 0.9106 | 0.9106 | 6.9417 | 9.4962 | 0.7630 | 8395267 | 133.6110 |  |
| E5 | resnet_finetuning_unfreeze_more | layer4+layer3 descongeladas | COMPLETADO | 0.9224 | 0.9225 | 6.1008 | 8.6022 | 0.8055 | 10494979 | 129.2471 |  |
| E5 | resnet_finetuning_lr_low | learning rate menor (1e-5) | COMPLETADO | 0.9089 | 0.9089 | 13.0883 | 18.6242 | 0.0884 | 8395267 | 131.4867 |  |
| E5 | resnet_finetuning_lambda_high | lambda_age=0.1 | COMPLETADO | 0.9159 | 0.9159 | 5.7752 | 8.3604 | 0.8163 | 8395267 | 133.5335 |  |
| E6 | cnn_lambda_e6_0001 | lambda_age=0.001 | COMPLETADO | 0.8760 | 0.8757 | 12.8291 | 18.3405 | 0.1160 | 286755 | 132.1828 |  |
| E6 | cnn_lambda_e6_001 | lambda_age=0.01 (referencia) | COMPLETADO | 0.8746 | 0.8744 | 9.6607 | 13.2543 | 0.5383 | 286755 | 130.9733 |  |
| E6 | cnn_lambda_e6_01 | lambda_age=0.1 | COMPLETADO | 0.8521 | 0.8519 | 9.3570 | 12.9573 | 0.5588 | 286755 | 130.1819 |  |
| E6 | cnn_lambda_e6_1 | lambda_age=1.0 | COMPLETADO | 0.8035 | 0.8015 | 8.9831 | 12.5987 | 0.5829 | 286755 | 131.3148 |  |
| E6 | cnn_lambda_e6_10 | lambda_age=10.0 | COMPLETADO | 0.7495 | 0.7497 | 9.2154 | 12.8461 | 0.5663 | 286755 | 130.4626 |  |
