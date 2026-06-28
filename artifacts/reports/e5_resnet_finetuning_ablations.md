# Reporte de experimentos

| strategy_id | experiment_name | changed_component | status | gender_accuracy | gender_f1 | age_mae | age_rmse | age_r2 | trainable_parameters | training_seconds | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E5 | resnet_finetuning_base | layer4 descongelada | COMPLETADO | 0.9106 | 0.9106 | 6.9417 | 9.4962 | 0.7630 | 8395267 | 133.6110 |  |
| E5 | resnet_finetuning_unfreeze_more | layer4+layer3 descongeladas | COMPLETADO | 0.9224 | 0.9225 | 6.1008 | 8.6022 | 0.8055 | 10494979 | 129.2471 |  |
| E5 | resnet_finetuning_lr_low | learning rate menor (1e-5) | COMPLETADO | 0.9089 | 0.9089 | 13.0883 | 18.6242 | 0.0884 | 8395267 | 131.4867 |  |
| E5 | resnet_finetuning_lambda_high | lambda_age=0.1 | COMPLETADO | 0.9159 | 0.9159 | 5.7752 | 8.3604 | 0.8163 | 8395267 | 133.5335 |  |
