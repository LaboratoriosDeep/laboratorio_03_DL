# Reporte de experimentos

| strategy_id | experiment_name | changed_component | status | gender_accuracy | gender_f1 | age_mae | age_rmse | age_r2 | trainable_parameters | training_seconds | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E4 | resnet_frozen_base | ninguno | COMPLETADO | 0.8367 | 0.8363 | 10.5745 | 14.0454 | 0.4816 | 1539 | 131.6568 |  |
| E4 | resnet_frozen_no_augmentation | sin aumentacion | COMPLETADO | 0.8310 | 0.8307 | 10.5627 | 14.0809 | 0.4789 | 1539 | 82.3529 |  |
| E4 | resnet_frozen_lambda_low | lambda_age=0.001 | COMPLETADO | 0.8389 | 0.8389 | 11.2315 | 15.2017 | 0.3927 | 1539 | 138.9502 |  |
| E4 | resnet_frozen_lambda_high | lambda_age=0.1 | COMPLETADO | 0.8367 | 0.8363 | 10.5218 | 13.9497 | 0.4886 | 1539 | 137.7559 |  |
