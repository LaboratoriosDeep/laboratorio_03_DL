# Reporte de experimentos

| strategy_id | experiment_name | changed_component | status | gender_accuracy | gender_f1 | age_mae | age_rmse | age_r2 | trainable_parameters | training_seconds | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E6 | cnn_lambda_e6_0001 | lambda_age=0.001 | COMPLETADO | 0.8760 | 0.8757 | 12.8291 | 18.3405 | 0.1160 | 286755 | 132.1828 |  |
| E6 | cnn_lambda_e6_001 | lambda_age=0.01 (referencia) | COMPLETADO | 0.8746 | 0.8744 | 9.6607 | 13.2543 | 0.5383 | 286755 | 130.9733 |  |
| E6 | cnn_lambda_e6_01 | lambda_age=0.1 | COMPLETADO | 0.8521 | 0.8519 | 9.3570 | 12.9573 | 0.5588 | 286755 | 130.1819 |  |
| E6 | cnn_lambda_e6_1 | lambda_age=1.0 | COMPLETADO | 0.8035 | 0.8015 | 8.9831 | 12.5987 | 0.5829 | 286755 | 131.3148 |  |
| E6 | cnn_lambda_e6_10 | lambda_age=10.0 | COMPLETADO | 0.7495 | 0.7497 | 9.2154 | 12.8461 | 0.5663 | 286755 | 130.4626 |  |
