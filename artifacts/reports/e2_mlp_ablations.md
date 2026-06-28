# Reporte de experimentos

| strategy_id | experiment_name | changed_component | status | gender_accuracy | gender_f1 | age_mae | age_rmse | age_r2 | trainable_parameters | training_seconds | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E2 | mlp_base | ninguno | COMPLETADO | 0.8465 | 0.8465 | 12.2991 | 16.8009 | 0.2582 | 77136899 | 133.8626 |  |
| E2 | mlp_no_dropout | dropout=0.0 | COMPLETADO | 0.8577 | 0.8577 | 11.7485 | 16.5388 | 0.2812 | 77136899 | 132.2027 |  |
| E2 | mlp_lambda_low | lambda_age=0.001 | COMPLETADO | 0.8431 | 0.8431 | 16.9758 | 23.7452 | -0.4818 | 77136899 | 133.6641 |  |
| E2 | mlp_lambda_high | lambda_age=0.1 | COMPLETADO | 0.8406 | 0.8403 | 10.6029 | 14.8253 | 0.4224 | 77136899 | 134.3023 |  |
