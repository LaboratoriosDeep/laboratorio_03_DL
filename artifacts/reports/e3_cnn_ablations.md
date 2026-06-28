# Reporte de experimentos

| strategy_id | experiment_name | changed_component | status | gender_accuracy | gender_f1 | age_mae | age_rmse | age_r2 | trainable_parameters | training_seconds | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E3 | cnn_base | ninguno | COMPLETADO | 0.8746 | 0.8744 | 9.6607 | 13.2543 | 0.5383 | 286755 | 133.0638 |  |
| E3 | cnn_no_augmentation | sin aumentacion | COMPLETADO | 0.8766 | 0.8762 | 9.6720 | 13.5238 | 0.5194 | 286755 | 85.4907 |  |
| E3 | cnn_no_dropout | dropout=0.0 | COMPLETADO | 0.8811 | 0.8808 | 9.4201 | 12.9366 | 0.5602 | 286755 | 137.5448 |  |
| E3 | cnn_lambda_low | lambda_age=0.001 | COMPLETADO | 0.8760 | 0.8757 | 12.8291 | 18.3405 | 0.1160 | 286755 | 133.9898 |  |
| E3 | cnn_lambda_high | lambda_age=0.1 | COMPLETADO | 0.8521 | 0.8519 | 9.3570 | 12.9573 | 0.5588 | 286755 | 136.5507 |  |
