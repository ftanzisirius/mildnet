source settings.cfg

source $1
gcloud ai-platform jobs submit training $2 \
--module-name=trainer.task \
--package-path=trainer/ \
--job-dir=$MILDNET_JOB_DIR$2 \
--region=$MILDNET_REGION \
--config=$config \
-- \
--data-path=$MILDNET_DATA_PATH \
--model-id=$model_id \
--loss=$loss \
--optimizer=$optimizer \
--train-csv=$train_csv \
--val-csv=$val_csv \
--train-epocs=$train_epocs \
--batch-size=$batch_size \
--lr=$lr