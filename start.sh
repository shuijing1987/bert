export BERT_BASE_DIR=./GLUE/BERT_BASE_DIR/uncased_L-24_H-1024_A-16
export GLUE_DIR=./GLUE
rm -rf /tmp/mrpc_output/

/usr/local/bin/python3.9 run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/glue_data/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/tmp/mrpc_output/
