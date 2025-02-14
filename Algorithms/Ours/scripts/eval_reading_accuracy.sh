export Root_Data_Path=/mnt/sdd
export Root_Path=/root/HELIOS
export CUDA_VISIBLE_DEVICES=0
python ${Root_Path}/Algorithms/ChainOfSkills/FiE_reader/train_qa_fie.py \
--do_predict \
--model_name google/electra-large-discriminator \
--train_batch_size 2 \
--gradient_accumulation_steps 2 \
--predict_batch_size 1 \
--output_dir reader_output \
--num_train_steps 5000 \
--use_layer_lr \
--layer_decay 0.9 \
--eval-period 500 \
--learning_rate 5e-5 \
--max_ans_len 15 \
--gradient_checkpointing \
--num_ctx 50 \
--max_grad_norm 1.0 \
--train_file ${Root_Data_Path}/OTT-QAMountSpace/Dataset/COS/ott_train_reader.json \
--predict_file reader_input_path.json \
--init_checkpoint ${Root_Data_Path}/OTT-QAMountSpace/ModelCheckpoints/COS/ott_fie_checkpoint_best.pt \
--save_prediction dev_results