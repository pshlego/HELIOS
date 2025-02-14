export Root_Data_Path=/mnt/sdd
export Modelcheckpoint_PATH=${Root_Data_Path}/OTT-QAMountSpace/ModelCheckpoints/Ours/llm/Meta-Llama-3.1-8B-Instruct

docker run --gpus all \
    -p 30000:30000 \
    --env "CUDA_VISIBLE_DEVICES=0,1,2,3" \
    --ipc=host \
    --mount type=bind,source=/mnt,target=/mnt \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path ${Modelcheckpoint_PATH} --host 0.0.0.0 --port 30000 --dp 4