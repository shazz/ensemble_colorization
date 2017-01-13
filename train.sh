export EPOCH=30
export MODEL_SAVE_RATE=30000
export IMAGE_SAVE_RATE=10
echo Training red model
python3 train.py dataset/sorted/red dataset/summary/red --epochs $EPOCH --model-save-rate $MODEL_SAVE_RATE --image-save-rate $IMAGE_SAVE_RATE --final-model model/model_red
echo training green model
python3 train.py dataset/sorted/green dataset/summary/green --epochs $EPOCH --model-save-rate $MODEL_SAVE_RATE --image-save-rate $IMAGE_SAVE_RATE --final-model model/model_green
echo training blue model
python3 train.py dataset/sorted/blue dataset/summary/blue --epochs $EPOCH --model-save-rate $MODEL_SAVE_RATE --image-save-rate $IMAGE_SAVE_RATE --final-model model/model_blue
echo training blue/green model
python3 train.py dataset/sorted/blue_green dataset/summary/blue_green --epochs $EPOCH --model-save-rate $MODEL_SAVE_RATE --image-save-rate $IMAGE_SAVE_RATE --final-model model/model_blue_green
echo Done !

