model_name="lab_1"
model_type="best"
data_dir="/home/work/Circuit/dataset/"

python -m main.colorizer.train_colorizer --model AnchorColorProb --dataset custom --batch_size 16 --data_dir $data_dir --save_dir model/$model_name --ckpt_dir  model/$model_name/spixel/checkpts/model_$model_type.pth.tar --exp_name disco --dense_pos --enhanced