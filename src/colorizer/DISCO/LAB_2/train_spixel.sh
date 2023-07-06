model_name="lab_3"
model_type="best"
data_dir="/home/work/Circuit/dataset/"

python -m main.spixelseg.train_spixel --dataset custom --epochs 100 --data_dir $data_dir --save_dir model/$model_name --exp_name spixel