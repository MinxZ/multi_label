sudo python data_split.py

# for model_name in "ResNet50" "DenseNet121" "DenseNet169" ;
for model_name in "ResNet50" "DenseNet169" ;
do
  echo $model_name
  python train.py --model $model_name
  python test.py --model $model_name
done
