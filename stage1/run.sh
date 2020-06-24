# parser = argparse.ArgumentParser(description="xgboost")
# parser.add_argument('-t' ,'--train_data_len' , type=int , help='训练长度', default=500)
# parser.add_argument('-v','--valdata_len', type=str, help='迭代时的长度',default=20)
# parser.add_argument('-d','--max_depth', type=int, help='xgb max_length', default=7)
# parser.add_argument('-e'  ,'--eta' , type = float , help = 'xgb eta' , default=0.1)
# parser.add_argument('-g'  ,'--gamma' , type = float , help = 'xgb gamma' , default=0.3)
# parser.add_argument('-p'  ,'--prob' , type = float , help = 'xgb prob output threshold' , default=0.5)
# parser.add_argument('-n' , '--n_estimator' , type=int  , help='num of the estimate time' , default=100)
# args = parser.parse_args()
n_estimators='50'
eta='0.01'
depth='8'
for train in 650 #500 550 600
do
  for val in 15 20 25 1 5 10
  do
    for prob in 0.55 0.6 0.65
        do      
        python stack/xgb_main.py -t=$train -v=$val -d=$depth -e=$eta -p=$prob -n=$n_estimators  
        done
  done
done