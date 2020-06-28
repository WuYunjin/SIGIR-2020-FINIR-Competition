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
depth='10'
day='20'
log='0.6'
for metal in 0 1 2 3 5 
do
    for train in 400 450 500 550 600 #650 600 550 500
    do
        for val in 1 3 5 10 #25 20 15 10 5 3 1
        do
            for prob in 0.7 0.65 0.6 0.55 0.5 0.45 0.4
            do      
                python stage2/XGB/test_search.py -t=$train -v=$val -d=$depth -e=$eta -p=$prob -n=$n_estimators -m=$metal -D=$day -l=$log
            done
        done
    done
done
