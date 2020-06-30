# parser.add_argument('-t' ,'--train_data_len' , type=int , help='training len', default=500)
# parser.add_argument('-v','--valdata_len', type=int, help='valuation len',default=1)
# parser.add_argument('-p'  ,'--prob' , type = float , help = 'xgb prob output threshold' , default=0.5)
# parser.add_argument('-d','--max_depth', type=int, help='xgb max_length', default=8)
# parser.add_argument('-e'  ,'--eta' , type = float , help = 'xgb eta' , default=0.01)
# parser.add_argument('-g'  ,'--gamma' , type = float , help = 'xgb gamma' , default=0.0)
# parser.add_argument('-m'  ,'--metal' , type = int , help = 'index of the target metal' , default=0)
# parser.add_argument('-n' , '--n_estimators' , type=int  , help='num of the estimate time' , default=50)
# parser.add_argument('-D' , '--day' , type=int  , help='day to predict' , default=1)
# parser.add_argument('-l' , '--log' , type=float  , help='log the result better than this value' , default=0.55)
# parser.add_argument('-u' , '--use_diff' , type=bool  , help='Whether to add diff to the feature' , default=False)
n_estimators='50'
eta='0.01'
depth='10'
day='60'
log='0.65'
#不要用逗号，定义变量后面不能直接加注释
use_diff_list=("False" "True")


for metal in 0 #0 1 2 3 4 5 
do
    for train in 100 150 200 250 300 #350 400 450 500 #650 600 550 500
    do
        for val in 1 3 5 10 #25 20 15 10 5 3 1
        do
            for prob in  0.7 0.65 0.6 0.55 0.45 0.4 0.3 #0.7 0.65 0.6
            do   
                for use_diff in ${use_diff_list[@]}
                do
                    python stage2/XGB/test_search.py -t=$train -v=$val -d=$depth -e=$eta -p=$prob -n=$n_estimators -m=$metal -D=$day -l=$log -u=$use_diff
                done
            done
        done
    done
done

