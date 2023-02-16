# MIMIC-III
## Have to create the data first (using Harutyunyan's code) for each task including 1. mortality 2. decompensation.
    The outputs from creation are 
1. data folder with subfolder "train" and "test" (we will then have /path/to/data/train or /path/to/data/test)
2. listfiles: train_listfile.csv, test_listfile.csv, val_listfile.csv (we will then have /path/to/listfiles/)

## There are 2 tasks of interest
1. In-house Mortality Task
2. Decompensation Task

## There are 4 models involved
1. Adacare
2. RETAIN
3. SoPa (or RLR or RRR)
4. Logistic Regression  

To replicate the result on In-house mortality, use the followings to train the model 
1. In-house Mortality Task
e.g. /path/to/data/='./mimic3benchmark/in-hospital-mortality/'

1.1. Adacare
python -u Adacare_ihm.py --kernel_num=64 --kernel_size=2 --lr=0.001 --output_dim=1 --r_conv=4 --r_visit=4 --rnn_dim=384 --batch_size=100 --data='/path/to/data/' --gpu='5' --file_name='210429_21_18_Adacare_ihm'

1.2 RETAIN
python -u 210424_21_00_RETAIN_v2_ihm.py --batch_size=100 --clip=0.0 --data=/path/to/data/ --dim_alpha=128 --dim_beta=128 --dropout_context=0.0 --dropout_emb=0.0 --emb_dim=128 --epochs=100 --file_name='210310_13_38_RETAIN_v2_ihm_128emb_128alpha_128beta' --gpu='1' --lr=0.001 --num_features=76 --timestep=1.0

1.3 SoPa
python -u 201128_03_34_Train_SoPa_MLP_Inhousemort_Deepsuperv_LogSpaceMaxTimesSemiring.py --pattern_specs="10-2_20-2_50-2" --batch_size=128 --dropout=0.0 --clip=0 --mlp_hidden_dim=0 --num_mlp_layers=1 --target_repl_coef=0.5 --epochs=200 --file_name="210128_11_00_10-2_20-2_50-2_1layer_0.5alpha_0dropout_LogSpaceMaxTimesSemiring_200epochs_inhousemort_deepsuperv" --gpu="2"

1.4 Logistic Regression (This is from Harutyunyan's code)
python -um mimic3models.in_hospital_mortality.logistic.main --l2 --C 0.001 --output_dir mimic3models/in_hospital_mortality/logistic

2. Decompensation Task
#e.g. /path/to/data/='./mimic3benchmark/decompensation/'

2.1 Adacare
python -u train_adacare_decomp.py --test_mode=0 --rnn_dim=128 --batch_size=128 --data_path='/path/to/data/' --gpu='5' --file_name='210418_Adacare_Decomp_Retraining_128batch_128rnndim'

2.2 RETAIN
python -u 210310_15_25_RETAIN_v2_Decompensation_noemb_test.py --batch_size=200 --clip=0.0 --data='/path/to/data/' --deep_supervision=False --dim_alpha=300 --dim_beta=300 --dropout=0.0 --dropout_context=0.1 --dropout_emb=0.1 --emb_dim=76 --epochs=100 --file_name='210310_21_06_RETAIN_v2_Decomp_300alpha_300beta' --gpu='5' --lr=0.001 --num_features=76 --output_dir='.' --timestep=1.0 --train_nbatches=200 --val_nbatches=200

2.3 SoPa
python -u 201202_02_47_Train_SoPa_MLP_Decomp_Deepsuperv_LogSpaceMaxTimesSemiring.py --epochs=100 --pattern_specs='3-30_4-30_5-30' --mlp_hidden_dim=0 --num_mlp_layers=1 --dropout=0 --batch_size=50 --file_name="210110_17_19_3-30_4-30_5-30_1layer_0dropout_100epochs_decomp_LogSpaceMaxTimesSemiring" --gpu="1" 

2.4 Logistic Regression (This is from Harutyunyan's code)
python -um mimic3models.decompensation.logistic.main --output_dir mimic3models/decompensation/logistic


