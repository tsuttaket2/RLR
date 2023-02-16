import argparse, time, pickle, math, itertools
import numpy as np
import xgboost, sklearn
from sklearn import linear_model
from resources.utils.utils import split_by_comma, time_str, pprint
from resources.utils.metrics import get_metrics_binary
import gzip #Thiti
from io import StringIO #Thiti
XGB_EARLY_STOP_ROUNDS = 10

#parameters for xgboost
XGB_PARAMS = { 'colsample_bytree': 0.6, #0.6, 1.0
               'max_depth': 6, #6
               'min_child_weight': 1, #1
               'gamma': 5, #5, 0
               'reg_alpha': 0, #50, 0
               'reg_lambda': 1, #1.0
               'n_estimators': 100, #100
               'nthread': 10} #10

#parameters and their values to grid search over
#XGB_PARAMS_SEARCH = [('n_estimators', [1, 5, 10, 100, 200]), ('reg_alpha', [0,2])] #for testing
#XGB_PARAMS_SEARCH = [('n_estimators', [1, 5, 10, 100, 200])] #for testing
XGB_PARAMS_SEARCH = [('n_estimators', [50, 100, 200, 300])]

#parameters for xgboost
LR_PARAMS = { 'C': 1, #1
              'penalty': 'elasticnet', #'elasticnet', 'l1', 'l2'
              'class_weight': 'balanced', #'balanced', None
              'solver': 'saga', #'saga' for elasticnet, ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’
              'max_iter': 5000, # 100
              'n_jobs': 10, #10
              'l1_ratio': 1.0 } #'1.0' is l1, '0.0' is l2
LR_PARAMS_SEARCH = [('C', [0.1, 1, 10, 100])]

Y_LABEL_HEADER = 'Y_LABEL' #y-label must be last column #for debugging: 'Col8'


def get_searched_params_as_str(model_name, params):
    if model_name == 'xgb':
        params_search = XGB_PARAMS_SEARCH
    else:
        assert model_name == 'lr'
        params_search = LR_PARAMS_SEARCH

    return '-'.join([f'{p}_{params[p]}' for p,_ in params_search])

def get_header(filename):
    if filename.find(".Z")!=-1: fin = gzip.open(filename, 'rt')    #Thiti
    else:                              fin = open(filename)               #Thiti
    header_line = fin.readline().strip()
    header = split_by_comma(header_line)
    fin.close()   #Thiti
    assert header[-1] == Y_LABEL_HEADER, f"expect last col to be {Y_LABEL_HEADER}, instead got {header[-1]}"
    return header[:-1]

def get_result_str(res):
    return f"auprc: {res['auprc']:.6f}, auroc: {res['auroc']:.6f}, minpse: {res['minpse']:.6f}, acc: {res['acc']:.6f}, " \
           f"prec0: {res['prec0']:.6f}, prec1: {res['prec1']:.6f}, rec0: {res['rec0']:.6f}, rec1: {res['rec1']:.6f}, " \
           f"f1_score: {res['f1_score']:.6f}"

def get_examples_and_labels(data_filename):
    #Thiti
    ############################################
    if data_filename.find(".Z")!=-1: fin = gzip.open(data_filename, 'rt')    #Thiti
    else:                              fin = open(data_filename)               #Thiti
    #Header
    header = fin.readline()
    #data part
    dataset = fin.read()
    dataset = np.loadtxt(StringIO(dataset), delimiter=",") 
    fin.close()
    #dataset = np.loadtxt(data_filename, delimiter=",", skiprows=1)  # skip header
    ############################################
    num_cols = dataset.shape[1]
    X = dataset[:, 0:(num_cols-1)] #last column is the label
    y = dataset[:,   (num_cols-1)]
    y= y.astype(int)
    return X, y

def save_model(model, out_model_filename):
    with open(out_model_filename, 'wb') as fout:
        pickle.dump(model, fout)

def load_model(in_model_filename):
    with open(in_model_filename, 'rb') as fin:
        model = pickle.load(fin)
    return model

def train_xgb(X, y, valid_X, valid_y, params):
    model = xgboost.XGBClassifier(use_label_encoder=False,
                                  verbosity=0,
                                  nthread = params['nthread'],
                                  colsample_bytree=params['colsample_bytree'],
                                  max_depth=params['max_depth'],
                                  min_child_weight=params['min_child_weight'],
                                  gamma=params['gamma'],
                                  reg_alpha=params['reg_alpha'],
                                  reg_lambda=params['reg_lambda'],
                                  n_estimators=params['n_estimators'])
    model.fit(X, y, eval_set=[(valid_X, valid_y)], early_stopping_rounds=XGB_EARLY_STOP_ROUNDS, verbose=False)
    return model

def train_lr(X, y, params):
    model = linear_model.LogisticRegression(C=params['C'],
                                                  penalty=params['penalty'],
                                                  class_weight=params['class_weight'],
                                                  solver=params['solver'],
                                                  max_iter=params['max_iter'],
                                                  n_jobs=params['n_jobs'],
                                                  l1_ratio=params['l1_ratio'])
    model.fit(X,y)
    return model

def train(model_name, X, y, valid_X, valid_y, params):
    if model_name == 'xgb': return train_xgb(X,y,valid_X,valid_y, params)
    assert model_name == 'lr'
    return train_lr(X,y,params)

def eval(model, X, y):
    y_pred_probs = model.predict_proba(X) # y_pred_probs: [[prob0, prob1],[prob0, prob1]]
    classes = model.classes_
    assert classes[0] == 0 and classes[1], f"expect: classes [0,1], but got [{classes[0]},{classes[1]}]"
    metrics = get_metrics_binary(y, y_pred_probs, verbose=False)
    return metrics

def get_params_to_search(model_name):
    if model_name == 'xgb':
        params = XGB_PARAMS.copy()
        params_search = XGB_PARAMS_SEARCH
    else:
        assert model_name == 'lr'
        params = LR_PARAMS.copy()
        params_search = LR_PARAMS_SEARCH


    all_lists = []
    for param, values in params_search:
        param_value_pairs = [(param, v) for v in values]
        all_lists.append(param_value_pairs)

    for param_value_pairs in itertools.product(*all_lists):
        for p,v in param_value_pairs:
            params[p] = v
        yield params

def get_feature_importance(model_name, model):
    if model_name == 'xgb': return list(model.feature_importances_)
    assert model_name == 'lr'
    return model.coef_[0]

def run(model_name, train_filename, valid_filename, test_filename, out_model_name, out_feature_importance_filename):
    train_X, train_y = get_examples_and_labels(train_filename)
    valid_X, valid_y = get_examples_and_labels(valid_filename)
    test_X,  test_y  = get_examples_and_labels(test_filename)
    pprint(f'{train_filename} has {train_X.shape[0]} examples, and {train_X.shape[1]} columns')
    pprint(f'{valid_filename} has {valid_X.shape[0]} examples, and {valid_X.shape[1]} columns')
    pprint(f'{test_filename} has {test_X.shape[0]} examples, and {test_X.shape[1]} columns')

    # #debugging
    # model = load_model('xgbmodel-n_estimators_100')
    # model = load_model('lr_model-C_1') #0.582655
    # print(eval(model, test_X, test_y))
    # return

    pprint('param search and training')
    ssec = time.perf_counter()
    list_of_params = get_params_to_search(model_name)
    best_model = None
    best_score = -math.inf
    best_param_str = None
    for params in list_of_params:
        ssec2 = time.perf_counter()
        model = train(model_name, train_X, train_y, valid_X, valid_y, params)
        pprint(f'\ttraining took {time_str(time.perf_counter() - ssec2)}')
        valid_res = eval(model, valid_X, valid_y)
        test_res  = eval(model,  test_X,  test_y)
        params_str = get_searched_params_as_str(model_name, params)
        pprint(f"\tparams: {params_str}, valid_auprc: {valid_res['auprc']}, "
               f"test_auprc: {test_res['auprc']}")
        if valid_res['auprc'] > best_score:
            best_score = valid_res['auprc']
            best_model = model
            best_params_str = params_str
    pprint(f'param search and training took {time_str(time.perf_counter() - ssec)}')

    out_fn = out_model_name+"-"+best_params_str
    pprint(f"saving best model as {out_fn}")
    save_model(best_model,out_fn)

    ssec = time.perf_counter()
    result = eval(best_model, test_X, test_y)
    pprint(f'\ton test set: {get_result_str(result)}')
    pprint(f'evaluation took {time_str(time.perf_counter() - ssec)}')

    feature_imp = get_feature_importance(model_name, best_model)
    features = get_header(train_filename)
    pprint(f'#features: {len(features)}')
    assert len(features) == len(feature_imp), f'expect same #features=#feature_imp, {len(features)}, {len(feature_imp)}'
    feature_imp_pairs = list(zip(features,feature_imp))
    feature_imp_pairs.sort(key=lambda e: -abs(e[1]))

    out_fn = out_feature_importance_filename+"-"+best_params_str
    pprint(f"saving features (ranked by importance) in {out_fn}")
    with open(out_fn, 'w') as fout:
        for f, i in feature_imp_pairs:
            fout.write(f'{i:.6f},{f}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help="'lr' or 'xgb'")
    parser.add_argument('--train_filename', type=str, help='training examples e.g., /home/skok/static_foot.train')
    parser.add_argument('--valid_filename', type=str, help='validation examples, e.g., /home/skok/static_foot.valid')
    parser.add_argument('--test_filename',  type=str, help='test examples, e.g., /home/skok/static_foot.test')
    parser.add_argument('--out_model_filename', type=str, help='file for storing learned model, e.g., /home/skok/xgb')
    parser.add_argument('--out_feat_filename', type=str, help='file for saving important features, e.g., /home/skok/features.txt')
    args = parser.parse_args()
    pprint(f'{args}')

    # #for debugging
    # args.train_filename = 'train-pima.csv'
    # args.valid_filename = 'valid-pima.csv'
    # args.test_filename  = 'test-pima.csv'
    # model = 'lr' #'lr', 'xgb'
    # args.model_name     = f'{model}' #lr, 'xgb'
    # args.out_model_filename = f'{model}_model' #'lr_model', 'xgb_model'
    # args.out_feat_filename  = f'{model}_impt_feat.txt' #'lr_impt_feat.txt', 'xgb_impt_feat.txt'
    # Y_LABEL_HEADER = 'Col8'

    assert args.model_name == 'lr' or args.model_name == 'xgb'
    ssec = time.perf_counter()
    run(args.model_name, args.train_filename, args.valid_filename, args.test_filename, args.out_model_filename, args.out_feat_filename)
    pprint(f'TOTAL TIME: {time_str(time.perf_counter() - ssec)}')

