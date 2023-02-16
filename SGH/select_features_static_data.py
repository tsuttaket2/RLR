import argparse, gzip, json, math, time
import numpy as np
from resources.utils.utils import get_values_to_idx_map, split_by_comma, get_start_end_indices, pprint, time_str
from prepare_static_data import compute_length_and_begin_end_pos_with_stats, create_out_header
from multiprocessing import Process, Manager
from io import StringIO
import os
TARGET_FEATURE_NAMES = ['CODES_SNOMED_Diagnosis codes',    # 23220 values
                        'VISITS_Admissions',               # 16924 values
                        'VISITS_Admissions_CGH',           # 10177 values
                        'VISITS_Emergency department',     #  2809 values
                        'VISITS_Emergency department_CGH', #  2185 values
                        'VISITS_Outpatient visits',        #   551 values
                        'CODES_SHPWKC_AC_Diagnosis codes'] #   435 values, total = 56301

NUM_COLS_TO_KEEP = [1000, 2500, 5000, 7500, 10000, 20000, 30000, 40000]


class CorrInfo:
    def __init__(self, target_col_name, target_col_idx):
        self.target_col_name = target_col_name
        self.target_col_idx  = target_col_idx,
        self.x   = 0.0
        self.y   = 0.0
        self.x2  = 0.0 #x*x
        self.y2  = 0.0 #y*y
        self.xy  = 0.0 #x*y
        self.cnt = 0

    def add_value(self, x, y):
        self.x   += x
        self.y   += y
        self.x2  += x*x
        self.y2  += y*y
        self.xy  += x*y
        self.cnt += 1

    def correlation(self):
        if self.cnt == 0: return 0
        numer = self.cnt * self.xy - self.x * self.y
        denom = np.sqrt(self.cnt*self.x2 - self.x**2) * np.sqrt(self.cnt*self.y2 - self.y**2)
        if denom == 0.0: return math.inf
        return numer/denom


def get_target_header_names_and_indices(config_filename):
    with open(config_filename) as fin:
        config = json.load(fin)
    id_to_feature          = config['id_to_channel']
    is_categorical_feature = config['is_categorical_channel']
    possible_values        = config['possible_values']
    possible_values = dict([(feature, get_values_to_idx_map(value_list)) for feature, value_list in possible_values.items()])
    feature_to_id   = get_values_to_idx_map(id_to_feature)

    out_header = create_out_header(id_to_feature, is_categorical_feature, possible_values)
    _, begin_pos, end_pos = compute_length_and_begin_end_pos_with_stats(id_to_feature, is_categorical_feature,
                                                                       possible_values)
    assert len(out_header) == end_pos[-1]+1, f'expect same length for out_header, begin_pos, end_pos;'\
                                                              f' {len(out_header)}, {end_pos[-1]+1}'
    target_features = TARGET_FEATURE_NAMES
    target_indices = []
    target_header  = []
    for f in target_features:
        idd  = feature_to_id[f]
        bpos = begin_pos[idd]
        epos = end_pos[idd]
        target_indices.extend(list(range(bpos,epos)))
        target_header.extend(out_header[bpos:epos])
    assert len(target_header) == len(target_indices), f'expect len(target_header) == len(target_indices), ' \
                                                      f'{len(target_header)}, {len(target_indices)}'
    return target_header, target_indices

def compute_correlations(proc_id, num_proc, data_filename, gzip_files, shared_correlation_list):
    ssec = time.perf_counter()
    sidx, eidx = get_start_end_indices(len(shared_correlation_list), num_proc, proc_id)

    #shared_correlation_list: list of (correlation, col_name, col_index) triples
    corr_info_list = [ CorrInfo(shared_correlation_list[i][1], shared_correlation_list[i][2]) for i in range(sidx, eidx) ]

    if gzip_files: fin = gzip.open(data_filename, 'rt')
    else:          fin =      open(data_filename)
    out_header = None
    line = fin.readline()
    cnt = 1
    while line:
        if cnt == 1: #header
            out_header = split_by_comma(line.strip())
            assert out_header[-1] == 'Y_LABEL'
        else:
            cols = split_by_comma(line.strip())
            assert len(out_header) == len(cols)
            for corr_info in corr_info_list:
                col_name = corr_info.target_col_name
                col_idx  = corr_info.target_col_idx[0]
                assert out_header[col_idx] == col_name
                col_val = float(cols[col_idx])
                y_label = float(cols[-1])
                corr_info.add_value(col_val, y_label)
        line = fin.readline()
        cnt += 1
    fin.close()

    cnt = 0
    for corr_info in corr_info_list:
        correlation, col_name, col_idx = shared_correlation_list[sidx+cnt]
        assert col_name == corr_info.target_col_name, f'expect same col name, {col_name}, {corr_info.target_col_name}'
        assert col_idx  == corr_info.target_col_idx[0],  f'expect same col idx, {col_idx}, {corr_info.target_col_idx[0]}'
        shared_correlation_list[sidx+cnt] = (corr_info.correlation(), col_name, col_idx)
        cnt += 1
    assert cnt == eidx-sidx, f"expect cnt == eidx, {cnt}, {eidx-sidx}"
    pprint(f'{proc_id}: compute_correlations took {time_str(time.perf_counter() - ssec)}')


def create_new_file(data_filename, out_filename, gzip_files, target_indices, keep_indices):
    if gzip_files: fin = gzip.open(data_filename, 'rt')
    else:          fin =      open(data_filename)
    if gzip_files: fout = gzip.open(out_filename, "wt")
    else:          fout =      open(out_filename, 'w')

    def write_row(cur_line):
        cols = split_by_comma(cur_line)
        num_kept_cols = 0
        for i, v in enumerate(cols):
            if i not in target_indices or i in keep_indices:
                if ',' in v:
                    fout.write(f'"{v}"')
                else:
                    fout.write(v)
                if i != len(cols) - 1: fout.write(',')
                else:                  fout.write('\n')
                num_kept_cols += 1
        return num_kept_cols

    #Header
    header = fin.readline()
    write_row(header)
    #data part
    data=fin.read()
    data = np.loadtxt(StringIO(data), delimiter=",") 
    kept_cols=[True if i not in target_indices or i in keep_indices else False for i in range(data.shape[1]) ]
    data=data[:,kept_cols]
    np.savetxt(fout,data,delimiter=",")
    fin.close()
    fout.close()

def merge_files(num_processes, gzip_files, saved_filename,files_to_merge):   #Thiti
    out_header_str = None
    if gzip_files: fout = gzip.open(saved_filename , "wt")  # write gzip file; can view file with zcat <filename> in terminal
    else:          fout =      open(saved_filename , 'w')       # use this to write to uncompressed file
    for in_fn in files_to_merge:
        if not os.path.exists(in_fn) or not os.path.isfile(in_fn):
            pprint(f"ERROR: {in_fn} does not exist or is not a file!")
            continue
        if gzip_files: fin = gzip.open(in_fn, 'rt')
        else:          fin =      open(in_fn)

        line = fin.readline()
        cnt = 1
        while line:
            if cnt == 1:  # header
                if out_header_str is None:
                    out_header_str = line
                    fout.write(out_header_str)
            else:
                fout.write(line)
            line = fin.readline()
            cnt += 1
        fin.close()
    fout.close()
    for fn in files_to_merge:
        os.remove(fn)

def multiproc_write(pid,NUM_COLS_TO_KEEP_pid,corr_list,target_indices, filename_pid,out_suffix,gzip_files):
    num_cols_to_keep = NUM_COLS_TO_KEEP_pid
    if num_cols_to_keep >= len(corr_list):
        pprint(f'num_cols_to_keep {num_cols_to_keep} > number of target columns {len(corr_list)},so keep all cols')
        return
    keep_indices = set(target_indices[:num_cols_to_keep])
    for fn in [filename_pid]:
        create_new_file(fn, fn+f'{out_suffix}{num_cols_to_keep}', gzip_files, target_indices, keep_indices)

def run(num_processes, train_filename, valid_filename, test_filename, out_suffix, out_corr_filename, config_filename,
        gzip_files):

    ssec = time.perf_counter()
    target_col_names, target_col_indices = get_target_header_names_and_indices(config_filename)
    pprint(f'get_target_header_names_and_indices took {time_str(time.perf_counter() - ssec)}')

    pprint("multiprocessing")
    manager = Manager()
    #shared_list: list of (correlation, col_name, col_index) triples
    shared_list = manager.list( [(0.0, target_col_names[i], target_col_indices[i]) for i in range(len(target_col_indices))] )

    jobs = []
    ssec = time.perf_counter()
    for pid in range(num_processes):
        p = Process(target=compute_correlations, args=(pid, num_processes, train_filename, gzip_files, shared_list))
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()
    pprint(f'multiprocessing ALL csv files took {time_str(time.perf_counter() - ssec)}')

    corr_list = list(shared_list) # list of (correlation, col_name, col_index) triples
    corr_list.sort(key=lambda e: -abs(e[0])) #sort in decreasing order of |correlation|; Manager.list can't be sorted

    with open(out_corr_filename, 'w') as fout:
        for correlation, target_name, target_index in corr_list:
            fout.write(f'{correlation},{target_name},{target_index}\n')

    target_indices = [idx for _, _, idx in corr_list]

    ssec = time.perf_counter()
    for NUM_COLS_TO_KEEP_pid in NUM_COLS_TO_KEEP:
        for filename in [train_filename,valid_filename,test_filename]:
            jobs = []
            files_to_merge=[]
            for pid in range(num_processes):
                if gzip_files:
                    filename1,filename2=filename.split('.')
                    filename_pid=filename1+"-"+str(pid)+"."+"Z"
                else:
                    filename_pid= filename+"-"+str(pid)
                files_to_merge.append(filename_pid+f'{out_suffix}{NUM_COLS_TO_KEEP_pid}')
                p = Process(target=multiproc_write, args=(pid,NUM_COLS_TO_KEEP_pid,corr_list,target_indices, filename_pid,out_suffix,gzip_files))
                jobs.append(p)
                p.start()
            for j in jobs:
                j.join()
            if gzip:
                saved_filename = filename.split(".")[0] + f'_{out_suffix}-{NUM_COLS_TO_KEEP_pid}.Z'
            else:
                saved_filename = filename + f'_{out_suffix}-{NUM_COLS_TO_KEEP_pid}'
            merge_files(num_processes, gzip_files, saved_filename,files_to_merge)
    pprint(f'creating new train/valid/test files took {time_str(time.perf_counter() - ssec)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_filename',         type=str, help='training examples e.g., /home/skok/static_foot.train')
    parser.add_argument('--valid_filename',         type=str, help='validation examples, e.g., /home/skok/static_foot.valid')
    parser.add_argument('--test_filename',          type=str, help='test examples, e.g., /home/skok/static_foot.test')
    parser.add_argument('--out_suffix',             type=str, help='suffix appended to output train/valid./test files with columns dropped', default='-topfeat')
    parser.add_argument('--out_corr_filename',       type=str, help='output file to store correlation of columns with label')
    parser.add_argument('--discretizer_filename',   type=str, help='path to discretizer json, e.g., /home/skok/discretizer_config.json')
    parser.add_argument('--gzip_files',             type=int, help='0/1 gzip output files', default=1)
    parser.add_argument('--number_processes', '-p', type=int, help='', default=10)
    args = parser.parse_args()
    pprint(f'{args}')

    run(args.number_processes, args.train_filename, args.valid_filename, args.test_filename, args.out_suffix,
        args.out_corr_filename, args.discretizer_filename, args.gzip_files)




