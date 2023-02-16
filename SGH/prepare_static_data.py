import argparse, os, time, json, math, gzip
from resources.utils.utils import time_str, split_by_comma, split_by_delimiter, get_values_to_idx_map
from resources.utils.utils import get_header_to_idx_map, get_start_end_indices, write_list_as_csv, pprint
from multiprocessing import Process

MULTIPLE_VALUES_DELIMITER = '|'
END_DATE_TRAILING_FLAG = "_0"

class FeatureInfo:
    def __init__(self, id_to_feature, feature_to_id, is_categorical_feature, possible_values, feature_set,
                 num_out_features, out_begin_pos, num_out_features2):
        self.id_to_feature          = id_to_feature
        self.feature_to_id          = feature_to_id
        self.is_categorical_feature = is_categorical_feature
        self.possible_values        = possible_values
        self.feature_set            = feature_set
        self.num_out_features       = num_out_features
        self.out_begin_pos          = out_begin_pos
        self.num_out_features2      = num_out_features2
        self.days_idx = -1

class StatsInfo:
    def __init__(self):
        self._total  = 0.0 #sum of X
        self._total2 = 0.0 #sum of X^2
        self._count = 0
        self._min =  math.inf
        self._max = -math.inf

    def add_new_value(self, v):
        #if type(v) != float: pprint(f"bad: {v}")
        self._total  += v
        self._total2 += v*v
        self._count  += 1
        if v < self._min: self._min = v
        if v > self._max: self._max = v

    def mean(self):
        if self._count == 0: return 0.0
        return self._total / self._count
    def std(self):
        if self._count == 0: return 0.0
        return math.sqrt(self._total2 / self._count - self.mean()**2)
    def min(self):
        if self._count == 0: return 0.0
        return self._min
    def max(self):
        if self._count == 0: return 0.0
        return self._max
    def get_stats(self): return [self.mean(), self.std(), self.min(), self.max()]

    @staticmethod
    def get_feature_names(feature_name):
        return [feature_name+"_MEAN", feature_name+"_STD", feature_name+"_MIN", feature_name+"_MAX"]

    @staticmethod
    def get_number_stats(): return 4

def compute_length_and_begin_end_pos_helper(id_to_feature, is_categorical_feature, possible_values,
                                            num_cells_for_non_categorical):
    num_features = len(id_to_feature)
    cur_len = 0
    begin_pos = [0] * num_features
    end_pos   = [0] * num_features
    for i in range(num_features):
        feature = id_to_feature[i]
        begin_pos[i] = cur_len
        if is_categorical_feature[feature]:
            end_pos[i] = begin_pos[i] + len(possible_values[feature])
        else:
            end_pos[i] = begin_pos[i] + num_cells_for_non_categorical
        cur_len = end_pos[i]
    return cur_len, begin_pos, end_pos

def compute_length_and_begin_end_pos(id_to_feature, is_categorical_feature, possible_values):
    return compute_length_and_begin_end_pos_helper(id_to_feature, is_categorical_feature, possible_values, 1)

def compute_length_and_begin_end_pos_with_stats(id_to_feature, is_categorical_feature, possible_values):
    return compute_length_and_begin_end_pos_helper(id_to_feature, is_categorical_feature, possible_values,
                                                   StatsInfo.get_number_stats())

def create_out_header(id_to_feature, is_categorical_feature, possible_values):
    length, begin_pos, end_pos = compute_length_and_begin_end_pos_with_stats(id_to_feature, is_categorical_feature,
                                                                             possible_values)
    assert len(begin_pos) == len(end_pos) == len(id_to_feature)

    ##debug
    # num_items = 0
    # for i in range(len(id_to_feature)):
    #     feature = id_to_feature[i]
    #     if is_categorical_feature[feature]: num_items += len(possible_values[feature])
    #     else:                               num_items += 4
    # assert num_items == length

    out_header = [''] * (length + 1)  # +1 is for y label, len(out_header) == 57265

    for i in range(len(id_to_feature)):
        feature = id_to_feature[i]
        sidx = begin_pos[i]
        eidx = end_pos[i]
        if is_categorical_feature[feature]:
            assert eidx-sidx == len(possible_values[feature])
            for value, idx in possible_values[feature].items():
                feature_name = feature+"_"+value
                if ',' in feature_name:
                    feature_name = f'"{feature_name}"'
                out_header[sidx+idx] = feature_name
        else:
            assert eidx-sidx == StatsInfo.get_number_stats()
            feature_names = StatsInfo.get_feature_names(feature)
            assert len(feature_names) == StatsInfo.get_number_stats()
            for j in range(len(feature_names)):
                out_header[sidx+j] = feature_names[j]

    #for i in range(len(out_header)-100,len(out_header)): pprint(f'{i}: {out_header[i]}') #debug
    out_header[-1] = "Y_LABEL"
    #for i in range(len(out_header)-100,len(out_header)): pprint(f'{i}: {out_header[i]}') #debug
    return out_header

def create_stats_list(feature_info):
    num_out_features       = feature_info.num_out_features
    begin_pos              = feature_info.out_begin_pos
    id_to_feature          = feature_info.id_to_feature
    is_categorical_feature = feature_info.is_categorical_feature

    stats_list = [0] * num_out_features
    for i in range(len(id_to_feature)):
        feature = id_to_feature[i]
        if not is_categorical_feature[feature]:
            bpos = begin_pos[i]
            stats_list[bpos] = StatsInfo()
    return stats_list

def update_stats(out_features, stats_list):
    assert len(out_features) == len(stats_list)
    for i in range(len(stats_list)):
        if type(stats_list[i]) == StatsInfo: #this feature is non-categorical feature
            stats = stats_list[i]
            stats.add_new_value(out_features[i])
        else:
            #is categorical feature
            assert type(stats_list[i]) == int or type(stats_list[i]) == float, f'type(stats_list[i])={type(stats_list[i])}'
            v = out_features[i]
            if stats_list[i] == 0:
                stats_list[i] = v
            else:
                assert stats_list[i] == 1

def compute_stats(stats_list, num_out_features2):
    final_stats = [None] * (num_out_features2+1) #+1 for the label
    out_idx = 0
    for i in range(len(stats_list)):
        if type(stats_list[i]) == StatsInfo: #if non-categorical value
            statistics = stats_list[i].get_stats()
            for s in statistics:
                final_stats[out_idx] = s
                out_idx += 1
        else:
            #is categorical value
            assert type(stats_list[i]) == int or type(stats_list[i]) == float
            final_stats[out_idx] = stats_list[i]
            out_idx += 1
    assert out_idx == num_out_features2
    return final_stats

def populate_feature(out_features, feature, value, feature_info):
    feature_to_id          = feature_info.feature_to_id
    is_categorical_feature = feature_info.is_categorical_feature
    possible_values        = feature_info.possible_values
    begin_pos              = feature_info.out_begin_pos

    #handle prescription end date
    def get_category_ids():
        value_to_id_map = possible_values[feature]
        cat_id = value_to_id_map.get(value)
        if cat_id is not None: return [cat_id]
        values = split_by_delimiter(value, MULTIPLE_VALUES_DELIMITER)
        cat_ids = []      #categories to set to 1
        cat_ids_zero = [] #categories to set to 0
        for i,v in enumerate(values):
            cid = value_to_id_map.get(v)
            if cid is not None:
                cat_ids.append(cid)
            else: # this is categorical value that must be set to 0
                idx = v.rfind(END_DATE_TRAILING_FLAG)
                assert idx >= 0, f'bad feature value {v}'
                v = v[:idx]
                cid = value_to_id_map.get(v)
                assert cid is not None, f'value {v} is not found for {feature}'
                cat_ids_zero.append(cid)
        return cat_ids, cat_ids_zero

    # def get_category_ids():
    #     value_to_id_map = possible_values[feature]
    #     cat_id = value_to_id_map.get(value)
    #     if cat_id is not None: return [cat_id]
    #     values = split_by_delimiter(value, MULTIPLE_VALUES_DELIMITER)
    #     cat_ids = [-1]*len(values)
    #     for i,v in enumerate(values):
    #         cid = value_to_id_map.get(v)
    #         assert cid is not None, f'value {v} is not found for {feature}'
    #         cat_ids[i] = cid
    #     return cat_ids

    def convert_to_float(sstr):
        try:
            v = float(sstr)
            return v
        except ValueError: #handle multiple values for numerical feature
            str_list = split_by_delimiter(sstr, MULTIPLE_VALUES_DELIMITER)
            total = 0.0
            for s in str_list:
                total += float(s)
            v = total / len(str_list)
            return v

    feature_id = feature_to_id[feature]
    if is_categorical_feature[feature]:
        num_values  = len(possible_values[feature])
        one_hot     = [0] * num_values #np.zeros((num_values,))
        category_ids, category_ids_zero = get_category_ids()
        for cid in category_ids: #handle multiple values for categorical feature
            one_hot[cid] = 1
        for cid in category_ids_zero: #handle prescription end date, set categorical value to 0
            one_hot[cid] = 0

        s_idx = begin_pos[feature_id]
        e_idx = s_idx + num_values
        out_features[s_idx:e_idx] = one_hot
    else:
        out_features[begin_pos[feature_id]] = convert_to_float(value)

def create_features(in_feature_list, in_header, feature_info):
    feature_set      = feature_info.feature_set
    num_out_features = feature_info.num_out_features

    assert len(in_feature_list) == len(in_header)
    out_features = [0] * num_out_features

    for i in range(len(in_feature_list)):
        feature = in_header[i]
        value   = in_feature_list[i]
        if value == '' or feature not in feature_set: continue
        populate_feature(out_features, feature, value, feature_info)

    return out_features

def process_subject_csv_file(csv_filename, y_label, feature_info, max_past_years):
    #create list containing StatsInfo for non-categorical features and 0 for categorical features
    stats_list = create_stats_list(feature_info)

    def process_line(line, in_header):
        line = line.strip()
        in_features = split_by_comma(line)
        out_features = create_features(in_features, in_header, feature_info)
        update_stats(out_features, stats_list)

    if max_past_years > 0:
        with open(csv_filename) as fin:
            lines = fin.readlines()
        in_header = split_by_comma(lines[0])
        if feature_info.days_idx < 0:
            header_to_index_map = get_header_to_idx_map(in_header)
            feature_info.days_idx = header_to_index_map['DAYS']
        days_idx = feature_info.days_idx
        last_line = lines[-1]
        last_day = split_by_comma(last_line)[days_idx]
        earliest_day = int(last_day) - max_past_years*365

        first_idx = 0
        lines = lines[1:]
        for i in range(len(lines)):
            line = lines[i]
            day = split_by_comma(line)[days_idx]
            if int(day) >= earliest_day:
                first_idx = i
                break
        lines = lines[first_idx:]

        for line in lines:
            process_line(line, in_header)
    else:
        with open(csv_filename) as fin:
            line = fin.readline()
            cnt = 1
            in_header = split_by_comma(line)
            while line:
                if cnt > 1:
                    process_line(line, in_header)
                line = fin.readline()
                cnt += 1

    final_stats = compute_stats(stats_list, feature_info.num_out_features2)
    final_stats[-1] = y_label
    return final_stats

def get_csv_filename_ylabel_pairs(list_filename):
    with open(list_filename) as fin:
        lines = fin.readlines()
    lines = lines[1:] #ignore header
    file_ylabel_pairs = []
    for line in lines:
        in_subject_csv_filename, _, y_label = split_by_comma(line)
        file_ylabel_pairs.append( (in_subject_csv_filename, y_label) )
    return file_ylabel_pairs

def sample_neg_examples(file_ylabel_pairs, frac_neg_keep, pid):
    assert 0 <= frac_neg_keep <= 1.0
    num_neg = 0
    for _, ylabel in file_ylabel_pairs:
        if ylabel == '0': num_neg += 1
    num_neg_keep = int(frac_neg_keep * num_neg)

    cnt = 0
    last_idx = len(file_ylabel_pairs)
    for i, (_, ylabel) in enumerate(file_ylabel_pairs):
        if ylabel == '0':
            cnt += 1
            if cnt >= num_neg_keep:
                last_idx = i
                break
    pprint(f'{pid}: keep {cnt} out of {num_neg} examples; {last_idx} examples (+ve & -ve)  kept')
    return file_ylabel_pairs[:last_idx]

def prepare_data(proc_id, num_proc, in_list_filepath, in_csv_dir, saved_filename, config_path,
                 max_past_years, max_csv_files, gzip_files, frac_neg_examples):
    start_sec = time.perf_counter()

    with open(config_path) as fin:
        config = json.load(fin)
    id_to_feature          = config['id_to_channel']
    is_categorical_feature = config['is_categorical_channel']
    possible_values        = config['possible_values']
    feature_set            = set(id_to_feature)
    possible_values = dict([(feature, get_values_to_idx_map(value_list)) for feature, value_list in possible_values.items()])
    feature_to_id   = dict(zip(id_to_feature, range(len(id_to_feature))))

    #compute these once here rather than repeatedly later
    #num of out features after representing categorical feature as one-hot
    ssec = time.perf_counter()
    num_out_features, out_begin_pos, _ = compute_length_and_begin_end_pos(id_to_feature, is_categorical_feature,
                                                                          possible_values)
    #num of out features after representing categorical feature as one-hot
    #  and using one or more statistics for non-categorical features
    num_out_features2, _, _ = compute_length_and_begin_end_pos_with_stats(id_to_feature, is_categorical_feature,
                                                                          possible_values)
    pprint(f'{proc_id}: compute_length_and_begin_end_pos took {time_str(time.perf_counter() - ssec)}')

    feature_info = FeatureInfo(id_to_feature, feature_to_id, is_categorical_feature, possible_values, feature_set,
                               num_out_features, out_begin_pos, num_out_features2)

    file_ylabel_pairs = get_csv_filename_ylabel_pairs(in_list_filepath)
    file_ylabel_pairs.sort(key=lambda e: e[0])
    if 0 <= frac_neg_examples < 1.0:
        file_ylabel_pairs = sample_neg_examples(file_ylabel_pairs, frac_neg_examples, proc_id)

    sidx, eidx = get_start_end_indices(len(file_ylabel_pairs), num_proc, proc_id)
    sub_file_ylabel_pairs = file_ylabel_pairs[sidx:eidx]
    pprint(f'proc {proc_id}: processing sub_file_ylabel_pairs[{sidx}:{eidx}]')
    if 0 <= max_csv_files < len(sub_file_ylabel_pairs):
        sub_file_ylabel_pairs = sub_file_ylabel_pairs[:max_csv_files]
        pprint(f'proc {proc_id}: limit to {len(sub_file_ylabel_pairs)} csv files')

    if gzip_files: fout = gzip.open(saved_filename+f'-{proc_id}.Z', 'wt') #write gzip file; can view file with zcat <filename> in terminal
    else:          fout =      open(saved_filename+f'-{proc_id}',   'w')  #use this to write to uncompressed file

    ssec = time.perf_counter()
    out_header = create_out_header(id_to_feature, is_categorical_feature, possible_values)
    pprint(f'{proc_id}: create_out_header took {time_str(time.perf_counter() - ssec)}')

    write_list_as_csv(out_header, fout)

    ssec = time.perf_counter()
    cnt = 0
    for in_subject_csv_filename, y_label in sub_file_ylabel_pairs:

        if cnt % 1000 == 0:
            pprint(f'{proc_id}: processing {cnt} csv files took {time_str(time.perf_counter() - ssec)}')

        in_csv_fn = os.path.join(in_csv_dir, in_subject_csv_filename)
        row = process_subject_csv_file(in_csv_fn, y_label, feature_info, max_past_years)
        write_list_as_csv(row, fout)  # y_label is last value in row
        cnt += 1

    fout.close()
    pprint(f'proc {proc_id}: processing csv files took {time_str(time.perf_counter() - start_sec)}')

def merge_files(num_processes, gzip_files, saved_filename):
    out_header_str = None
    if gzip_files: fout = gzip.open(saved_filename+".Z", "wt")  # write gzip file; can view file with zcat <filename> in terminal
    else:          fout =      open(saved_filename , 'w')       # use this to write to uncompressed file
    for pid in range(num_processes):
        if gzip_files: in_fn = saved_filename+f'-{pid}.Z'
        else:          in_fn = saved_filename+f'-{pid}'
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

def multiprocess(num_processes, list_filepath, in_csv_dir, saved_filename, config_path, max_past_years, max_csv_files,
                 gzip_files, frac_neg_examples):
    jobs = []
    for pid in range(_num_processes):
        p = Process(target=prepare_data,
                    args=(pid, num_processes, list_filepath, in_csv_dir, saved_filename, config_path,
                          max_past_years, max_csv_files, gzip_files, frac_neg_examples))
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_csv_dir',      type=str, help='directory containing all subject.csv files', default='.')
    parser.add_argument('--list_filepath',        type=str, help='listfile name, e.g., /home/skok/train_listfile.csv',default='.')
    parser.add_argument('--saved_filename',       type=str, help='saved data file name, e.g., /home/skok/static_foot.train')
    parser.add_argument('--discretizer_filename', type=str, help='full path to discretizer json, e.g., /home/skok/discretizer_config.json')
    parser.add_argument('--max_past_years',       type=int, help='look back this number of years at most; negative value means use all data', default=-1)
    parser.add_argument('--max_csv_files',        type=int, help='max number of csv files to process; negative value means process all', default=-1)
    parser.add_argument('--number_processes', '-p', type=int, help='', default=10)
    parser.add_argument('--gzip_files',             type=int, help='0/1 gzip output files', default=1)
    parser.add_argument('--frac_neg_examples',    type=float, help='fraction of negative examples to use; negative means use all', default=-1)
    args = parser.parse_args()
    pprint(f'{args}')

    #example values
    #--data='output_from_mortality_folder'
    #--list_filepath='./mortality/test_listfile-1.csv'
    #--saved_filename='test_listfile-1'
    #--read_chunk_size=100
    #--n_examples=10000
    #--saved_folder='./Test_Mortality_Static_Data/' &

    # #for debugging
    # dirr='/Users/stanleykok/PycharmProjects/empower-dataprep-thiti'
    # args.subject_csv_dir = dirr+'/mortality_test/test'
    # args.list_filepath   = dirr+'/mortality_test/test_listfile.csv'
    # args.saved_filename  = 'test_listfile-1'
    # args.resources_folder= dirr+'/resources/discretizer_config.json'
    # args.max_past_years  = 5
    # args.max_csv_files   = 10
    # args.num_processes   = 2
    # args.gzip_files      = 0
    # args.frac_neg_examples = -1

    _in_csv_dir      = args.subject_csv_dir
    _list_filepath   = args.list_filepath
    _saved_filename  = args.saved_filename
    _config_path     = args.discretizer_filename #'/dir/discretizer_config.json'
    _max_past_years  = args.max_past_years
    _max_csv_files   = args.max_csv_files
    _num_processes   = args.number_processes
    _gzip_files      = bool(args.gzip_files)
    _frac_neg_examples = args.frac_neg_examples

    begin_sec = time.perf_counter()

    pprint('multiprocessing csv files')
    start_sec3 = time.perf_counter()
    multiprocess(_num_processes, _list_filepath, _in_csv_dir, _saved_filename, _config_path, _max_past_years,
                 _max_csv_files, _gzip_files, _frac_neg_examples)
    pprint(f'multiprocessing ALL csv files took {time_str(time.perf_counter() - start_sec3)}')

    pprint('merging files')
    start_sec3 = time.perf_counter()
    merge_files(_num_processes, _gzip_files, _saved_filename)
    pprint(f'merging files to create {_saved_filename} took {time_str(time.perf_counter() - start_sec3)}')
    pprint(f'TOTAL TIME TAKEN by prepareStaticData_Mortality.py: {time_str(time.perf_counter() - begin_sec)}')
