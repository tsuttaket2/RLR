#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import sys
import os, argparse, random, time, json
import pandas as pd
from resources.utils.utils import read_episode_str_categorical_col2, get_header_to_idx_map, get_values_to_idx_map
from resources.utils.utils import split_by_comma, split_by_delimiter, time_str, get_start_end_indices
from resources.utils.utils import write_as_csv_omit_idx, pprint
from resources.utils.utils import create_directory #Thiti
from multiprocessing import Process
import chardet
MULTIPLE_VALUES_DELIMITER = '|'


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
class ComorbidityInfo:

    #in comorbidity_*.json, only need to modify 'snomed' to 'comorbidity_name'
    def __init__(self, config_file, sgh_path, features_config):
        with open(config_file) as fin:
            config = json.load(fin)
            self.snomed = set(config["snomed"])
            self.shpwkc = set(config["shpwkc"])
            self.icd10  = set(config["icd10"])  #not really needed because only appear in outcomes
            self.all_codes = self.snomed.union(self.shpwkc).union(self.icd10)
            self.features_to_check = set(config["features"])

            #"OUTCOME_AMI", "OUTCOME_STROKE HAEMORRHAGIC", or "OUTCOME_STROKE ISCHAEMIC"
            features_to_add_code = config["features_to_add_code"]
            for f in features_to_add_code:
                codes = features_config['possible_values'][f]
                self.all_codes = self.all_codes.union(codes)

            #in .json cannot be empty; diagnosis code used to determine first day of comorbidity
            self.snomed_filename = ComorbidityInfo.prepend(ComorbidityInfo,sgh_path, config["snomed_filename"])
            self.shpwkc_filename = ComorbidityInfo.prepend(ComorbidityInfo,sgh_path, config["shpwkc_filename"])
            self.adm1_filename   = ComorbidityInfo.prepend(ComorbidityInfo,sgh_path, config["adm1_filename"])
            self.adm2_filename   = ComorbidityInfo.prepend(ComorbidityInfo,sgh_path, config["adm2_filename"])
            self.visit1_filename = ComorbidityInfo.prepend(ComorbidityInfo,sgh_path, config["visit1_filename"])
            self.visit2_filename = ComorbidityInfo.prepend(ComorbidityInfo,sgh_path, config["visit2_filename"])
            #self.visit3_filename = ComorbidityInfo.prepend(sgh_path, config["visit3_filename"])

            #in .json non-empty only if can be used to determine first day of comorbidity
            self.outcome_death_filename      = ComorbidityInfo.prepend(ComorbidityInfo,sgh_path, config["outcome_death_filename"])
            self.outcome_ami_filename        = ComorbidityInfo.prepend(ComorbidityInfo,sgh_path, config["outcome_ami_filename"])
            self.outcome_stroke_hae_filename = ComorbidityInfo.prepend(ComorbidityInfo,sgh_path, config["outcome_stroke_hae_filename"])
            self.outcome_stroke_isc_filename = ComorbidityInfo.prepend(ComorbidityInfo,sgh_path, config["outcome_stroke_isc_filename"])
            self.outcome_cabg_filename       = ComorbidityInfo.prepend(ComorbidityInfo,sgh_path, config["outcome_cabg_filename"])
            self.outcome_pci_filename        = ComorbidityInfo.prepend(ComorbidityInfo,sgh_path, config["outcome_pci_filename"])
            self.outcome_hf_filename         = ComorbidityInfo.prepend(ComorbidityInfo,sgh_path, config["outcome_hf_filename"])

            self.comorbidity_filename = ComorbidityInfo.prepend(ComorbidityInfo,sgh_path, config["comorbidity_filename"])
            self.comorbidity_name     = config["comorbidity_name"]

    @staticmethod
    def prepend(self, pre_path, filename):
        if filename == '': return ''
        return os.path.join(pre_path, filename)


def get_all_pos_patients(in_fn, comorbidity_name): #even those without date of comorbidity
    if in_fn == '' or comorbidity_name == '': return {}
    all_pos_patients=set()   #Thiti
    with open(in_fn,'rb') as fin:
        header_to_idx_map = None
        line = fin.readline()
        detected_encoding= chardet.detect(line)["encoding"] #Thiti
        line = line.decode(detected_encoding) #Thiti
        cnt = 1
        while line:
            if cnt == 1: #header
                header = split_by_comma(line)
                header_to_idx_map = get_header_to_idx_map(header)
            else:
                cols = split_by_comma(line)
                yes_no_comorbidity_to_idx_map = header_to_idx_map[comorbidity_name] #Thiti
                yes_no_comorbidity=cols[yes_no_comorbidity_to_idx_map]   #Thiti
                if yes_no_comorbidity.lower() == 'yes':
                    dcof_idx = header_to_idx_map['DCOF_ID']
                    dcof_id  = cols[dcof_idx].strip()
                    all_pos_patients.add(dcof_id) 
            line = fin.readline()
            if line: 
                detected_encoding= chardet.detect(line)["encoding"] #Thiti
                line = line.decode(detected_encoding) #Thiti
            cnt += 1
    return all_pos_patients

def get_pos_patients_helper(in_fn, subject_to_date_map, codes, date_header_str):
    with open(in_fn,'rb') as fin:    #Thiti:insert 'rb'
        header_to_idx_map = None
        line = fin.readline()
        detected_encoding= chardet.detect(line)["encoding"] #Thiti
        line = line.decode(detected_encoding) #Thiti
        cnt = 1
        while line:
            if cnt == 1: #header
                header = split_by_comma(line)
                header_to_idx_map = get_header_to_idx_map(header)
            else:
                cols = split_by_comma(line)
                if codes is not None:
                    diag_code_idx = header_to_idx_map['DIAGNOSIS_CODE']
                    diag_code = cols[diag_code_idx].strip()
                if codes is None or diag_code in codes:
                    dcof_idx = header_to_idx_map['DCOF_ID']
                    date_idx = header_to_idx_map[date_header_str] #'VISIT_DATE', 'ADMISSION_DATE', 'DATE_OF_VISIT', etc.
                    dcof_id  = cols[dcof_idx].strip()
                    new_date = pd.to_datetime(cols[date_idx])
                    cur_date = subject_to_date_map.get(dcof_id)
                    if cur_date is None or new_date < cur_date:
                        subject_to_date_map[dcof_id] = new_date
            line = fin.readline()
            if line:  #Thiti
                detected_encoding= chardet.detect(line)["encoding"] #Thiti
                line = line.decode(detected_encoding) #Thiti
            cnt += 1

def get_pos_patient_to_date_map(comorb_info):
    subject_to_date_map = {}
    llist = [(comorb_info.snomed_filename, comorb_info.snomed,    'VISIT_DATE'),
             (comorb_info.shpwkc_filename, comorb_info.shpwkc,    'VISIT_DATE'),
             (comorb_info.adm1_filename,   comorb_info.all_codes, 'ADMISSION_DATE'),
             (comorb_info.adm2_filename,   comorb_info.all_codes, 'ADMISSION_DATE'),
             (comorb_info.visit1_filename, comorb_info.all_codes, 'DATE_OF_VISIT'),
             (comorb_info.visit2_filename, comorb_info.all_codes, 'DATE_OF_VISIT'),
             #(comorb_info.visit3_filename, comorb_info.all_codes, 'DATE_OF_VISIT'),
             (comorb_info.outcome_death_filename,      None, 'DEATH_DATE'),
             (comorb_info.outcome_ami_filename,        None, 'ADMISSION_DATE'),
             (comorb_info.outcome_stroke_hae_filename, None, 'ADMISSION_DATE'),
             (comorb_info.outcome_stroke_isc_filename, None, 'ADMISSION_DATE'),
             (comorb_info.outcome_hf_filename,         None, 'ADMISSION_DATE'),
             (comorb_info.outcome_cabg_filename,       None, 'SURGERY_DATE_CABG'),
             (comorb_info.outcome_pci_filename,        None, 'DATE_OF_PCI')]

    for fn, codes, date_name in llist:
        if (fn != ' ') and (fn != ''): get_pos_patients_helper(fn, subject_to_date_map, codes, date_name)

    return subject_to_date_map

def read_episode_str_categorical_col(subject_path, dtype_specs, header, header_to_idx_map):
    try:
        episode = pd.read_csv(os.path.join(subject_path, 'episode.csv'), header=0, index_col=None,
                              error_bad_lines=True, dtype=dtype_specs)
    except:
        eprint(f"Error {subject_path}")
    if header is None: #do this once
        header = episode.columns.values
        header_to_idx_map = get_header_to_idx_map(header)
    rows = episode.values
    charttime_idx = header_to_idx_map['CHARTTIME']
    for r in rows:
        try:
            r[charttime_idx] = pd.to_datetime(r[charttime_idx]) #convert charttime from string to time object
        except:
            print(f"subject_path {subject_path}")
    return header, rows, header_to_idx_map

def split_train_val_test(llist, frac_valid, frac_test):
    num_valid = int(len(llist) * frac_valid)
    num_test  = int(len(llist) * frac_test)
    num_train = len(llist) - num_valid - num_test
    assert num_train+num_valid+num_test == len(llist)
    assert num_train > 0 and num_test > 0
    train = llist[0:num_train]
    valid = llist[num_train:(num_train+num_valid)]
    test = llist[(num_train+num_valid):]
    assert len(train)+len(valid)+len(test) == len(llist)
    return train, valid, test

def multiprocess(number_processes, subjects_root_path, config_path, comorb_info_file, sgh_path, time_before_event,
                 output_path, num_subjects, years_back):
    jobs = []
    for pid in range(number_processes):
        p = Process(target=prepare_subject_csv_files, args=(pid, number_processes, subjects_root_path, config_path,
                                                            comorb_info_file, sgh_path, time_before_event, output_path,
                                                            num_subjects, years_back))
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()

def add_to_list(llist, filename):
    for pid in range(_number_processes):
        fn = os.path.join(_output_path, f'{filename}-{pid}')
        with open(fn) as fin: #no header
            lines = fin.readlines() #no need to strip newline, will write it out later
            llist.extend(lines)

def write_list(llist, filename):
    with open(filename, 'w') as fout:
        fout.write('patient,t,y_true\n')
        for l in llist:
            fout.write(l) #l has newline

def split_triples_train_test_valid():
    pos_lines = []
    neg_lines = []
    add_to_list(pos_lines, 'pos_triples')
    add_to_list(neg_lines, 'neg_triples')
    random.shuffle(pos_lines)
    random.shuffle(neg_lines)
    pos_train, pos_valid, pos_test = split_train_val_test(pos_lines, _frac_validation, _frac_test)
    neg_train, neg_valid, neg_test = split_train_val_test(neg_lines, _frac_validation, _frac_test)
    train_lines = pos_train + neg_train
    valid_lines = pos_valid + neg_valid
    test_lines  = pos_test  + neg_test
    pprint(f'#pos_train, #pos_valid, #pos_test = {pos_train}, {pos_valid}, {pos_test}')
    pprint(f'#neg_train, #neg_valid, #neg_test = {neg_train}, {neg_valid}, {neg_test}')
    pprint(f'#train,     #valid,     #test     = {train_lines}, {valid_lines}, {test_lines}')
    random.shuffle(train_lines)
    random.shuffle(valid_lines)
    random.shuffle(test_lines)
    write_list(train_lines, os.path.join(_output_path, f'train_listfile.csv'))
    write_list(valid_lines, os.path.join(_output_path, f'valid_listfile.csv'))
    write_list(test_lines,  os.path.join(_output_path, f'test_listfile.csv'))

def prepare_subject_csv_files(proc_id, num_proc, patients_root_path, config_path, comorb_info_file, sgh_path,
                              time_before_event, output_path, max_num_patients, years_ago):
    #prepare csv first, split into train test validation later

    def choose_rows_after_n_years_ago(epi_rows, charttime_idx, years_ago):
        date_n_years_ago = epi_rows[-1][charttime_idx] - pd.Timedelta(years_ago * 365, 'D')
        first_idx = 0
        for i in range(len(epi_rows)):
            if epi_rows[i][charttime_idx] >= date_n_years_ago:
                first_idx = i
                break
        return epi_rows[first_idx:]

    def remove_fields(epi_rows, epi_header_to_idx_map, comorb_info):
        for row in epi_rows:
            for feature in comorb_info.features_to_check:
                feat_idx = epi_header_to_idx_map[feature]
                feat_value = row[feat_idx]
                if type(feat_value) == str and len(feat_value) > 0: #diagnosis codes are strings
                    values = split_by_delimiter(feat_value, MULTIPLE_VALUES_DELIMITER)
                    bad_indices = {}
                    for i,val in enumerate(values):
                        if val in comorb_info.all_codes:
                            bad_indices[i] = i
                    if len(bad_indices) > 0:
                        new_values = []
                        for i, v in enumerate(values):
                            if i not in bad_indices:
                                new_values.append(v)
                        new_str = MULTIPLE_VALUES_DELIMITER.join(new_values)
                        row[feat_idx] = new_str

    def prune_rows_remove_fields_write_to_csv(selected_episode_rows, charttime_idx, patient, comorb_info):
        prune_events_time = 0
        if years_ago > 0:
            prune_events_time = time.perf_counter()
            selected_episode_rows = choose_rows_after_n_years_ago(selected_episode_rows, charttime_idx, years_ago)
            prune_events_time = time.perf_counter()-prune_events_time
            if len(selected_episode_rows) == 0:
                pprint(f"\n\tproc {proc_id}: (no events2) {patient}")
                return -1, prune_events_time, 0 #return t, prune_events_time, remove_field_time

        # t is number of days spanning first and last event
        days_idx = episode_header_to_idx_map['DAYS']
        t = int(selected_episode_rows[-1][days_idx]) - int(selected_episode_rows[0][days_idx])

        if t == 0:
            pprint(f"\n\tproc {proc_id}: (only 1 day data) {patient}")
            return -1, prune_events_time, 0

        remove_time = time.perf_counter()
        remove_fields(selected_episode_rows, episode_header_to_idx_map, comorb_info)
        remove_time = time.perf_counter() - remove_time

        # write out to csv but omit charttime field
        fn = os.path.join(output_path, patient + '.csv')
        write_as_csv_omit_idx(fn, selected_episode_rows, episode_header, charttime_idx)
        return t, prune_events_time, remove_time

    def write_triples(filename, triples):
        # no need to write header, will combine later
        with open(filename, 'w') as fout:
            for x, t, y in triples:
                fout.write(f'{x},{t},{y}\n')


    start_sec = time.perf_counter()

    with open(config_path) as fin:
        config = json.load(fin)
    is_categorical_channel = config['is_categorical_channel']
    dtype_specs = {k: str for k, v in is_categorical_channel.items() if v}


    #This shouldn't take long, so do this in every process
    ssec = time.perf_counter()
    comorb_info = ComorbidityInfo(comorb_info_file, sgh_path, config)
    pos_patient_to_date_map = get_pos_patient_to_date_map(comorb_info)
    pos_patients = set(pos_patient_to_date_map.keys())
    all_pos_patients = get_all_pos_patients(comorb_info.comorbidity_filename, comorb_info.comorbidity_name)
    num_intersect = len(pos_patients.intersection(all_pos_patients))
    pprint(f'proc {proc_id}: #pos_patients with dates, #all_pos_patients, #intersect = '
          f'{len(pos_patients)}, {len(all_pos_patients)}, {num_intersect}')

    pprint(f'proc {proc_id}: get_pos_subjects_to_date_map took {time_str(time.perf_counter()-ssec)}')

    ssec = time.perf_counter()
    patients = os.listdir(patients_root_path)
    patients.sort() # sort so that every process has the same ordering of subjects
    pprint(f'proc {proc_id}: sorting patients took {time_str(time.perf_counter()-ssec)}')
    pprint(f'proc {proc_id}:  #total_pos_patients/#total_patients = {len(pos_patients)}/{len(patients)}')

    start_idx, end_idx = get_start_end_indices(len(patients), num_proc, proc_id)
    sub_patients = patients[start_idx:end_idx]
    pprint(f'proc {proc_id}: processing patients[{start_idx}:{end_idx}]')

    if 0 <= max_num_patients < len(patients):
        sub_patients = patients[:max_num_patients]
        pprint(f'proc {proc_id}: limit to {len(sub_patients)} patients')

    episode_header, episode_header_to_idx_map = None, None

    xty_triples_pos = []
    xty_triples_neg = []
    cnt = -1
    read_episode_time = 0
    prune_early_events_time = 0
    remove_field_time = 0
    ssec = time.perf_counter()

    for patient in sub_patients:
        cnt += 1
        if cnt % 1000 == 0:
            pprint(f'proc {proc_id}: {cnt} patients took {time_str(time.perf_counter() - ssec)}')

        patient_folder = os.path.join(patients_root_path, patient)
        patient_ts_files = os.path.join(patient_folder,'episode.csv') #Thiti
        if patient in pos_patients  and os.path.isfile(patient_ts_files): #has comorbidity  #Thiti
            read_time = time.perf_counter()
            episode_header, episode_rows, episode_header_to_idx_map \
                = read_episode_str_categorical_col(patient_folder, dtype_specs, episode_header, episode_header_to_idx_map)
            #episode_rows[pd.isnull(episode_rows)] = '' #slow; iterating through every cell of every row
            charttime_idx = episode_header_to_idx_map['CHARTTIME']
            read_episode_time += time.perf_counter() - read_time

            comorbid_date_2 = pos_patient_to_date_map[patient]
            comorbid_date_1 = comorbid_date_2 - pd.Timedelta(time_before_event, 'D')  # x days before comorbidity

            last_idx = -1
            for i in range(len(episode_rows)-1,0-1,-1):
                r = episode_rows[i]
                if r[charttime_idx] < comorbid_date_2:
                    last_idx = i
                    break
            selected_episode_rows = episode_rows[0:last_idx+1]

            if len(selected_episode_rows) == 0:
                pprint(f"\n\tproc {proc_id}: (no events) {patient}")
                continue

            #if all charttimes < mortality_date_1
            if selected_episode_rows[-1][charttime_idx] < comorbid_date_1:
                comorbidity = 0
            else:
                comorbidity = 1

            t, ptime, rtime = prune_rows_remove_fields_write_to_csv(selected_episode_rows, charttime_idx, patient,
                                                                    comorb_info)
            prune_early_events_time += ptime
            remove_field_time += rtime
            if t < 0: continue
            if comorbidity == 1: xty_triples_pos.append((patient+'.csv', t, comorbidity))
            else:                xty_triples_neg.append((patient+'.csv', t, comorbidity))

        elif patient not in all_pos_patients and os.path.isfile(patient_ts_files): #no comorbidity; also ignore pos patients without date of comorbidity
            read_time = time.perf_counter()
            episode_header, selected_episode_rows, episode_header_to_idx_map \
                = read_episode_str_categorical_col(patient_folder, dtype_specs, episode_header, episode_header_to_idx_map)
            #selected_episode_rows[pd.isnull(selected_episode_rows)] = '' #slow; iterating through every cell of every row
            read_episode_time += time.perf_counter() - read_time
            charttime_idx = episode_header_to_idx_map['CHARTTIME']
            t, ptime, rtime = prune_rows_remove_fields_write_to_csv(selected_episode_rows, charttime_idx, patient,
                                                                    comorb_info)
            prune_early_events_time += ptime
            remove_field_time += rtime
            if t < 0: continue
            xty_triples_neg.append((patient+'.csv', t, 0))

    pprint(f"proc {proc_id}: read_episode_time:       {time_str(read_episode_time)}")
    pprint(f"proc {proc_id}: prune_early_events_time: {time_str(prune_early_events_time)}")
    pprint(f"proc {proc_id}: remove_field_time:       {time_str(remove_field_time)}")
    pprint(f"proc {proc_id}: #pos_triples, #neg_triples: {len(xty_triples_pos)}, {len(xty_triples_neg)}")
    pprint(f'proc {proc_id}: created {cnt} csv for {len(sub_patients)} patients; took {time_str(time.perf_counter()-start_sec)}')

    random.shuffle(xty_triples_pos)
    random.shuffle(xty_triples_neg)

    #create_directory(os.path.join(output_path, f'pos_triples-{proc_id}'))    #Thiti
    #create_directory(os.path.join(output_path, f'neg_triples-{proc_id}'))    #Thiti
    write_triples(os.path.join(output_path, f'pos_triples-{proc_id}'), xty_triples_pos)
    write_triples(os.path.join(output_path, f'neg_triples-{proc_id}'), xty_triples_neg)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('sgh_path',                  type=str, help='Directory containing SGH CSV files.')
    parser.add_argument('output_path',               type=str, help="Directory where the created data should be stored.")
    parser.add_argument('--subjects_root_path','-s', type=str, help='Directory containing ALL subject sub-directories (not splitted in train/test).')
    parser.add_argument('--resources_folder',        type=str, help='Resources folder')
    parser.add_argument('--comorbidity_info_file',   type=str, help='file path to comorbidity info, e.g., comorbidity_foot.json')
    parser.add_argument('--num_subjects',      '-n', type=int, help='Limit the number of subjects to process; negative value means all', default=-1)
    parser.add_argument('--time_gap',                type=int, help="event must be within this time period (and before)\nif there is too large gap from the last measurement\ndate and the eventdate",default=365)
    parser.add_argument('--years_back',              type=int, help="number of most recent years of patient history to consider (negative or zero value means consider all years in patient history)",default=5)
    parser.add_argument('--number_processes',  '-p', type=int, help='', default=10)
    parser.add_argument('--frac_validation',         type=float, help='', default=0.1)
    parser.add_argument('--frac_test',               type=float, help='', default=0.2)
    args = parser.parse_args()

    _sgh_path           = args.sgh_path
    _output_path        = args.output_path
    _subjects_root_path = args.subjects_root_path #directory containing all subjects sub-directories; not divided into train/valid/test
    _config_path        = os.path.join(args.resources_folder, 'discretizer_config.json')
    _comorb_info_file   = args.comorbidity_info_file
    _num_subjects       = args.num_subjects
    _time_gap           = args.time_gap
    _years_back         = args.years_back
    _number_processes   = args.number_processes
    _frac_validation    = args.frac_validation
    _frac_test          = args.frac_test
    _time_before_event  = _time_gap  #die within this number of days from date of death

    assert 0 <= _frac_validation <= 1.0 and 0 <= _frac_test <= 1.0 and 0 <= _frac_validation+_frac_test <= 1.0

    begin_sec = time.perf_counter()

    if not os.path.exists(_output_path):
        os.mkdir(_output_path)
        pprint(f'{_output_path} created')
    else:
        pprint(f'{_output_path} exists')

    pprint('multiprocessing patients')
    start_sec2 = time.perf_counter()
    multiprocess(_number_processes, _subjects_root_path, _config_path, _comorb_info_file, _sgh_path, _time_before_event,
                 _output_path, _num_subjects, _years_back)
    pprint(f'multiprocessing ALL patients took {time_str(time.perf_counter() - start_sec2)}')

    ssec = time.perf_counter()
    split_triples_train_test_valid()
    pprint(f'merge_triples took {time_str(time.perf_counter() - ssec)}')
    pprint(f'create_subject_csv took {time_str(time.perf_counter() - begin_sec)}')
