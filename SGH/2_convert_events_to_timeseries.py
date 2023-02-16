#!/usr/bin/env python
# coding: utf-8

import os, sys, time, argparse, json
import pandas as pd
from resources.utils.utils import get_start_end_indices, write_list_as_csv, split_by_comma
from resources.utils.utils import get_header_to_idx_map, time_str
from multiprocessing import Process

MULTIPLE_VALUES_DELIMITER = '|'
END_DATE_TRAILING_FLAG='_0'   #Thiti handle prescription end date
parser = argparse.ArgumentParser(description='Split data into train and test sets.')
parser.add_argument('--subjects_root_path','-s', type=str, help='Directory containing subject sub-directories.')
parser.add_argument('--resources_folder',  '-r', type=str, help='Resources folder')
parser.add_argument('--number_processes',  '-p', type=int, help='', default=10)
parser.add_argument('--num_subjects',      '-n', type=int, help='Limit the number of subjects to process; negative value means all', default=-1)
args, _ = parser.parse_known_args()

_subjects_root_path = args.subjects_root_path
_config_path        = os.path.join(args.resources_folder, 'discretizer_config.json')
_num_processes      = args.number_processes
_num_subjects       = args.num_subjects
_variable_map_file  = os.path.join(args.resources_folder,'itemid_to_variable_map.csv')

begin_sec = time.perf_counter()

def read_itemid_to_variable_map():
    with open(_variable_map_file) as fin:
        lines = fin.readlines()
    lines = lines[1:] #skip header: VARIABLE,ITEMID,TABLE
    itemid_to_var_map = {}
    for line in lines:
        cols = split_by_comma(line.strip())
        variable = cols[0].strip()
        itemid   = cols[1].strip()
        itemid_to_var_map[itemid] = variable
    return itemid_to_var_map

def get_variables():
    with open(_variable_map_file) as fin:
        lines = fin.readlines()
        lines = lines[1:] #skip header: VARIABLE,ITEMID,TABLE
        variables = []
        for line in lines:
            cols = split_by_comma(line.strip())
            variable = cols[0].strip()
            variables.append(variable)
        return variables

def read_events_str(subject_path, header_to_idx_map):
    path = os.path.join(subject_path, 'events.csv')
    if not os.path.exists(path):
        print(f'WARNING: {path} not found in read_events_str2!')
        raise
    charttime_to_events_map = {}
    with open(path) as fin:
        lines = fin.readlines()
    if header_to_idx_map is None:
        header = lines[0]
        header = split_by_comma(header.strip())
        header_to_idx_map = get_header_to_idx_map(header) # create once to save time
    lines = lines[1:] #skip header
    for line in lines:
        cols = split_by_comma(line.strip()) #cols: ['dcof_id1','charttime1','itemid1','value1','valueuom1']
        cols = [c.strip() for c in cols]
        charttime_idx = header_to_idx_map['CHARTTIME']
        value_idx     = header_to_idx_map['VALUE']
        charttime     = cols[charttime_idx]
        value         = cols[value_idx]
        if len(value) > 0:
            events_list = charttime_to_events_map.get(charttime)
            if events_list is None:
                events_list = []
                charttime_to_events_map[charttime] = events_list
            events_list.append(cols)
    # charttime_to_events_map {'2001-01-15': [['dcof_id1','charttime1','itemid1','value1','valueuom1'], ]}
    return charttime_to_events_map, header_to_idx_map

def set_value(values_arr, idx, val):
    val = str(val)
    if values_arr[idx] == '':           values_arr[idx] = val
    elif type(values_arr[idx]) == str:  values_arr[idx] = [values_arr[idx], val]
    elif type(values_arr[idx]) == list: values_arr[idx].append(val)

def run_process(proc_id, num_proc, subjects_root_path, config_path, max_num_subjects):
    start_sec = time.perf_counter()

    subject_dirs = os.listdir(subjects_root_path)
    subject_dirs.sort() # sort that every process has the same ordering of subjects
    assert proc_id < num_proc < len(subject_dirs)
    start_idx, end_idx = get_start_end_indices(len(subject_dirs), num_proc, proc_id)
    sub_subject_dirs = subject_dirs[start_idx:end_idx]
    if 0 <= max_num_subjects < len(sub_subject_dirs):
        sub_subject_dirs = sub_subject_dirs[:max_num_subjects]
    print(f'proc {proc_id}: processing subject_dirs[{start_idx}:{end_idx}]')

    itemid_to_event_var_map = read_itemid_to_variable_map()

    with open(config_path) as fin:
        config = json.load(fin)
    out_header = config['header']
    assert out_header[0] == "Unnamed: 0", f'out_header[0] == "Unnamed: 0" but got {out_header[0]}'
    out_header[0] = 'CHARTTIME'
    out_header_to_idx_map = get_header_to_idx_map(out_header)

    events_header_to_idx_map = None
    event_variables = get_variables()

    demog_header = None  #demographic variables names
    demog_header_to_idx_map = None

    read_events_sec   = 0.0
    write_episode_sec = 0.0

    count = 0
    for subject_dir in sub_subject_dirs:
        if count > 0 and count % 2000 == 0:
            print(f'\tproc {proc_id}: {count} subjects took {time_str(time.perf_counter()-start_sec)}\n')
        count += 1

        dn = os.path.join(subjects_root_path, subject_dir)
        if not os.path.exists(dn):
            print(f'proc {proc_id}: WARNING: {dn} does not exists!')
            continue

        try:
            ssec = time.perf_counter()
            # charttime_to_events_map, e.g.,{'2001-01-15': [['dcof_id1','charttime1','itemid1','value1','valueuom1'], ]}
            # events_header set only once for efficiency because it is always the same: 'DCOF_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM'
            dirr = os.path.join(subjects_root_path, subject_dir)
            charttime_to_events_map, events_header_to_idx_map = read_events_str(dirr, events_header_to_idx_map)
            with open(os.path.join(subjects_root_path, subject_dir, "subjects.csv")) as fin:
                lines = fin.readlines()
            assert len(lines) == 2, f'proc {proc_id}: {subject_dir}/subjects.csv has {len(lines)} lines; expected 2 lines'
            if demog_header is None: #set once because it is always the same
                demog_header = split_by_comma(lines[0])
                demog_header_to_idx_map = get_header_to_idx_map(demog_header)
            demog_values = lines[1]
            demog_values = split_by_comma(demog_values) # one subject only ['dcof_id1','gender1',...]
            assert len(demog_values) == len(demog_header), f"expect demographic values and header to have same lengths, {len(demog_values)} != {len(demog_header)}"
            read_events_sec += time.perf_counter() - ssec
        except:
            print(f'proc {proc_id}: error reading from disk when processing subject {subject_dir}!')
            continue
        else:
            #print(f'proc {proc_id}: got {len(events)} events...')
            sys.stdout.flush()

        if len(charttime_to_events_map) == 0:
            print(f'proc {proc_id}: WARNING: no events for {subject_dir}!')
            continue
        if len(demog_values) == 0:
            print(f'proc {proc_id}: WARNING: no demographic values for {subject_dir}!')
            continue

        # sort events in order of increasing chart time
        a = [(pd.to_datetime(charttime), charttime, events) for charttime, events in charttime_to_events_map.items()]
        a.sort(key=lambda e: e[0])

        first_charttime = a[0][0]

        ssec = time.perf_counter()
        fn = os.path.join(subjects_root_path, subject_dir, 'episode.csv')
        with open(fn,'w') as fout:
            write_list_as_csv(out_header, fout, check_nan=True)

            for timeobj, charttime, events in a:
                values = [''] * len(out_header)
                out_idx = out_header_to_idx_map['CHARTTIME']
                set_value(values, out_idx, charttime)

                for event in events:  # event: ['dcof_id1','charttime1','itemid1','value1','valueuom1']
                    item_idx = events_header_to_idx_map['ITEMID']
                    val_idx  = events_header_to_idx_map['VALUE']
                    itemid = event[item_idx]
                    value  = event[val_idx]
                    variable = itemid_to_event_var_map[itemid]
                    out_idx  = out_header_to_idx_map[variable]
                    set_value(values, out_idx, value)
                #charttime1, var1_val, var2_val,...

                dcof_id_idx      = demog_header_to_idx_map['DCOF_ID']
                dob_idx          = demog_header_to_idx_map['DOB']
                diab_diag_yr_idx = demog_header_to_idx_map['DIABETES_DIAGNOSIS_YEAR']
                diab_type_idx    = demog_header_to_idx_map['DIABETES_TYPE']

                for i in range(len(demog_values)):  # demog_values[0] is dcof_id; NOTE: dcof_id is not written out
                    if i == dcof_id_idx or i == diab_type_idx:  # omit DCOF_ID and DIABETES_TYPE
                        continue
                    elif i == dob_idx:  # DOB
                        dob = pd.to_datetime(demog_values[i])
                        age = (timeobj - dob).days / 365.0
                        out_idx = out_header_to_idx_map['AGE']
                        set_value(values, out_idx, age)
                    elif i == diab_diag_yr_idx:  # DIABETES_DIAGNOSIS_YEAR
                        year = pd.to_datetime(demog_values[i])
                        diff_yrs = (timeobj - year).days / 365.0
                        out_idx = out_header_to_idx_map['DIABETES_DIAGNOSIS_YEAR']
                        set_value(values, out_idx, diff_yrs)
                    else:
                        hdr = demog_header[i]
                        value = demog_values[i]
                        out_idx = out_header_to_idx_map[hdr]
                        set_value(values, out_idx, value)
                 #charttime1, var1_val, var2_val,..., gender1, ethnicity1,...

                out_idx = out_header_to_idx_map['DAYS']
                set_value(values, out_idx, (timeobj - first_charttime).days)
                # charttime1, var1_val, var2_val,..., gender1, ethnicity1,...,days1

                for i in range(len(values)):
                    val = values[i]
                    if type(val) != list and pd.isnull(val): val = '' #pd.isnull applies to every element in list
                    if type(val) != list and val==END_DATE_TRAILING_FLAG: val = ''
                    if type(val) == str:
                        if "," in val:
                            fout.write(f'"{val}"')
                        else:
                            fout.write(val)
                    else:
                        assert type(val) == list, f"expect type(val) == list instead of {type(val)}"
                        val=[v for v in val if not (pd.isnull(v) or v==END_DATE_TRAILING_FLAG)]
                        val = MULTIPLE_VALUES_DELIMITER.join(val)
                        fout.write(f'"{val}"')
                    if i != len(values) - 1: fout.write(',')
                    else:                    fout.write('\n')

        write_episode_sec += time.perf_counter() - ssec

    print(f'proc {proc_id}: read_events_sec   = {time_str(read_events_sec)}')
    print(f'proc {proc_id}: write_episode_sec = {time_str(write_episode_sec)}')
    print(f'proc {proc_id}: processing all sub_subject_dirs took {time_str(time.perf_counter() - start_sec)}')


print('multiprocessing subject directories')
start_sec3 = time.perf_counter()
jobs = []
for pid in range(_num_processes):
    # create episodes.csv
    p = Process(target=run_process, args=(pid, _num_processes, _subjects_root_path, _config_path,_num_subjects))
    jobs.append(p)
    p.start()
for j in jobs:
    j.join()
print(f'multiprocessing ALL subject directories took {time_str(time.perf_counter() - start_sec3)}')

print(f'TOTAL TIME TAKEN by 2_convert_events_to_timeseries.py: {time_str(time.perf_counter() - begin_sec)}')
