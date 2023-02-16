#!/usr/bin/env python
# coding: utf-8

import argparse, os, sys, csv, chardet, time, copy
import pandas as pd
from datetime import timedelta
from multiprocessing import Process
from resources.utils.utils import write_list_as_csv, split_by_comma, get_dataframe_header_and_rows
from resources.utils.utils import get_header_to_idx_map, time_str, get_start_end_indices

parser = argparse.ArgumentParser(description='Extract per-subject data from SGH CSV files.')
parser.add_argument('sgh_path',                   type=str, help='Directory containing SGH CSV files.')
parser.add_argument('output_path',                type=str, help='Directory where per-subject data should be written.')
parser.add_argument('--number_processes',   '-p', type=int, help='', default=10)
parser.add_argument('--max_num_tables',     '-t', type=int, help='Limit the number of tables to process; negative value means all', default=-1)
parser.add_argument('--max_num_rows',       '-r', type=int, help='Limit the number of rows per table to process; negative value means all', default=-1)
parser.add_argument('--incr_adm_prescribed','-i', type=int, help='create row by extending admission and prescribed dates', default=1)
parser.add_argument('--incr_by_days',       '-d', type=int, help='size of timesteps to extend admission and prescribed dates; must be positive', default=1)
args = parser.parse_args()
assert args.incr_by_days > 0, f"expect args.incr_by_days > 0 but is {args.incr_by_days}"

_sgh_path       = args.sgh_path
_output_path    = args.output_path
_num_processes  = args.number_processes
_max_num_tables = args.max_num_tables
_max_num_rows   = args.max_num_rows
_incr_adm_prescribed = bool(args.incr_adm_prescribed)
_incr_by_days   = args.incr_by_days

begin_sec = time.perf_counter()

#also see item_id_dict and tables data structure
MASTER_LIST_FILENAME                    = '1_MASTER LIST (2013-2018).csv'
COMORBIDITY_MACROVASCULAR_FILENAME      = 'COMORBIDITY_MACROvascular_Complications (2013-2018).csv'
COMORBIDITY_MICROVASCULAR_FILENAME      = 'COMORBIDITY_MICROvascular_Complications (2013-2018).csv'
COMORBIDITY_ALL_FILENAME                = 'COMORBIDITY_ALL (2013-2018).csv'
OUTCOME_DEATH_FILENAME                  = 'OUTCOME_DEATH (2013-Jun2019).csv'
VISITS_OUTPATIENT_FILENAME              = 'VISITS_Outpatient visits (2013-2018).csv'
MEDS_ANTIHYPERTENSIVE_DISPENSE_FILENAME = 'MEDS_Anti-hypertensive_Dispensed (2013-2018).csv'
CODES_SHPWKC_FILENAME                   = 'CODES_SHPWKC_AC_Diagnosis codes (2013-2018).csv'
#handle prescription end date
HANDLE_PRESCRIPTION_END_DATE_FLAG       = "SETZERO"
END_DATE_TRAILING_FLAG                  = "_0"

item_id_dict={
     'CE_Physical Measurements (2013-2018).csv':'CE1',
     'LAB_ALBUMIN, SERUM (2013-2018).csv':'LAB1',
     'LAB_ALT (2013-2018).csv':'LAB2',
     'LAB_AST (2013-2018).csv':'LAB3',
     'LAB_CK (2013-2018).csv':'LAB4',
     'LAB_CREATININE, SERUM (2013-2018).csv':'LAB5',
     'LAB_GGT (2013-2018).csv':'LAB6',
     'LAB_GLUCOSE, FASTING (2013-2018).csv':'LAB7',
     'LAB_HBA1C (2013-2018).csv':'LAB8',
     'LAB_LIPID PANEL (2013-2018).csv':'LAB9',
     'LAB_MICROALB CREATININE RATIO, URINE (2013-2018).csv':'LAB10',
     'LAB_NT-PROBNP, SERUM (2013-2018).csv':'LAB11',
     'LAB_PLATELET (2013-2018).csv':'LAB12',
     'LAB_POTASSIUM, SERUM (2013-2018).csv':'LAB13',
     'LAB_PROTEIN CREATININE RATIO, URINE (2013-2018).csv':'LAB14',
     'LAB_UREA, SERUM (2013-2018).csv':'LAB15',
     'CODES_SHPWKC_AC_Diagnosis codes (2013-2018).csv':"CODES1",
     'CODES_SNOMED_Diagnosis codes (2013-2018).csv':"CODES2",
     'OUTCOME_AMI (2011-2018).csv':"OUTCOME_LS1",
     'OUTCOME_STROKE HAEMORRHAGIC (2011-2018).csv':"OUTCOME_LS2",
     'OUTCOME_STROKE ISCHAEMIC (2011-2018).csv':"OUTCOME_LS3",
     'MEDS_Anti-hypertensive_Dispensed (2013-2018).csv':"MEDS_LS1_1",
     'MEDS_Anti-lipids_Dispensed (2013-2018).csv':"MEDS_LS1_2",
     'MEDS_Anti-Platelet_Dispensed (2013-2018).csv':"MEDS_LS1_3",
     'MEDS_GLP-1 RA_Dispensed (2013-2018).csv':"MEDS_LS1_4",
     'MEDS_Insulin_Dispensed (2013-2018).csv':"MEDS_LS1_5",
     'MEDS_Oral DM_Dispensed (2013-2018).csv':"MEDS_LS1_6",
     'MEDS_Anti-hypertensive_Prescribed (2013-2018).csv':"MEDS_LS2_1",
     'MEDS_Anti-lipids_Prescribed (2013-2018).csv':"MEDS_LS2_2",
     'MEDS_Anti-Platelet_Prescribed (2013-2018).csv':"MEDS_LS2_3",
     'MEDS_DM_Prescribed (2013-2018).csv':"MEDS_LS2_4",
     'VISITS_Admissions (2011-2018).csv':'VISITS_LS_1',
     'VISITS_Admissions_CGH (2011-2018).csv':'VISITS_LS_2',
     'OUTCOME_CABG (2011-2018).csv':'OUTCOME_1',
     'OUTCOME_PCI (2011-2018).csv':'OUTCOME_2',
     'OUTCOME_HOSPITALISATION_HF (2011-2018).csv':'OUTCOME_3',
     'SURGERY_Bariatric surgery (2013-2018).csv':'SURGERY_1',
     'SURGERY_LL bypass & revascularisation (2014-2018).csv':'SURGERY_2',
     'SURGERY_Lower limp amputations (2013-2018).csv':'SURGERY_3',
     'VISITS_Emergency department (2011-2018).csv':'VISITS_1',
     'VISITS_Emergency department_CGH (2011-2018).csv':'VISITS_2',
     'VISITS_Outpatient visits (2013-2018).csv':'VISITS_3'
}


def dataframe_from_csv(path, header=0, index_col=0):
    return pd.read_csv(path, header=header, index_col=index_col, error_bad_lines=False, dtype=str)

def read_subjects(subject_path):
    fn = os.path.join(subject_path, MASTER_LIST_FILENAME)
    df = dataframe_from_csv(fn, index_col=None)
    return get_dataframe_header_and_rows(df)

def read_comorbid(data_path):

    def get_lines(filename):
        with open(filename) as fin:
            lines = fin.readlines()
        header = lines[0]
        header = split_by_comma(header.strip())
        header_to_idx_map = dict([(h.strip(), i) for i, h in enumerate(header)])
        lines = lines[1:]  # skip header
        return header, lines, header_to_idx_map

    def create_comorbidities_map():
        return {'YEAR': '2013', #unused, so always set to same value
                'ESTABLISHED_CVD': 'NO',
                'IHD_PRIOR': 'NO',
                'PAD_PRIOR': 'NO',
                'STROKE_COMBINED_PRIOR': 'NO',
                'STROKE_ISCHAEMIC_PRIOR': 'NO',
                'STROKE_HAEMORRHAGIC_PRIOR': 'NO',
                'TIA_PRIOR': 'NO',
                'ATRIAL_FIBRILLATION_PRIOR': 'NO',
                'MICROVASCULAR_CX': 'NO',
                'EYE_CX': 'NO',
                'NEPHROPATHY': 'NO',
                'NEUROPATHY': 'NO',
                'DIABETIC_FOOT': 'NO'}

    def get_comorbidities(id_to_comorbidites, idd):
        comorbs = id_to_comorbidites.get(idd)
        if comorbs is None:
            comorbs = create_comorbidities_map()
            id_to_comorbidites[idd] = comorbs
        return comorbs

    def set_comorbidities(lines, header_to_idx_map, comorb_names, id_to_comorbidites_map):
        for l in lines:
            cols    = split_by_comma(l.strip())  # cols[1] is YEAR
            dcof_id = cols[header_to_idx_map['DCOF_ID']]
            comorbidites = get_comorbidities(id_to_comorbidites_map, dcof_id)
            for comorb_name in comorb_names:
                comorb_idx = header_to_idx_map[comorb_name]
                if cols[comorb_idx].lower() != 'no': comorbidites[comorb_name] = cols[comorb_idx]

    header1, lines1, header1_to_idx_map = get_lines( os.path.join(data_path, COMORBIDITY_MACROVASCULAR_FILENAME) )
    header2, lines2, header2_to_idx_map = get_lines( os.path.join(data_path, COMORBIDITY_MICROVASCULAR_FILENAME) )

    comorb_names1 = ['ESTABLISHED_CVD', 'IHD_PRIOR', 'PAD_PRIOR', 'STROKE_COMBINED_PRIOR', 'STROKE_ISCHAEMIC_PRIOR',
                     'STROKE_HAEMORRHAGIC_PRIOR', 'TIA_PRIOR', 'ATRIAL_FIBRILLATION_PRIOR']
    comorb_names2 = ['MICROVASCULAR_CX', 'EYE_CX', 'NEPHROPATHY', 'NEUROPATHY', 'DIABETIC_FOOT']

    id_to_comorbidites_map = {} #dcof_id to comorbidities
    set_comorbidities(lines1, header1_to_idx_map, comorb_names1, id_to_comorbidites_map)
    set_comorbidities(lines2, header2_to_idx_map, comorb_names2, id_to_comorbidites_map)

    # omit "YEAR" from out_header because it is year of audit not year of comorbidity
    out_header = ["DCOF_ID", "ESTABLISHED_CVD", "IHD_PRIOR", "PAD_PRIOR", "STROKE_COMBINED_PRIOR",
                  "STROKE_ISCHAEMIC_PRIOR", "STROKE_HAEMORRHAGIC_PRIOR", "TIA_PRIOR", "ATRIAL_FIBRILLATION_PRIOR",
                  "MICROVASCULAR_CX", "EYE_CX", "NEPHROPATHY", "NEUROPATHY", "DIABETIC_FOOT"]
    out_header_to_idx_map = get_header_to_idx_map(out_header)
    rows = []
    for dcof_id, comorb in id_to_comorbidites_map.items():
        row = [dcof_id]
        for i in range(1, len(out_header)):
            header_name = out_header[i]
            row.append(comorb[header_name])
        rows.append(row)

    return out_header, rows, out_header_to_idx_map

def read_csv_encoding_adjusted(sgh_path,table):
    lines=[]
    with open(os.path.join(sgh_path, table), 'rb') as f:
         for line in f:
                detected_encoding= chardet.detect(line)["encoding"]
                try:
                    line=line.decode("utf-8")
                except:
                    line=line.decode(detected_encoding)
                lines.append(line)
    return csv.DictReader(lines)

def row_out_table(table,row):
    row_out = None
    if item_id_dict[table].find("CE1") != -1:
        row_out = {'DCOF_ID': row['DCOF_ID'],
                'CHARTTIME': row['MEASUREMENT_TAKEN_DATE'],
                'ITEMID': item_id_dict[table],
                'VALUE': row['MEASUREMENT_VAL'],
                'VALUEUOM': ''}
        if row["MEASUREMENT_ID"].strip()=='BLOOD_PRESSURE_SYSTOLIC':
            row_out['ITEMID']="CE1_1"
        elif row["MEASUREMENT_ID"].strip()=='BLOOD_PRESSURE_DIASTOLIC':
            row_out['ITEMID']="CE1_2"
        elif row["MEASUREMENT_ID"].strip()=='HEIGHT_CM':
            row_out['ITEMID']="CE1_3"
        elif row["MEASUREMENT_ID"].strip()=='WEIGHT_KG':
            row_out['ITEMID']="CE1_4"
        elif row["MEASUREMENT_ID"].strip()=='BMI':
            row_out['ITEMID']="CE1_5"
        elif row["MEASUREMENT_ID"].strip()=='HEARTRATE':
            row_out['ITEMID']="CE1_6"
    elif item_id_dict[table].find("LAB") != -1:
        row_out = {'DCOF_ID': row['DCOF_ID'],
                'CHARTTIME': row['SPECIMEN_DATE'],
                'ITEMID': item_id_dict[table],
                'VALUE': row['RESULT_VALUE'],
                'VALUEUOM': row['UNIT_OF_MEASURE']}
        if item_id_dict[table]=="LAB7":
            if row["LAB_TEST_ITEM_NAME"].strip()=="GLUCOSE (FASTING), SERUM".strip():
                row_out['ITEMID']="LAB7_1"
            elif row["LAB_TEST_ITEM_NAME"].strip()=="GLUCOSE (FASTING), PLASMA".strip():
                row_out['ITEMID']="LAB7_2"
            elif row["LAB_TEST_ITEM_NAME"].strip()=="GLUCOSE, PLASMA".strip():
                row_out['ITEMID']="LAB7_3"
            elif row["LAB_TEST_ITEM_NAME"].strip()=="GLUCOSE, SERUM".strip():
                row_out['ITEMID']="LAB7_4"
            elif row["LAB_TEST_ITEM_NAME"].strip()=="GLUCOSE, PLASMA, FASTING".strip():
                row_out['ITEMID']="LAB7_5"
            else:
                assert False, "Missing Labtest Name LAB7"
        elif item_id_dict[table]=="LAB8":
            if row["LAB_TEST_ITEM_NAME"].strip()=="Glycosylated Haemoglobin Alc".strip():
                row_out['ITEMID']="LAB8_1"
            elif row["LAB_TEST_ITEM_NAME"].strip()=="HbA1C (NGSP)".strip() or row["LAB_TEST_ITEM_NAME"].strip()=="HbA1C (NGSP.)".strip():
                row_out['ITEMID']="LAB8_2"
            elif row["LAB_TEST_ITEM_NAME"].strip()=="HBA1c, blood".strip():
                row_out['ITEMID']="LAB8_3"
            else:
                assert False, "Missing Labtest Name LAB8"
        elif item_id_dict[table]=="LAB9":
            if row["LAB_TEST_ITEM_NAME"].strip()=="CHOLESTEROL HDL, SERUM".strip():
                row_out['ITEMID']="LAB9_1"
            elif row["LAB_TEST_ITEM_NAME"].strip()=="CHOLESTEROL LDL, CALC".strip():
                row_out['ITEMID']="LAB9_2"
            elif row["LAB_TEST_ITEM_NAME"].strip()=="CHOLESTEROL TOTAL, SERUM".strip():
                row_out['ITEMID']="LAB9_3"
            elif row["LAB_TEST_ITEM_NAME"].strip()=="LDL CHOLESTEROL, DIRECT, SERUM".strip():
                row_out['ITEMID']="LAB9_4"
            elif row["LAB_TEST_ITEM_NAME"].strip()=="TRIGLYCERIDES, SERUM".strip():
                row_out['ITEMID']="LAB9_5"
            elif row["LAB_TEST_ITEM_NAME"].strip()=="LDL (CALCULATED)".strip():
                row_out['ITEMID']="LAB9_6"
            else:
                assert False, "Missing Labtest Name LAB9"
        elif item_id_dict[table]=="LAB14":
            if row["LAB_TEST_ITEM_NAME"].strip()=="Protein (24-hour), urine".strip():
                row_out['ITEMID']="LAB14_1"
            elif row["LAB_TEST_ITEM_NAME"].strip()=="Protein/Creatinine Ratio".strip():
                row_out['ITEMID']="LAB14_2"
            elif row["LAB_TEST_ITEM_NAME"].strip()=="Urine Protein/Cre Ratio".strip():
                row_out['ITEMID']="LAB14_3"
            elif row["LAB_TEST_ITEM_NAME"].strip()=="Protein Creatinine Ratio".strip():
                row_out['ITEMID']="LAB14_4"
            else:
                assert False, "Missing Labtest Name LAB14"
    elif item_id_dict[table].find("CODES") != -1:
        row_out = {'DCOF_ID': row['DCOF_ID'],
                'CHARTTIME': row['DIAGNOSIS_DATE'],
                'ITEMID': item_id_dict[table],
                'VALUE': row['DIAGNOSIS_CODE'],
                'VALUEUOM':"" }
    elif item_id_dict[table].find("OUTCOME_LS") != -1:
        #Thiti handle prescription end date
        if row["LENGTH_OF_STAY"] == HANDLE_PRESCRIPTION_END_DATE_FLAG: suffix = END_DATE_TRAILING_FLAG
        else:                                                          suffix = ""
        row_out = {'DCOF_ID': row['DCOF_ID'],
                'CHARTTIME': row['ADMISSION_DATE'],
                'ITEMID': item_id_dict[table],
                'VALUE': row['DIAGNOSIS_CODE']+row['DIAGNOSIS_CODING_TYPE']+suffix,             #Thiti handle prescription end date
                'VALUEUOM':"" }
    elif item_id_dict[table].find("MEDS_LS1") != -1:
        #handle prescription end date
        if row["DURATION"] == HANDLE_PRESCRIPTION_END_DATE_FLAG: suffix = END_DATE_TRAILING_FLAG
        else:                                                    suffix = ""
        row_out = {'DCOF_ID':   row['DCOF_ID'],
                   'CHARTTIME': row['DISPENSED_DATE'],
                   'ITEMID':    item_id_dict[table],
                   'VALUE':     row['ITEM_CODE']+row['MEDICATION_GROUP_1']+row['MEDICATION_SUB_GROUP_1']+suffix,
                   'VALUEUOM':  "" }
    elif item_id_dict[table].find("MEDS_LS2") != -1:
        #Thiti handle prescription end date
        if row["DURATION"] == HANDLE_PRESCRIPTION_END_DATE_FLAG: suffix = END_DATE_TRAILING_FLAG
        else:                                                    suffix = ""
        if item_id_dict[table]=="MEDS_LS2_4":
            row_out = {'DCOF_ID': row['DCOF_ID'],
                'CHARTTIME': row['PRESCRIBED_DATE'],
                'ITEMID': item_id_dict[table],
                'VALUE': row['DRUG_NAME']+row['PRESCMED_GROUP_1']+suffix,           #Thiti handle prescription end date
                'VALUEUOM':"" }
        else:
            row_out = {'DCOF_ID': row['DCOF_ID'],
                    'CHARTTIME': row['PRESCRIBED_DATE'],
                    'ITEMID': item_id_dict[table],
                    'VALUE': row['DRUG_NAME']+row['MEDICATION_GROUP_1']+row['MEDICATION_SUB_GROUP_1']+suffix,    #Thiti handle prescription end date
                    'VALUEUOM':"" }
    elif item_id_dict[table].find("VISITS_LS") != -1:
        #Thiti handle prescription end date
        if row['LENGTH_OF_STAY'] == HANDLE_PRESCRIPTION_END_DATE_FLAG: suffix = END_DATE_TRAILING_FLAG
        else:                                                          suffix = ""
        row_out = {'DCOF_ID': row['DCOF_ID'],
                'CHARTTIME': row['ADMISSION_DATE'],
                'ITEMID': item_id_dict[table],
                'VALUE': row['DIAGNOSIS_CODE']+suffix,              #Thiti handle prescription end date
                'VALUEUOM':"" }
    elif item_id_dict[table].find("OUTCOME_") != -1:
        suffix=""                                                   #Thiti handle prescription end date
        if item_id_dict[table]=="OUTCOME_1":
            CHARTTIME=row['SURGERY_DATE_CABG']
        elif item_id_dict[table]=="OUTCOME_2":
            CHARTTIME=row['DATE_OF_PCI']
        elif item_id_dict[table]=="OUTCOME_3":
            #Thiti handle prescription end date
            if row["DISCHARGE_DATE"] == HANDLE_PRESCRIPTION_END_DATE_FLAG: suffix = END_DATE_TRAILING_FLAG      
            else:                                                          suffix = ""
            CHARTTIME=row['ADMISSION_DATE']                                    
        else:
            assert False, f'expected OUTCOME_1/2/3 but got {item_id_dict[table]}'
        row_out = {'DCOF_ID': row['DCOF_ID'],
                'CHARTTIME': CHARTTIME,
                'ITEMID': item_id_dict[table],
                'VALUE': str(1)+suffix,                         #Thiti handle prescription end date
                'VALUEUOM':"" }
    elif item_id_dict[table].find("SURGERY_") != -1:
        row_out = {'DCOF_ID': row['DCOF_ID'],
                'CHARTTIME': row['SURGERY_DATE'],
                'ITEMID': item_id_dict[table],
                'VALUE': row['SURGERY_CODE'],
                'VALUEUOM':"" }
    elif item_id_dict[table].find("VISITS_") != -1:
        if item_id_dict[table]=="VISITS_3":
            value= row['VISIT_SPECIALTY']+row['VISIT_TYPE']
        else:
            value=row['DIAGNOSIS_CODE']
        row_out = {'DCOF_ID': row['DCOF_ID'],
                'CHARTTIME': row['DATE_OF_VISIT'],
                'ITEMID': item_id_dict[table],
                'VALUE': value,
                'VALUEUOM':"" }
    
    if not row_out:
        try:
            assert False," undetected Table "
        except:
            print("table ",table)
            print("row_out ",row_out)
            print("row ",row)   

    assert not pd.isnull(row_out['CHARTTIME'])
    row_out['CHARTTIME']=pd.to_datetime(row_out['CHARTTIME'])

    if pd.isnull(row_out['VALUE']): row_out['VALUE'] = ''

    return row_out

def read_events_table_by_row(sgh_path, table):
    #assume iterating through csv.DicReader will be fast unlike pandas.dataframe
    if table in [VISITS_OUTPATIENT_FILENAME, CODES_SHPWKC_FILENAME]:
        reader = read_csv_encoding_adjusted(sgh_path,table) #this returns a csv.DictReader too
    else:
        reader = csv.DictReader(open(os.path.join(sgh_path, table), 'r'))
    for i, row in enumerate(reader):
        yield row, i

def read_events_table_by_row_los(sgh_path, table, do_incr, incr_by_days):
    reader = csv.DictReader(open(os.path.join(sgh_path, table), 'r'))
    for i, row in enumerate(reader):
        if item_id_dict[table]=="OUTCOME_3":
            los = pd.to_datetime(row['DISCHARGE_DATE'])-pd.to_datetime(row['ADMISSION_DATE'])
            los = los.days
            first_admission_date = pd.to_datetime(row['ADMISSION_DATE'])
        else:
            los=int(row['LENGTH_OF_STAY'])
            first_admission_date = pd.to_datetime(row['ADMISSION_DATE'])


        #handle prescription end date
        if not do_incr: # do not indicate end of prescription
            dur = 0
            end_idx = 1
        else: # indicate end of prescription
            dur = los
            end_idx = 2

        # if not indicating end of prescription, then execute loop once; otherwise twice
        for j in range(0,end_idx):
            if j == 0: days_to_add = timedelta(days=0)
            else:      days_to_add = timedelta(days=dur)

            row['ADMISSION_DATE'] = first_admission_date + days_to_add
            if j != 0: 
                if item_id_dict[table]=="OUTCOME_3":
                    row["DISCHARGE_DATE"] = HANDLE_PRESCRIPTION_END_DATE_FLAG #indicate that this must be set to 0
                else:
                    row['LENGTH_OF_STAY'] = HANDLE_PRESCRIPTION_END_DATE_FLAG

            yield row, i

def med_duration(dur):

    if not dur[0]:
        return 1
    else:
        dur[0]=int(dur[0])
        if dur[1].strip()=='W' or dur[1].strip()=='weeks':
            dur[1]=7
        elif dur[1].strip()=='D' or dur[1].strip()=='days':
            dur[1]=5
        elif dur[1].strip()=='M' or dur[1].strip()=='months':
            dur[1]=30
        elif dur[1].strip()=='hour' or dur[1].strip()=='hours' or dur[1].strip()=='H':
            dur[1]=1/24
        elif dur[1].strip()=='Y':
            dur[1]=365
        else:
            print("d[0] ",dur[0])
            print("d[1] ",dur[1])
            assert False, "Undetected Time Unit" 

        try:
            dur=round(dur[0]*dur[1])
        except:
            print("d[0] ",dur[0])
            print("d[1] ",dur[1])
            
        return dur

def read_meds_table_by_row_los(sgh_path, table, do_incr, incr_by_days):
    if table in [MEDS_ANTIHYPERTENSIVE_DISPENSE_FILENAME]:
        reader = read_csv_encoding_adjusted(sgh_path,table)
    else:
        reader = csv.DictReader(open(os.path.join(sgh_path, table), 'r'))
    for i, row in enumerate(reader):
        if item_id_dict[table].find("MEDS_LS1") != -1:
            first_dispensed_date=pd.to_datetime(row['DISPENSED_DATE'])
        elif item_id_dict[table].find("MEDS_LS2") != -1:
            first_dispensed_date=pd.to_datetime(row['PRESCRIBED_DATE'])
        else:
            assert False, "first_dispensed_date is not assigned"

        #handle prescription end date
        if not do_incr: # do not indicate end of prescription
            dur = 0
            end_idx = 1
        else: # indicate end of prescription
            dur = row["DURATION"].split(" ")
            dur = med_duration(dur)
            end_idx = 2

        # if not indicating end of prescription, then execute loop once; otherwise twice
        for j in range(0,end_idx):
            if j == 0: days_to_add = timedelta(days=0)
            else:      days_to_add = timedelta(days=dur)

            if item_id_dict[table].find("MEDS_LS1") != -1:
                row['DISPENSED_DATE'] = first_dispensed_date + days_to_add
            elif item_id_dict[table].find("MEDS_LS2") != -1:
                row['PRESCRIBED_DATE'] = first_dispensed_date + days_to_add

            if j != 0: row["DURATION"] = HANDLE_PRESCRIPTION_END_DATE_FLAG #indicate that this must be set to 0

            yield row, i

        # COMMENTED OUT: use code above to handle prescription end date
        # if do_incr:
        #     dur = row["DURATION"].split(" ")
        #     dur = med_duration(dur)
        # else:
        #     dur = 0
        #
        # #why capture the dispensed date over full duration? This can be slow => Thiti needs this per time-step
        # for j in range(0,dur+1,incr_by_days):
        #     if item_id_dict[table].find("MEDS_LS1") != -1:
        #         row['DISPENSED_DATE'] = first_dispensed_date + timedelta(days=j)
        #     elif item_id_dict[table].find("MEDS_LS2") != -1:
        #         row['PRESCRIBED_DATE'] = first_dispensed_date + timedelta(days=j)
        #     yield row, i

def read_events_table_and_break_up_by_subject(proc_id, sgh_path, table, output_path, max_num_rows,
                                              do_incr, incr_by_days, subjects_to_keep):
    obs_header = ['DCOF_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']

    class DataStats(object):
        def __init__(self):
            self.curr_subject_id = ''
            self.last_write_no = 0
            self.last_write_nb_rows = 0
            self.last_write_subject_id = ''
            self.curr_obs = []

    data_stats = DataStats()

    def write_current_observations():
        #write only if subject_id directory exists, i.e., in master list
        dirn = os.path.join(output_path, str(data_stats.curr_subject_id))
        if not os.path.exists(dirn): return

        data_stats.last_write_no += 1
        data_stats.last_write_nb_rows = len(data_stats.curr_obs)
        data_stats.last_write_subject_id = data_stats.curr_subject_id
        dn = os.path.join(output_path, str(data_stats.curr_subject_id))
        try:
            os.makedirs(dn)
        except:
            pass
        #write 'DCOF_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM' to events.csv file
        fn = os.path.join(dn, f'events.csv-{proc_id}')
        if not os.path.exists(fn) or not os.path.isfile(fn):
            with open(fn, 'w') as fout:
                fout.write(','.join(obs_header)+'\n')
        with open(fn, 'a') as fout:
            w = csv.DictWriter(fout, fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
            w.writerows(data_stats.curr_obs)
        data_stats.curr_obs.clear()  #data_stats.curr_obs = []; clear rather than create a new list

    # get reader for each table
    if item_id_dict[table].find("MEDS_LS") != -1:
        reader = read_meds_table_by_row_los(sgh_path, table, do_incr, incr_by_days)
    elif (item_id_dict[table].find("OUTCOME_LS") != -1) or (item_id_dict[table].find("VISITS_LS") != -1) or (item_id_dict[table].find("OUTCOME_3") != -1):
        reader = read_events_table_by_row_los(sgh_path, table, do_incr, incr_by_days)
    else:
        reader = read_events_table_by_row(sgh_path, table)

    #sort the row by DCOF_ID so that you don't need to repeated open and close files multiple times for a subject
    if not do_incr:
        row_no_pairs = [(row, row_no) for row, row_no in reader]
    else:
        row_no_pairs = [(copy.deepcopy(row), copy.deepcopy(row_no)) for row, row_no in reader]
    row_no_pairs.sort(key=lambda pair: pair[0]['DCOF_ID'])  # sort pairs by DCOF_ID

    row_num = 0
    for row, row_no in row_no_pairs:

        if 0 <= max_num_rows <= data_stats.last_write_no:
            sys.stdout.write('\rproc_id {2} read {1} max rows in {0}.\n'.format(table, max_num_rows, proc_id))
            break
        row_num += 1

        if row_no % 100 == 0:
            sys.stdout.write('\rproc_id {2} processing {0}: ROW {1}...'.format(table, row_no, proc_id))

        row_out = row_out_table(table,row) #row_out contains values for ['DCOF_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']
        if row_out is None: continue
        if (subjects_to_keep is not None) and (row_out['DCOF_ID'] not in subjects_to_keep):
            continue
        if data_stats.curr_subject_id != '' and data_stats.curr_subject_id != row['DCOF_ID']:
            write_current_observations()
        data_stats.curr_obs.append(row_out)
        data_stats.curr_subject_id = row['DCOF_ID']
        row_num = row_no #for later

    if data_stats.curr_subject_id != '':
        write_current_observations()

    sys.stdout.write('\rproc_id {5} finished processing {0}: ROW {1} ...last write '
                     '({2}) {3} rows for subject {4}...DONE!\n'.format(table, row_num,
                                                                       data_stats.last_write_no,
                                                                       data_stats.last_write_nb_rows,
                                                                       data_stats.last_write_subject_id, proc_id))

def break_up_subject(subjects_header, subjects_rows, subjects_header_to_idx_map, output_path):
    header, rows, header_to_idx_map = subjects_header, subjects_rows, subjects_header_to_idx_map
    header_str = ','.join(header)
    for row in rows: # for each subject; if duplicate subject, just keep the last
        dcof_id_idx = header_to_idx_map['DCOF_ID']
        if pd.isnull(row[dcof_id_idx]): continue
        subject_id = str(row[dcof_id_idx])
        dn = os.path.join(output_path, subject_id)
        try:    os.makedirs(dn)
        except: pass
        with open(os.path.join(dn, 'subjects.csv'), 'w') as fout:
            fout.write(header_str+'\n')
            write_list_as_csv(row, fout, check_nan=True)

def break_up_comorbidities_by_subject(comorbidities_header, comorbidities_rows, comorbidity_to_idx_map, output_path,
                                      subjects_to_keep):
    header, rows, header_to_idx_map  = comorbidities_header, comorbidities_rows, comorbidity_to_idx_map
    header_str = ','.join(header)
    for row in rows: # for each subject; if duplicate subject, just keep the last
        dcof_id_idx = header_to_idx_map['DCOF_ID']
        if pd.isnull(row[dcof_id_idx]): continue
        subject_id = str(row[dcof_id_idx])
        if (subjects_to_keep is not None) and (subject_id not in subjects_to_keep):
            continue
        dn = os.path.join(output_path, subject_id)
        if os.path.exists(dn): #if subject is in master list
            with open(os.path.join(dn, 'comorbidities.csv'), 'w') as fout:
                fout.write(header_str+'\n')
                write_list_as_csv(row, fout, check_nan=True)
        else:
            sys.stdout.write(f'WARNING in break_up_comorbidities_by_subject2: {dn} does not exist.\n')

def break_up_mortality_by_subject(output_path, subjects_to_keep):
    mortality_df = dataframe_from_csv(os.path.join(_sgh_path, OUTCOME_DEATH_FILENAME), index_col=None)
    header, rows, header_to_idx_map = get_dataframe_header_and_rows(mortality_df)
    header_str = ','.join(header)
    for row in rows: # for each subject; if duplicate subject, just keep the last
        dcof_id_idx = header_to_idx_map['DCOF_ID']
        if pd.isnull(row[dcof_id_idx]): continue
        subject_id = str(row[dcof_id_idx])
        if (subjects_to_keep is not None) and (subject_id not in subjects_to_keep):
            continue
        dn = os.path.join(output_path, subject_id)
        if os.path.exists(dn): #if subject is in master list
            with open(os.path.join(dn, 'mortality.csv'), 'w') as fout:
                fout.write(header_str+'\n')
                write_list_as_csv(row, fout, check_nan=True)
        else:
            sys.stdout.write(f'WARNING in break_up_mortality_by_subject2: {dn} does not exist.\n')

start_sec = time.perf_counter()

subjects_header, subjects_rows, subjects_header_to_idx_map = read_subjects(_sgh_path)
sys.stdout.write(f'read_subjects() took: {time_str(time.perf_counter() - start_sec)}\n')

start_sec = time.perf_counter()
comorbidities_header, comorbidities_rows, comorbidity_to_idx_map = read_comorbid(_sgh_path)
sys.stdout.write(f'read_comorbid() took: {time_str(time.perf_counter() - start_sec)}\n')

start_sec = time.perf_counter()
break_up_subject(subjects_header, subjects_rows, subjects_header_to_idx_map, _output_path)
sys.stdout.write(f'break_up_subject() took: {time_str(time.perf_counter() - start_sec)}\n')

subjects_to_keep = os.listdir(_output_path)
subjects_to_keep = set([str(s) for s in subjects_to_keep])

start_sec = time.perf_counter()
break_up_comorbidities_by_subject(comorbidities_header, comorbidities_rows, comorbidity_to_idx_map, _output_path,
                                  subjects_to_keep)
sys.stdout.write(f'break_up_comorbidities_by_subject() took: {time_str(time.perf_counter() - start_sec)}\n')

start_sec = time.perf_counter()
break_up_mortality_by_subject(_output_path, subjects_to_keep)
sys.stdout.write(f'break_up_mortality_by_subject() took: {time_str(time.perf_counter() - start_sec)}\n')

def run_process(proc_id, total_num_process, sgh_path, output_path, max_num_tables, max_num_rows, do_incr, incr_by_days,
                subjects_to_keep):

    #distribute out the slow tables
    tables=[
        'MEDS_Anti-hypertensive_Dispensed (2013-2018).csv', #slow
        # 'MEDS_Anti-hypertensive_Prescribed (2013-2018).csv',
        'LAB_LIPID PANEL (2013-2018).csv', #slow
        'LAB_ALBUMIN, SERUM (2013-2018).csv',
        'LAB_ALT (2013-2018).csv',
        'LAB_AST (2013-2018).csv',
        'CODES_SNOMED_Diagnosis codes (2013-2018).csv',  # slow
        'MEDS_DM_Prescribed (2013-2018).csv',  # slow
        'LAB_CK (2013-2018).csv',
        'LAB_CREATININE, SERUM (2013-2018).csv',
        'LAB_GGT (2013-2018).csv',
        'LAB_GLUCOSE, FASTING (2013-2018).csv',
        'MEDS_Anti-lipids_Dispensed (2013-2018).csv',  # slow
        # 'MEDS_Anti-lipids_Prescribed (2013-2018).csv', #use 'dispensed' counterpart
        'MEDS_Anti-Platelet_Dispensed (2013-2018).csv',  # slow
        # 'MEDS_Anti-Platelet_Prescribed (2013-2018).csv',
        'LAB_HBA1C (2013-2018).csv',
        'LAB_MICROALB CREATININE RATIO, URINE (2013-2018).csv',
        'LAB_NT-PROBNP, SERUM (2013-2018).csv',
        'MEDS_Oral DM_Dispensed (2013-2018).csv',  # slow
        'LAB_PLATELET (2013-2018).csv',
        'LAB_POTASSIUM, SERUM (2013-2018).csv',
        'LAB_PROTEIN CREATININE RATIO, URINE (2013-2018).csv',
        'LAB_UREA, SERUM (2013-2018).csv',
        'VISITS_Admissions (2011-2018).csv',  # slow
        'CODES_SHPWKC_AC_Diagnosis codes (2013-2018).csv',
        'OUTCOME_AMI (2011-2018).csv',
        'OUTCOME_STROKE HAEMORRHAGIC (2011-2018).csv',
        'OUTCOME_STROKE ISCHAEMIC (2011-2018).csv',
        'MEDS_GLP-1 RA_Dispensed (2013-2018).csv',
        'MEDS_Insulin_Dispensed (2013-2018).csv',
        'VISITS_Admissions_CGH (2011-2018).csv', #slow
        'OUTCOME_CABG (2011-2018).csv',
        'OUTCOME_PCI (2011-2018).csv',
        'OUTCOME_HOSPITALISATION_HF (2011-2018).csv',
        'SURGERY_Bariatric surgery (2013-2018).csv',
        'SURGERY_LL bypass & revascularisation (2014-2018).csv',
        'SURGERY_Lower limp amputations (2013-2018).csv',
        'VISITS_Emergency department (2011-2018).csv',
        'VISITS_Emergency department_CGH (2011-2018).csv',
        'VISITS_Outpatient visits (2013-2018).csv' #slow
    ]

    assert total_num_process <= len(tables), f"{proc_id}: expect total num processes {total_num_process} < len(tables) {len(tables)}"
    assert proc_id < total_num_process, f"{proc_id}: expect proc_id {proc_id} < total_num_process {total_num_process}"

    #split up tables
    start_idx, end_idx = get_start_end_indices(len(tables), total_num_process, proc_id)
    sub_tables = tables[start_idx:end_idx]
    sys.stdout.write(f'proc {proc_id}: processing tables[{start_idx}:{end_idx}]\n')

    if 0<= max_num_tables < len(sub_tables):
        sub_tables = sub_tables[:max_num_tables]
        sys.stdout.write(f'proc {proc_id}: limit to {max_num_tables} tables\n')

    start_sec2 = time.perf_counter()
    for table in sub_tables:
        start_sec = time.perf_counter()

        if os.path.isfile(sgh_path+"/"+table):
            read_events_table_and_break_up_by_subject(proc_id, sgh_path, table, output_path, max_num_rows,
                                                      do_incr, incr_by_days, subjects_to_keep)

        sys.stdout.write(f'proc {proc_id}: processing {table} took {time_str(time.perf_counter() - start_sec)}\n')
    sys.stdout.write(f'proc {proc_id}: processing all sub_tables took {time_str(time.perf_counter() - start_sec2)}\n')


sys.stdout.write('multiprocessing tables\n')
start_sec3 = time.perf_counter()
jobs = []
for pid in range(_num_processes):
    p = Process(target=run_process, args=(pid, _num_processes, _sgh_path, _output_path, _max_num_tables, _max_num_rows,
                                          _incr_adm_prescribed, _incr_by_days, subjects_to_keep))
    jobs.append(p)
    p.start()
for j in jobs:
    j.join()
sys.stdout.write(f'multiprocessing ALL TABLES took {time_str(time.perf_counter() - start_sec3)}\n')


def run_process2(proc_id, total_num_process, output_path):
    subject_dirs       = os.listdir(output_path)
    start_idx, end_idx = get_start_end_indices(len(subject_dirs), total_num_process, proc_id)
    sub_subject_dirs   = subject_dirs[start_idx:end_idx]

    for subject_dir in sub_subject_dirs:
        dn = os.path.join(output_path, subject_dir)
        fn = os.path.join(dn, 'events.csv')
        with open(fn, 'w') as fout:
            header = None
            for pid in range(_num_processes):
                in_fn = os.path.join(dn, f'events.csv-{pid}')
                if os.path.exists(in_fn) and os.path.isfile(in_fn):
                    with open(in_fn) as fin:
                        lines = fin.readlines()
                    if header is None:
                        header = lines[0]
                        fout.write(header)
                    lines = lines[1:]  # ignore header
                    for line in lines:
                        fout.write(line)

sys.stdout.write('merging events files\n')
start_sec3 = time.perf_counter()
jobs = []
for pid in range(_num_processes):
    p = Process(target=run_process2, args=(pid, _num_processes, _output_path))
    jobs.append(p)
    p.start()
for j in jobs:
    j.join()
sys.stdout.write(f'merging events files took {time_str(time.perf_counter() - start_sec3)}\n')

sys.stdout.write(f'TOTAL TIME TAKEN by 1_Read_Data.py: {time_str(time.perf_counter() - begin_sec)}\n')
