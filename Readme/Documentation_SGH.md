##################################### SGH data #####################################

##################################### Files #####################################
Document files
a. "Data Dicitionary v1.0_Empower Dataset (updated 15 Sep 2020).xlsx"
b. "Descriptive_v3.xlsx"

Raw CSV files are in
c. "Datasets 2020.05.12"
list of files is in sheet "Raw File Names" from "Data Dicitionary v1.0_Empower Dataset (updated 15 Sep 2020).xlsx"

d. "Diagnosis Codes for Vivian"
This contains the diagnosis codes for comorbidities.

The main files are
1. 1_Read_Data.py
2. 2_convert_events_to_timeseries.py
3. 4_create_subject_csv.py
4. prepare_dynamic_data.py
5. prepare_static_data.py
6. select_features_static_data.py
7. run_static_model.py

Only 7. run_static_model.py execute training/testing of static model (including XGBoost and Logistic Regression). Others are about creation of a task for a model.

Utility files are
8. 210404_listfile_adjust.py

This file is for adjusting the listfile (train/validation/test) in case they are different. For example, the output from 4_create_subject_csv.py for static and dynamic data might be different.

##################################### 1_Read_Data.py #####################################
This file reads nearly all the tables in csv files in "Datasets 2020.05.12". After reading the files, it is separated into folder of each patient (DCOF_ID). In the folder there will be 3 files, 
1. "subjects.csv": this is read from "1_MASTER LIST (2013-2018).csv"
2. "comorbidities.csv": this is not needed anymore.
"COMORBIDITY_ALL (2013-2018).csv"
"COMORBIDITY_MACROvascular_Complications (2013-2018).csv"
"COMORBIDITY_MICROvascular_Complications (2013-2018).csv"
3. "events.csv": this is read from the list of tables in line 637. The table that is not read includes

"CE_Physical Measurements (2013-2018).csv"
"MEDS_Anti-hypertensive_Prescribed (2013-2018).csv"
"MEDS_Anti-lipids_Prescribed (2013-2018).csv"
"MEDS_Anti-Platelet_Prescribed (2013-2018).csv"
"OUTCOME_DEATH (2013-Jun2019).csv"

The tables start with "MEDS" above are not read since "Dispensed" version is used instead of "Prescribed".

"events.csv"
In "events.csv" which are separated into each folder of a patient, there are 5 columns 
1. "DCOF_ID"  2. "CHARTTIME"  3. "ITEMID"  4. "VALUE" 5. "VALUEUOM"

These will be filled by the tables (in line 637). The value read from the table will be separated these 5 fields 
1. "DCOF_ID" : This suppose to keep the patient's dcof id.
2. "CHARTTIME" : This suppose to keep the date of the item of interest. For example, if the item of interest is lab result measuring protein creatinine ratio, CHARTTIME is date specimen was collected.
3. "ITEMID" : This suppose to keep the name of the item of interest.
4. "VALUE" : This suppose to keep the value of the item of interest.
5. "VALUEUOM" : This suppose to keep the unit of the item of interest.

Which columns are mapped to which field (e.g. "VALUE") are defined in function "row_out_table".

The tables deserved attention are the following tables
     'OUTCOME_AMI (2011-2018).csv'
     'OUTCOME_STROKE HAEMORRHAGIC (2011-2018).csv'
     'OUTCOME_STROKE ISCHAEMIC (2011-2018).csv'
     'MEDS_Anti-hypertensive_Dispensed (2013-2018).csv'
     'MEDS_Anti-lipids_Dispensed (2013-2018).csv'
     'MEDS_Anti-Platelet_Dispensed (2013-2018).csv'
     'MEDS_GLP-1 RA_Dispensed (2013-2018).csv'
     'MEDS_Insulin_Dispensed (2013-2018).csv'
     'MEDS_Oral DM_Dispensed (2013-2018).csv'
     'MEDS_DM_Prescribed (2013-2018).csv'
     'VISITS_Admissions (2011-2018).csv'
     'VISITS_Admissions_CGH (2011-2018).csv'

For example, "MEDS_Anti-Platelet_Dispensed" table contain a medication the patient received. The column "DURATION" in csv file indicate how long the medication lasts. Therefore it is desirable to increment the event read from the table by duration. To achieve this, the process will be separated into 3 steps
1. CHARTTIME will be filled with start date and VALUE will have its original value.
2. Another CHARTTIME will be filled with end date and VALUE will have its value added by END_DATE_TRAILING_FLAG to indicate that this is end date.
3. The value of the dates in between of start and end date will be populated by imputation process in "prepare_dynamic_data.py".

Whether or not to increment is controlled by the input argument "incr_adm_prescribed", when executed python code. To explain we use the following example.

Suppose "ANTI-HYPERTENSIVESACE INHIBITOR" is to be taken by a patient for 3 days. 

1. If "incr_adm_prescribed" is set to zero (no increment), the event will not be incremented. That is the value read can be something as follows
DCOF_ID,CHARTTIME,ITEMID,VALUE,VALUEUOM
S123,2013-08-15,MEDS_LS1_1,ANTI-HYPERTENSIVESACE INHIBITOR,

2. If "incr_adm_prescribed" is set to one (do increment), the event will be incremented. In the events.csv will be
DCOF_ID,CHARTTIME,ITEMID,VALUE,VALUEUOM
Sc59e189d84,2013-08-15,MEDS_LS1_1,ANTI-HYPERTENSIVESACE INHIBITOR,
Sc59e189d84,2013-08-17,MEDS_LS1_1,ANTI-HYPERTENSIVESACE INHIBITOR_0,

Here END_DATE_TRAILING_FLAG is "_0". The imputation of this will be later explained.


Example
#For dynamic model
python -u 1_Read_Data.py '/data/Thiti/data/Datasets 2020.05.12' /data/Thiti/data/1_Read_Data/210404_1_Read_Data_1incradm --number_processes=14 --incr_adm_prescribed=1

#For static model
python -u 1_Read_Data.py '/data/Thiti/data/Datasets 2020.05.12' /data/Thiti/data/1_Read_Data/210404_1_Read_Data_0incradm --number_processes=14 --incr_adm_prescribed=0

##################################### 2_convert_events_to_timeseries.py #####################################
The file transforms the events read in "subjects.csv" , "comorbidiites.csv" and "events.csv" to "episode.csv". The transformation is done as follows 
Suppose the events.csv file is like this
DCOF_ID,CHARTTIME,ITEMID,VALUE,VALUEUOM
S123,2013-08-15,LAB7_1,0.5,
S123,2013-08-15,LAB6,1.1,
and 
subjects.csv file is like this
DCOF_ID,GENDER
S123,FEMALE

The resulting "episode.csv" will be
CHARTTIME,AGE,BLOOD_PRESSURE_DIASTOLIC,BLOOD_PRESSURE_SYSTOLIC,BMI,CHOLESTEROL_HDL_SERUM,CHOLESTEROL_LDL_CALC,CHOLESTEROL_TOTAL_SERUM,CODES_SHPWKC_AC_Diagnosis codes,CODES_SNOMED_Diagnosis codes,DIABETES_DIAGNOSIS_YEAR,ETHNICITY,GENDER,GLUCOSE_(FASTING)_PLASMA,GLUCOSE_(FASTING)_SERUM,GLUCOSE_PLASMA,GLUCOSE_PLASMA_FASTING,GLUCOSE_SERUM,Glycosylated Haemoglobin Alc,HBA1c_blood,HEARTRATE,HEIGHT_CM,HOUSING_TYPE,HYPERCHOLESTEROLAEMIA,HYPERTENSION,HbA1C (NGSP),LAB_ALBUMIN_SERUM,LAB_ALT,LAB_AST,LAB_CK,LAB_CREATININE_SERUM,LAB_GGT,LAB_MICROALB_CREATININE_RATIO_URINE,LAB_NT-PROBNP_SERUM,LAB_PLATELET,LAB_POTASSIUM_SERUM,LAB_UREA_SERUM,LDL (CALCULATED),LDL_CHOLESTEROL_DIRECT_SERUM,MEDS_Anti-Platelet_Dispensed,MEDS_Anti-Platelet_Prescribed,MEDS_Anti-hypertensive_Dispensed,MEDS_Anti-hypertensive_Prescribed,MEDS_Anti-lipids_Dispensed,MEDS_Anti-lipids_Prescribed,MEDS_DM_Prescribed,MEDS_GLP-1 RA_Dispensed,MEDS_Insulin_Dispensed,MEDS_Oral DM_Dispensed,OUTCOME_AMI,OUTCOME_CABG,OUTCOME_HOSPITALISATION_HF,OUTCOME_PCI,OUTCOME_STROKE HAEMORRHAGIC,OUTCOME_STROKE ISCHAEMIC,PRIMARY_SITE,Protein Creatinine Ratio,Protein/Creatinine Ratio,Protein_(24-hour)_urine,RENTAL_BLOCK,SMOKING,SURGERY_Bariatric surgery,SURGERY_LL bypass & revascularisation,SURGERY_Lower limp amputations,TRIGLYCERIDES_SERUM,Urine Protein/Cre Ratio,VISITS_Admissions,VISITS_Admissions_CGH,VISITS_Emergency department,VISITS_Emergency department_CGH,VISITS_Outpatient visits,WEIGHT_KG,DAYS
2013-08-15,,,,,,,,,,,,FEMALE,,0.5,,,,,,,,,,,,,,,,,1.1,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,0

Note that the header follows "header" in "./resources/discretizer_config.json" with additional CHARTTIME.


Example
#For dynamic model
python -u 2_convert_events_to_timeseries.py -s=/data/Thiti/data/2_convert_events_to_timeseries/210404_2_convert_events_to_timeseries_1incradm -r='./resources' -p=14 

#For static model
python -u 2_convert_events_to_timeseries.py -s=/data/Thiti/data/2_convert_events_to_timeseries/210404_2_convert_events_to_timeseries_0incradm -r='./resources' -p=14


##################################### 4_create_subject_csv.py #####################################
After having "episode.csv" for all patients, we can create a task defined how we map diagnosis codes to positive cases. These are in "./Prof_Code/config" (derived from "Diagnosis Codes for Vivian"). 

#"frac_validation" indicates percentage of a data that will be in validation set. "frac_test" indicates percentage of a data that will be in test set. "time_gap" indicates that event must be within this time period (and before). if there is too large gap from the last measurement date and the eventdate, it will be negtive. "years_back" indicates that the number of most recent years of patient history to consider (negative or zero value means consider all years in patient history)

#Create eye complication task. for static model
python -u 4_create_subject_csv.py '/data/Thiti/data/Datasets 2020.05.12' '/data/Thiti/data/Eye_Complication/210404_Eye_Complication_0incradm_365timegap_15fracvaltest' -s=/data/Thiti/data/2_convert_events_to_timeseries/210403_2_convert_events_to_timeseries_0incradm --resources_folder='./resources' --comorbidity_info_file='./Prof_Code/config/comorbidity_eye.json' --number_processes=14 --frac_validation=0.15 --frac_test=0.15 --time_gap=365 --years_back=-1 

#Create ischaemic stroke task. for static model
python -u 4_create_subject_csv.py '/data/Thiti/data/Datasets 2020.05.12' '/data/Thiti/data/Stroke_Ischaemic/210404_Stroke_Ischaemic_0incradm_365timegap_15fracvaltest' -s=/data/Thiti/data/2_convert_events_to_timeseries/210403_2_convert_events_to_timeseries_0incradm --resources_folder='./resources' --comorbidity_info_file='./Prof_Code/config/comorbidity_ischaemic_stroke.json' --number_processes=14 --frac_validation=0.15 --frac_test=0.15 --time_gap=365 --years_back=-1 

#Create eye complication task. for dynamic model
python -u 4_create_subject_csv.py '/data/Thiti/data/Datasets 2020.05.12' '/data/Thiti/data/Eye_Complication/210404_Eye_Complication_1incradm_365timegap_15fracvaltest' -s=/data/Thiti/data/2_convert_events_to_timeseries/210403_2_convert_events_to_timeseries_1incradm --resources_folder='./resources' --comorbidity_info_file='./Prof_Code/config/comorbidity_eye.json' --number_processes=14 --frac_validation=0.15 --frac_test=0.15 --time_gap=365 --years_back=-1

#Create stroke ischaemic task. for dynamic model
python -u 4_create_subject_csv.py '/data/Thiti/data/Datasets 2020.05.12' '/data/Thiti/data/Stroke_Ischaemic/210404_Stroke_Ischaemic_1incradm_365timegap_15fracvaltest' -s=/data/Thiti/data/2_convert_events_to_timeseries/210403_2_convert_events_to_timeseries_1incradm --resources_folder='./resources' --comorbidity_info_file='./Prof_Code/config/comorbidity_ischaemic_stroke.json' --number_processes=60 --frac_validation=0.15 --frac_test=0.15 --time_gap=365 --years_back=-1 

##################################### prepare_dynamic_data.py #####################################
Turn the result from 4_create_subject_csv.py into dynamic data

#"subject_csv_dir" is the path of output from 4_create_subject_csv.py / "list_filepath" is the path where listfile is located. /
"saved_filename" is the path to save the output / "gzip_files" is to indicate whether or not to save as gzip file. / "timestep" is to how many days will be combined into single step.

# create dynamic data for eye complication using train_listfile
python -u prepare_dynamic_data.py --subject_csv_dir='/data/Thiti/data/Eye_Complication/210404_Eye_Complication_1incradm_365timegap_15fracvaltest' --list_filepath='/home/empower/Documents/Thiti/mycode/210416/Eye_Complication/dynamic/listfile/Eye_Complication_dynamic_data/train_listfile.csv' --saved_filename='/data/Thiti/data/Eye_Complication/210404_Eye_Complication_dynamic_1incradm_365timegap_15fracvaltest' --discretizer_filename='./resources/discretizer_config.json' --max_past_years=-1 --number_processes=14 --gzip_files=1 --timestep=10 

# create dynamic data for eye complication using test_listfile
python -u prepare_dynamic_data.py --subject_csv_dir='/data/Thiti/data/Eye_Complication/210404_Eye_Complication_1incradm_365timegap_15fracvaltest' --list_filepath='/home/empower/Documents/Thiti/mycode/210416/Eye_Complication/dynamic/listfile/Eye_Complication_dynamic_data/test_listfile.csv' --saved_filename='/data/Thiti/data/Eye_Complication/210404_Eye_Complication_dynamic_1incradm_365timegap_15fracvaltest' --discretizer_filename='./resources/discretizer_config.json' --max_past_years=-1 --number_processes=14 --gzip_files=1 --timestep=10 

# create dynamic data for eye complication using valid_listfile
python -u prepare_dynamic_data.py --subject_csv_dir='/data/Thiti/data/Eye_Complication/210404_Eye_Complication_1incradm_365timegap_15fracvaltest' --list_filepath='/home/empower/Documents/Thiti/mycode/210416/Eye_Complication/dynamic/listfile/Eye_Complication_dynamic_data/valid_listfile.csv' --saved_filename='/data/Thiti/data/Eye_Complication/210404_Eye_Complication_dynamic_1incradm_365timegap_15fracvaltest' --discretizer_filename='./resources/discretizer_config.json' --max_past_years=-1 --number_processes=14 --gzip_files=1 --timestep=10 

##################################### prepare_static_data.py #####################################
Refer to script.sh for how to execute

##################################### select_features_static_data.py #####################################
Note that inside "select_features_static_data.py".
line 16: NUM_COLS_TO_KEEP = [1000, 2500, 5000, 7500, 10000, 20000, 30000, 40000]

Can be adjusted to only keep the first 1000 and 2500 features by changing to  
NUM_COLS_TO_KEEP = [1000, 2500]

##################################### Utility #####################################
210404_listfile_adjust.py
TASK_FOLDER_NO_INCR=/path/to/1st_listfiles_or_source
TASK_FOLDER_INCR=/path/to/2nd_listfiles_or_target
LISTFILES_TEMP_FOLDER=/path/to/folder/to/keep/output
python -u 210404_listfile_adjust.py -s=$TASK_FOLDER_NO_INCR -t=$TASK_FOLDER_INCR -o=$LISTFILES_TEMP_FOLDER

output will be as follows
"${LISTFILES_TEMP_FOLDER}/src_test_listfile.csv" 
"${LISTFILES_TEMP_FOLDER}/src_train_listfile.csv" 
"${LISTFILES_TEMP_FOLDER}/src_valid_listfile.csv"

"${LISTFILES_TEMP_FOLDER}/target_test_listfile.csv"
"${LISTFILES_TEMP_FOLDER}/target_train_listfile.csv"
"${LISTFILES_TEMP_FOLDER}/target_valid_listfile.csv"

##################################### Problem #####################################
'CE_Physical Measurements (2013-2018).csv' should be included in the table of read_data