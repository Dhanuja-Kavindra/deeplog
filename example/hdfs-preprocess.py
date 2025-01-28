import os
import sys
import logging
import pandas as pd
from spellpy import spell

logging.basicConfig(level=logging.WARNING, format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

def hdfs_log_df_transfer(df, event_id_map, dataset_name):
    # Log the initial number of rows
    initial_rows = len(df)

    # Convert date and time to datetime, coercing errors
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    
    # Identify invalid rows
    invalid_rows = df[df['datetime'].isna()]
    num_invalid_rows = len(invalid_rows)

    # Log invalid rows for this dataset
    if num_invalid_rows > 0:
        logger.warning(f"Dataset: {dataset_name} | Invalid datetime rows: {num_invalid_rows}")
        logger.debug(f"Invalid rows from {dataset_name}:\n{invalid_rows[['Date', 'Time']]}")
    else:
        logger.info(f"Dataset: {dataset_name} | No invalid datetime rows.")

    # Drop rows with NaT in the datetime column
    df_cleaned = df.dropna(subset=['datetime'])
    removed_rows = initial_rows - len(df_cleaned)

    # Filter and map EventId
    df_cleaned = df_cleaned.loc[:, ['datetime', 'EventId']]  # Explicitly select columns
    df_cleaned.loc[:, 'EventId'] = df_cleaned['EventId'].apply(lambda e: event_id_map[e] if event_id_map.get(e) else -1)
    
    # Resample and process
    hdfs_df = df_cleaned.set_index('datetime').resample('1min').apply(_custom_resampler).reset_index()
    return hdfs_df


def _custom_resampler(array_like):
    return list(array_like)

def hdfs_file_generator(filename, df):
    with open(filename, 'w') as f:
        for event_id_list in df['EventId']:
            for event_id in event_id_list:
                f.write(str(event_id) + ' ')
            f.write('\n')

if __name__ == '__main__':
    ##########
    # Parser #
    ##########
    input_dir = './data/HDFS/'
    output_dir = './hdfs_result/'
    log_format = '<Date>,<Time>,<Pid>,<Level>,<Source>,<Content>,<BlockId>,<Label>,<EventId>'
    log_main = 'hdfs_logs'
    tau = 0.5

    parser = spell.LogParser(
        indir=input_dir,
        outdir=output_dir,
        log_format=log_format,
        logmain=log_main,
        tau=tau,
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for log_name in ['hdfs_abnormal.log', 'hdfs_normal2.log', 'hdfs_normal1.log']:
        parser.parse(log_name)

    ##################
    # Transformation #
    ##################
    df = pd.read_csv(f'{output_dir}/hdfs_normal1.log_structured.csv')
    df_normal = pd.read_csv(f'{output_dir}/hdfs_normal2.log_structured.csv')
    df_abnormal = pd.read_csv(f'{output_dir}/hdfs_abnormal.log_structured.csv')

    event_id_map = dict()
    for i, event_id in enumerate(df['EventId'].unique(), 1):
        event_id_map[event_id] = i

    logger.info(f'length of event_id_map: {len(event_id_map)}')

    #########
    # Train #
    #########
    hdfs_train = hdfs_log_df_transfer(df, event_id_map, df)
    hdfs_file_generator('train', hdfs_train)

    ###############
    # Test Normal #
    ###############
    hdfs_test_normal = hdfs_log_df_transfer(df_normal, event_id_map, df_normal)
    hdfs_file_generator('test_normal', hdfs_test_normal)

    #################
    # Test Abnormal #
    #################
    hdfs_test_abnormal = hdfs_log_df_transfer(df_abnormal, event_id_map, df_abnormal)
    hdfs_file_generator('test_abnormal', hdfs_test_abnormal)
