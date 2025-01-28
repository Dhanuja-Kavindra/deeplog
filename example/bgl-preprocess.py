import os
import sys
import logging
import os
import sys
import logging
import pandas as pd
from spellpy import spell

logging.basicConfig(level=logging.WARNING, format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

def bgl_log_df_transfer(df, event_id_map):
    # Combine date and time into a single datetime column
    df['datetime'] = pd.to_datetime(df['Epoch'], unit='s')
    #df['datetime'] = pd.to_datetime(df['FullDatetime'], format='%Y-%m-%d-%H.%M.%S.%f')
    df = df[['datetime', 'EventId']]
    df['EventId'] = df['EventId'].apply(lambda e: event_id_map[e] if event_id_map.get(e) else -1)
    bgl_df = df.set_index('datetime').resample('1min').apply(_custom_resampler).reset_index()
    return bgl_df

def _custom_resampler(array_like):
    return list(array_like)

def bgl_file_generator(filename, df):
    with open(filename, 'w') as f:
        for event_id_list in df['EventId']:
            for event_id in event_id_list:
                f.write(str(event_id) + ' ')
            f.write('\n')

if __name__ == '__main__':
    ##########
    # Parser #
    ##########
    input_dir = '/content/drive/My Drive/DeepLog/BGL/'
    output_dir = './bgl_result/'
    log_format = '<LogType> <Epoch> <Date> <Node> <FullDatetime> <NodeRepeat> <Type> <Component> <Severity> <Content>'
    log_main = 'bgl_logs'
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

    for log_name in ['bgl_abnormal.log', 'bgl_normal2.log', 'bgl_normal1.log']:
        parser.parse(log_name)

    ##################
    # Transformation #
    ##################
    df = pd.read_csv(f'{output_dir}/bgl_normal1.log_structured.csv')
    df_normal = pd.read_csv(f'{output_dir}/bgl_normal2.log_structured.csv')
    df_abnormal = pd.read_csv(f'{output_dir}/bgl_abnormal.log_structured.csv')
    combined_df = pd.concat([df, df_normal, df_abnormal])

    event_id_map = dict()
    for i, event_id in enumerate(combined_df['EventId'].unique(), 1):
        event_id_map[event_id] = i

    logger.info(f'length of event_id_map: {len(event_id_map)}')

    #########
    # Train #
    #########
    bgl_train = bgl_log_df_transfer(df, event_id_map)
    bgl_file_generator('train', bgl_train)

    ###############
    # Test Normal #
    ###############
    bgl_test_normal = bgl_log_df_transfer(df_normal, event_id_map)
    bgl_file_generator('test_normal', bgl_test_normal)

    #################
    # Test Abnormal #
    #################
    bgl_test_abnormal = bgl_log_df_transfer(df_abnormal, event_id_map)
    bgl_file_generator('test_abnormal', bgl_test_abnormal)
