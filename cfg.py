import platform

SERVER_ROUNDS = 10
WORKER_1_RNDS = 1
WORKER_2_RNDS = 1
HETEROGENEITY = 'discrete'
BACKEND = 'mps'

ACC_FILE_NAME = f'{SERVER_ROUNDS}-{WORKER_1_RNDS}-{WORKER_2_RNDS}-{HETEROGENEITY}.csv'
LOSS_FILE_NAME = f'LOSS-{SERVER_ROUNDS}-{WORKER_1_RNDS}-{WORKER_2_RNDS}-{HETEROGENEITY}.csv'

ds1 = {'Windows':{'moderate':'../tpd_150k_b1357.csv',
                  'discrete':'../tpd_discrete_1357.csv',
                  'test':'../tpd_20k_test.csv'},
        'Darwin':{'moderate':'../tpd_150k_b1357.csv',
                  'discrete':'../tpd_discrete_1357.csv',
                  'test':'../tpd_20k_test.csv'},
        'Linux':{'moderate':'../tpd_150k_b1357.csv',
                 'discrete':'../tpd_discrete_1357.csv',
                 'test':'../tpd_20k_test.csv'},
            }

ds2 = {'Windows':{'moderate':'../tpd_150k_b2468.csv',
                  'discrete':'../tpd_discrete_02468.csv',
                  'test':'../tpd_20k_test.csv'},
        'Darwin':{'moderate':'../tpd_150k_b2468.csv',
                  'discrete':'../tpd_discrete_02468.csv',
                  'test':'../tpd_20k_test.csv'},
        'Linux':{'moderate':'../tpd_150k_b2468.csv',
                 'discrete':'../tpd_discrete_02468.csv',
                 'test':'../tpd_20k_test.csv'},
            }

WORKER_1_DS = ds1[platform.system()][HETEROGENEITY]
WORKER_2_DS = ds2[platform.system()][HETEROGENEITY]
TEST_DS = ds1[platform.system()]['test']
