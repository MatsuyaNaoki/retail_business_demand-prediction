import os
import sys
import argparse

import global_valiables
import Preprocessor
import modeler

inp_dir = '../data/'
out_dir = '../result/'
os.makedirs(out_dir, exist_ok=True)

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument('-b', '--beginDate', help='overwite learnData begin date', default='2018-01', type=str)
    parser.add_argument('-e', '--endDate', help='overwite learnData end date', default='2019-10', type=str)
    parser.add_argument('-p', '--purposeDate', help='overwite testData purpose date', default='2019-12', type=str)
    parser.add_argument('-a', '--algorithm', help='specify prediction algorithm', default='lightgbm', type=str)
    parser.add_argument('-o', '--optuna', help='specify optuna n_trials', default=0, type=int)
    parser.add_argument('--verbose', help='verbose mode', action='store_true')
    parser.add_argument('--save', help='save mode', action='store_true')

    args = parser.parse_args()

    modeltype = args.algorithm
    beginDate = args.beginDate
    endDate = args.endDate
    purposeDate = args.purposeDate
    n_trials = args.optuna
    global_valiables.verbose = args.verbose
    global_valiables.save = args.save

    prep = Preprocessor.Preprocessor()

    print('algorithm: {}'.format(modeltype))

    prep.load(inp_dir)
    X_train, y_train, X_test = prep.createData(beginDate, endDate, purposeDate)
    modeler.setup(modeltype, X_train, y_train, X_test)
    modeler.experiment(n_trials)
    pred = modeler.predict()

    if global_valiables.save:
        df_result = X_test.copy()
        df_result.insert(0, 'pred_売上個数', pred)
        df_result.to_csv(out_dir+'result.csv', encoding='shift-jis')
        df_result['pred_売上個数'].to_csv(out_dir+'result_for_submit.csv')

if __name__ == '__main__':
    main(sys.argv[1:])