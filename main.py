# Author:   Jon-Paul Boyd
# Date:     16/01/2018
# IMAT5234  Applied Computational Intelligence - Mini Project
# Customer Relationship Management - Predict churn, appetency and upselling on Orange dataset (KDD Cup 2009)
# Main

import logging
import sys
from os import path
from preprocessor import Preprocessor
from filehandler import Filehandler
from modeller import Modeller

envparm = {'PrepEnabled': False, 'ProcessDS01': False, 'ProcessDS02': False, 'ProcessDS03': False,
           'PredictChurn': False, 'PredictAppetency': False, 'PredictUpselling': False, 'GridSearchSingleRFC': False,
           'GridSearchSingleDTC': False, 'GridSearchSingleABC': False, 'GridSearchSingleGBC': False,
           'GridSearchSingleBGC': False, 'GridSearchMultiRFC': False, 'GridSearchMultiDTC': False,
           'GridSearchMultiABC': False, 'GridSearchMultiGBC': False, 'GridSearchMultiBGC': False,
           'FinalRFC': False, 'FinalDTC': False, 'FinalABC': False, 'FinalGBC': False, 'FinalBGC': False,
           'FinalVTC': False, 'BaselineRFC': False, 'BaselineDTC': False, 'BaselineABC': False, 'BaselineGBC': False,
           'BaselineBGC': False,  'BaselineVTC': False, 'PlotGraphs': False}


def process_dataset(filehandler, modeller, dataset_path):
    # Churn
    if envparm['PredictChurn']:
        logging.info("Loading churn labels")
        churn = filehandler.read_csv(filehandler.l_churn_path, names=['Churn'])
        dataset = filehandler.read_csv(dataset_path)  # Reload dataset to ensure no scaling on scaling successive preds
        modeller.train_test_split(dataset, churn, 'Churn')
        if envparm['ProcessDS01']:
            modeller.standard_scaler()
        if envparm['BaselineDTC']:
            modeller.score_baseline_dtc(envparm)
        if envparm['BaselineRFC']:
            modeller.score_baseline_rfc(envparm, filehandler)
        if envparm['BaselineBGC']:
            modeller.score_baseline_bgc(envparm)
        if envparm['BaselineABC']:
            modeller.score_baseline_abc(envparm, filehandler)
        if envparm['BaselineGBC']:
            modeller.score_baseline_gbc(envparm)
        if envparm['BaselineVTC']:
            modeller.score_baseline_vtc(envparm)

        modeller.classifier_tune(envparm)

        if envparm['FinalDTC']:
            modeller.score_final_dtc_churn(envparm)
        if envparm['FinalRFC']:
            modeller.score_final_rfc_churn(envparm, filehandler)
        if envparm['FinalBGC']:
            modeller.score_final_bgc_churn(envparm)
        if envparm['FinalABC']:
            modeller.score_final_abc_churn(envparm, filehandler)
        if envparm['FinalGBC']:
            modeller.score_final_gbc_churn(envparm)
        if envparm['FinalVTC']:
            modeller.score_final_vtc_churn(envparm)

    # Appetency
    if envparm['PredictAppetency']:
        logging.info("Loading appetency labels")
        appetency = filehandler.read_csv(filehandler.l_appetency_path, names=['Appetency'])
        dataset = filehandler.read_csv(dataset_path)  # Reload dataset to ensure no scaling on scaling successive preds
        modeller.train_test_split(dataset, appetency, 'Appetency')
        if envparm['ProcessDS01']:
            modeller.standard_scaler()
        if envparm['BaselineDTC']:
            modeller.score_baseline_dtc(envparm)
        if envparm['BaselineRFC']:
            modeller.score_baseline_rfc(envparm, filehandler)
        if envparm['BaselineBGC']:
            modeller.score_baseline_bgc(envparm)
        if envparm['BaselineABC']:
            modeller.score_baseline_abc(envparm, filehandler)
        if envparm['BaselineGBC']:
            modeller.score_baseline_gbc(envparm)
        if envparm['BaselineVTC']:
            modeller.score_baseline_vtc(envparm)

        modeller.classifier_tune(envparm)

        if envparm['FinalDTC']:
            modeller.score_final_dtc_appetency(envparm)
        if envparm['FinalRFC']:
            modeller.score_final_rfc_appetency(envparm, filehandler)
        if envparm['FinalBGC']:
            modeller.score_final_bgc_appetency(envparm)
        if envparm['FinalABC']:
            modeller.score_final_abc_appetency(envparm, filehandler)
        if envparm['FinalGBC']:
            modeller.score_final_gbc_appetency(envparm)
        if envparm['FinalVTC']:
            modeller.score_final_vtc_appetency(envparm)

    # Upselling
    if envparm['PredictUpselling']:
        logging.info("Loading upselling labels")
        upselling = filehandler.read_csv(filehandler.l_upselling_path, names=['Upselling'])
        dataset = filehandler.read_csv(dataset_path)  # Reload dataset to ensure no scaling on scaling successive preds
        modeller.train_test_split(dataset, upselling, 'Upselling')
        if envparm['ProcessDS01']:
            modeller.standard_scaler()
        if envparm['BaselineDTC']:
            modeller.score_baseline_dtc(envparm)
        if envparm['BaselineRFC']:
            modeller.score_baseline_rfc(envparm, filehandler)
        if envparm['BaselineBGC']:
            modeller.score_baseline_bgc(envparm)
        if envparm['BaselineABC']:
            modeller.score_baseline_abc(envparm, filehandler)
        if envparm['BaselineGBC']:
            modeller.score_baseline_gbc(envparm)
        if envparm['BaselineVTC']:
            modeller.score_baseline_vtc(envparm)

        modeller.classifier_tune(envparm)

        if envparm['FinalDTC']:
            modeller.score_final_dtc_upselling(envparm)
        if envparm['FinalRFC']:
            modeller.score_final_rfc_upselling(envparm, filehandler)
        if envparm['FinalBGC']:
            modeller.score_final_bgc_upselling(envparm)
        if envparm['FinalABC']:
            modeller.score_final_abc_upselling(envparm, filehandler)
        if envparm['FinalGBC']:
            modeller.score_final_gbc_upselling(envparm)
        if envparm['FinalVTC']:
            modeller.score_final_vtc_upselling(envparm)


def main():
    filehandler = Filehandler()
    modeller = Modeller()

    if envparm['PrepEnabled']:
        logging.info("Executing preprocessor")
        Preprocessor(envparm)

    if envparm['ProcessDS01']:
        process_dataset(filehandler, modeller, filehandler.dataset_prep_path_01)

    if envparm['ProcessDS02']:
        process_dataset(filehandler, modeller, filehandler.dataset_prep_path_02)

    if envparm['ProcessDS03']:
        process_dataset(filehandler, modeller, filehandler.dataset_prep_path_03)

    if modeller.scores:
        modeller.output_scores(filehandler)


if __name__ == '__main__':
    import logging.config
    log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logging.conf')
    logging.config.fileConfig(log_file_path)
    if sys.argv[1] == 'PrepEnabled=1':
        envparm['PrepEnabled'] = True
    if sys.argv[2] == 'ProcessDS01=1':
        envparm['ProcessDS01'] = True
    if sys.argv[3] == 'ProcessDS02=1':
        envparm['ProcessDS02'] = True
    if sys.argv[4] == 'ProcessDS03=1':
        envparm['ProcessDS03'] = True
    if sys.argv[5] == 'PredictChurn=1':
        envparm['PredictChurn'] = True
    if sys.argv[6] == 'PredictAppetency=1':
        envparm['PredictAppetency'] = True
    if sys.argv[7] == 'PredictUpselling=1':
        envparm['PredictUpselling'] = True
    if sys.argv[8] == 'GridSearchSingleRFC=1':
        envparm['GridSearchSingleRFC'] = True
    if sys.argv[9] == 'GridSearchSingleDTC=1':
        envparm['GridSearchSingleDTC'] = True
    if sys.argv[10] == 'GridSearchSingleABC=1':
        envparm['GridSearchSingleABC'] = True
    if sys.argv[11] == 'GridSearchSingleGBC=1':
        envparm['GridSearchSingleGBC'] = True
    if sys.argv[12] == 'GridSearchSingleBGC=1':
        envparm['GridSearchSingleBGC'] = True
    if sys.argv[13] == 'GridSearchMultiRFC=1':
        envparm['GridSearchMultiRFC'] = True
    if sys.argv[14] == 'GridSearchMultiDTC=1':
        envparm['GridSearchMultiDTC'] = True
    if sys.argv[15] == 'GridSearchMultiABC=1':
        envparm['GridSearchMultiABC'] = True
    if sys.argv[16] == 'GridSearchMultiGBC=1':
        envparm['GridSearchMultiGBC'] = True
    if sys.argv[17] == 'GridSearchMultiBGC=1':
        envparm['GridSearchMultiBGC'] = True
    if sys.argv[18] == 'FinalRFC=1':
        envparm['FinalRFC'] = True
    if sys.argv[19] == 'FinalDTC=1':
        envparm['FinalDTC'] = True
    if sys.argv[20] == 'FinalABC=1':
        envparm['FinalABC'] = True
    if sys.argv[21] == 'FinalGBC=1':
        envparm['FinalGBC'] = True
    if sys.argv[22] == 'FinalBGC=1':
        envparm['FinalBGC'] = True
    if sys.argv[23] == 'FinalVTC=1':
        envparm['FinalVTC'] = True
    if sys.argv[24] == 'BaselineRFC=1':
        envparm['BaselineRFC'] = True
    if sys.argv[25] == 'BaselineDTC=1':
        envparm['BaselineDTC'] = True
    if sys.argv[26] == 'BaselineABC=1':
        envparm['BaselineABC'] = True
    if sys.argv[27] == 'BaselineGBC=1':
        envparm['BaselineGBC'] = True
    if sys.argv[28] == 'BaselineBGC=1':
        envparm['BaselineBGC'] = True
    if sys.argv[29] == 'BaselineVTC=1':
        envparm['BaselineVTC'] = True
    if sys.argv[30] == 'PlotGraphs=1':
        envparm['PlotGraphs'] = True

    logging.info("Runtime configuration")
    logging.info("Data preparation - {} {} {} {}".format(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))
    logging.info("Targets - {} {} {}".format(sys.argv[5],  sys.argv[6],  sys.argv[7]))
    logging.info("Grid search single - {} {} {} {} {}".format(sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11],
                                                           sys.argv[12]))
    logging.info("Grid search multi - {} {} {} {} {}".format(sys.argv[13], sys.argv[14], sys.argv[15], sys.argv[16],
                                                          sys.argv[17]))
    logging.info("Final score - {} {} {} {} {}".format(sys.argv[18], sys.argv[19], sys.argv[20], sys.argv[21],
                                                    sys.argv[22], sys.argv[23]))
    logging.info("Baseline score - {} {} {} {} {}".format(sys.argv[24], sys.argv[25], sys.argv[26], sys.argv[27],
                                                       sys.argv[28], sys.argv[29]))
    logging.info("Plot Graphs - {}".format(sys.argv[30]))
    main()
