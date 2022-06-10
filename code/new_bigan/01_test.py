########################################################################
# import default libraries
########################################################################
import os
import csv
import sys
import gc
########################################################################


########################################################################
# import additional libraries
from sklearn.neighbors import LocalOutlierFactor
########################################################################
import numpy as np
import scipy.stats
# from import
from tqdm import tqdm
from sklearn import metrics
try:
    from sklearn.externals import joblib
except:
    import joblib
# original lib
import common as com
import keras_model
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
#######################################################################


########################################################################
# output csv file
########################################################################
def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


########################################################################


########################################################################
# main 01_test.py
#######################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    # make output result directory
    os.makedirs(param["result_directory"], exist_ok=True)

    # load base directory
    dirs = com.select_dirs(param=param, mode=mode)

    # initialize lines in csv for AUC and pAUC
    csv_lines = []

    if mode:
        performance_over_all = []

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {target_dir}".format(target_dir=target_dir, idx=idx+1, total=len(dirs)))
        machine_type = os.path.split(target_dir)[1]

        print("============== MODEL LOAD ==============")
        # # load model file
        # model_file = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory"],
        #                                                         machine_type=machine_type)
        # if not os.path.exists(model_file):
        #     com.logger.error("{} model not found ".format(machine_type))
        #     sys.exit(-1)
        # model = keras_model.load_model(model_file)
        # model.summary()

        # # load section names for conditioning
        # section_names_file_path = "{model}/section_names_{machine_type}.pkl".format(model=param["model_directory"],
        #                                                                   machine_type=machine_type)
        # trained_section_names = joblib.load(section_names_file_path)
        # n_sections = trained_section_names.shape[0]

        # # load anomaly score distribution for determining threshold
        # score_distr_file_path = "{model}/score_distr_{machine_type}.pkl".format(model=param["model_directory"],
        #                                                             machine_type=machine_type)
        # shape_hat, loc_hat, scale_hat = joblib.load(score_distr_file_path)

        # # determine threshold for decision
        # decision_threshold = scipy.stats.gamma.ppf(q=param["decision_threshold"], a=shape_hat, loc=loc_hat, scale=scale_hat)

        # if mode:
        #     # results for each machine type
        #     csv_lines.append([machine_type])
        #     csv_lines.append(["", "AUC (source)", "AUC (target)", "pAUC",
        #                     "precision (source)", "precision (target)", "recall (source)", "recall (target)",
        #                     "F1 score (source)", "F1 score (target)"])
        #     performance = []

        dir_name = "test"

        #list machine id
        section_names = com.get_section_names(target_dir, dir_name)

        for section_name in section_names:

            #search for section_name
            #if the section_name is not found in the trained_section_names, store -1 in section_idx
            # temp_array = np.nonzero(trained_section_names == section_name)[0]
            # if temp_array.shape[0] == 0:
            #     section_idx = -1
            # else:
            #     section_idx = temp_array[0]

            # load test file
            files, y_true = com.file_list_generator(target_dir=target_dir,
                                                    section_name=section_name,
                                                    dir_name=dir_name,
                                                    mode=mode)
            print(f"len files : {len(files)}")
            print(f"y_true : {y_true.shape}")

            # setup anomaly score file path
            anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{section_name}_{dir_name}.csv".format(result=param["result_directory"],
                                                                                                                machine_type=machine_type,
                                                                                                                section_name=section_name,
                                                                                                                dir_name=dir_name)
            anomaly_score_list = []

            # setup decision result file path
            decision_result_csv = "{result}/decision_result_{machine_type}_{section_name}_{dir_name}.csv".format(result=param["result_directory"],
                                                                                                                    machine_type=machine_type,
                                                                                                                    section_name=section_name,
                                                                                                                    dir_name=dir_name)
            decision_result_list = []

            if mode:
                domain_list = []

            print("\n============== LOAD TEST DATA ==============")
            y_pred = [0. for k in files]
            test = []
            for file_idx, file_path in tqdm(enumerate(files), total=len(files)):
                try:
                    data = com.file_to_vectors(file_path,
                                                    n_mels=param["feature"]["n_mels"],
                                                    n_frames=param["feature"]["n_frames"],
                                                    n_fft=param["feature"]["n_fft"],
                                                    hop_length=param["feature"]["hop_length"],
                                                    power=param["feature"]["power"])
                    # print(f"data 1 = {data.shape}")
                    data = data[: : 8, :]
                    # print(data.shape)
                    test.append(data)
                    # np.save("test.npy", data)
                except:
                    com.logger.error("File broken!!: {}".format(file_path))


                # make one-hot vector for conditioning
                #condition = np.zeros((data.shape[0], n_sections), float)
                # if the id_name was found in the trained_section_names, make a one-hot vector
                #if section_idx != -1:
                    #condition[:, section_idx : section_idx + 1] = 1

                # 1D vector to 2D image

                #data = data.reshape(data.shape[0], param["feature"]["n_frames"], param["feature"]["n_mels"], 1)
                # print(f"data 2 = {data.shape}")
            test = np.array(test)
            # test = test.reshape(len(files), 2048, 128)
            print(f"test shape : {test.shape}")

            # test = test.reshape(200, 64 * 128 * 1)
            # print(test.shape)
            # print(f"y_true : {y_true.shape}")
            # np.save("test.npy", test)
# ==============================================================================================
            source_x_test_normal = test[0:50, :]
            source_x_test_normal = source_x_test_normal.reshape(50*32, 64*128)
            print(source_x_test_normal.shape)
            source_x_test_anomaly = test[100:150, :]
            source_x_test_anomaly = source_x_test_anomaly.reshape(50*32, 64*128)

            target_x_test_normal = test[50:100, :]
            target_x_test_normal = target_x_test_normal.reshape(50*32, 64*128)
            target_x_test_anomaly = test[150:, :]
            target_x_test_anomaly = target_x_test_anomaly.reshape(50*32, 64*128)

            source_x_test = np.concatenate((source_x_test_normal, source_x_test_anomaly), axis = 0)
            target_x_test = np.concatenate((target_x_test_normal, target_x_test_anomaly), axis = 0)
            np.save(f"/run/media/viblab/Markov1/Bagus Adhi/bigan_dcase/data/test/x_test_{machine_type}_{section_name}_source.npy", source_x_test)
            np.save(f"/run/media/viblab/Markov1/Bagus Adhi/bigan_dcase/data/test/x_test_{machine_type}_{section_name}_target.npy", target_x_test)

# ===================================================================================================================

            #source_y_test_normal = y_true[0:50]
            #source_y_test_anomaly = y_true[100:150]

            #target_y_test_normal = y_true[50:100]
            #target_y_test_anomaly = y_true[150:]

            #source_y_test = np.concatenate((source_y_test_normal, source_y_test_anomaly), axis = 0)
            #target_y_test = np.concatenate((target_y_test_normal, target_y_test_anomaly), axis = 0)
            #np.save(f"/run/media/viblab/Markov1/Bagus Adhi/bigan_dcase/data/test/y_test_{machine_type}_{section_name}_source.npy", source_y_test)
            #np.save(f"/run/media/viblab/Markov1/Bagus Adhi/bigan_dcase/data/test/y_test_{machine_type}_{section_name}_target.npy", target_y_test)

#====================================================================================================================
            source_y_test_normal = np.zeros((50*32), float)
            source_y_test_anomaly = np.ones((50*32), float)

            target_y_test_normal = np.zeros((50*32), float)
            target_y_test_anomaly = np.ones((50*32), float)


            source_y_test = np.concatenate((source_y_test_normal, source_y_test_anomaly), axis = 0)
            target_y_test = np.concatenate((target_y_test_normal, target_y_test_anomaly), axis = 0)
            np.save(f"/run/media/viblab/Markov1/Bagus Adhi/bigan_dcase/data/test/y_test_{machine_type}_{section_name}_source.npy", source_y_test)
            np.save(f"/run/media/viblab/Markov1/Bagus Adhi/bigan_dcase/data/test/y_test_{machine_type}_{section_name}_target.npy", target_y_test)
