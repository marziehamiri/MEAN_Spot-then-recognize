import tensorflow.compat.v1 as tf
import os
from tensorflow import keras
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import LeaveOneGroupOut
from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d
from numpy import argmax
import time

from training_utils import *
from define_model import *
from sklearn.metrics import accuracy_score


random.seed(1)
tf.disable_v2_behavior()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def train_test_single(
        X, y,
        X1, y1,
        X2, y2,
        spot_train_idx, spot_test_idx,
        recog_train_idx, recog_test_idx,
        dataset_name,
        emotion_class,
        k, k_p,
        spot_weight_path=None,
        recog_weight_path=None,
        spot_recog_weight_path=None,
        epochs_spot=30,
        epochs_recog=200,
        batch_size=32,
        ratio=5,
        train=False
    ):

    # -------- Spotting split --------
    X_train = [X[i] for i in spot_train_idx]
    X_test  = [X[i] for i in spot_test_idx]
    y_train = [y[i] for i in spot_train_idx]
    y_test  = [y[i] for i in spot_test_idx]

    # -------- Recognition split --------
    X2_train = [X2[i] for i in recog_train_idx]
    y2_train = [y2[i] for i in recog_train_idx]
    X2_test  = [X2[i] for i in recog_test_idx]
    y2_test  = [y2[i] for i in recog_test_idx]

    # -------- Downsampling (فقط اگر train) --------
    if train:
        unique, counts = np.unique(y_train, return_counts=True)
        rem_count = int(min(counts) * ratio)
        idx = random.sample(range(len(y_train)), rem_count)
        X_train = [X_train[i] for i in idx]
        y_train = [y_train[i] for i in idx]

    # -------- Prepare Spotting input --------
    X_train = [np.array(X_train)[:,0], np.array(X_train)[:,1], np.array(X_train)[:,2]]
    X_test  = [np.array(X_test)[:,0],  np.array(X_test)[:,1],  np.array(X_test)[:,2]]

    # -------- Spotting model --------
    model_spot = MEAN_Spot()
    if not train and spot_weight_path is not None:
        model_spot.load_weights(spot_weight_path)
    elif train:
        model_spot.fit(X_train, np.array(y_train),
                       epochs=epochs_spot,
                       batch_size=batch_size,
                       verbose=0)

    # -------- Recognition model --------
    adam = keras.optimizers.Adam()
    model_recog = MEAN_Recog_TL(model_spot, adam, emotion_class)
    if not train and recog_weight_path is not None:
        model_recog.load_weights(recog_weight_path)
    elif train and len(X2_train) > 0:
        X2_train = [np.array(X2_train)[:,0], np.array(X2_train)[:,1], np.array(X2_train)[:,2]]
        model_recog.fit(X2_train, np.array(y2_train),
                        epochs=epochs_recog,
                        batch_size=batch_size,
                        verbose=0)

    # -------- Spot + Recog model --------
    model_final = MEAN_Spot_Recog_TL(model_spot, model_recog, adam)
    if not train and spot_recog_weight_path is not None:
        model_final.load_weights(spot_recog_weight_path)

    results = model_final.predict(X_test, verbose=0)

    return {
        'spot_prob': results[0],
        'recog_prob': results[1],
        'y_test': y_test,
        'y2_test': y2_test
    }


def train_test_late_fusion(
        face_data,
        brow_data,
        groupsLabel,
        groupsLabel1,
        dataset_name,
        emotion_class,
        final_subjects,
        final_emotions,
        final_samples,
        final_dataset_spotting,
        k, k_p,
        expression_type,
        result_merge_micro,
        result_merge_micro_2,
        result_merge_macro,
        resul_merge_micro_gt,
        spot_multiple,
        alpha=0.7,
        beta=0.3,
        train=False
    ):


    loso = LeaveOneGroupOut()
    subject_count = 0

    spot_index = list(loso.split(
    face_data['X'],
    face_data['y'],
    groupsLabel
))

    recog_index = list(loso.split(
    face_data['X1'],
    face_data['y1'],
    groupsLabel1
))


    metric_final = MeanAveragePrecision2d(num_classes=1)
    total_gt_spot = 0
    pred_spot_list, gt_spot_list = [], []
    pred_list, gt_tp_list = [], []

    for subject_index in range(len(final_subjects)):
        subject_count += 1

        # ---------- Face ----------
        out_face = train_test_single(
            **face_data,
            spot_train_idx=spot_index[subject_index][0],
            spot_test_idx=spot_index[subject_index][1],
            recog_train_idx=recog_index[subject_index][0],
            recog_test_idx=recog_index[subject_index][1],
            dataset_name=dataset_name,
            emotion_class=emotion_class,
            k=k, k_p=k_p,
            spot_weight_path='/content/drive/MyDrive/Mean/MEAN_Weights/' + dataset_name + '/' + 'spot'+ '/s' + str(subject_count) + '.hdf5'
,
            recog_weight_path='/content/drive/MyDrive/Mean/MEAN_Weights/' + dataset_name + '/' + 'recog' + '/s' + str(subject_count) + '.hdf5',
            spot_recog_weight_path='/content/drive/MyDrive/Mean/MEAN_Weights/' + dataset_name + '/' + 'spot_recog' + '/s' + str(subject_count) + '.hdf5',
            train=train
        )

        # ---------- Eyebrow ----------
        out_brow = train_test_single(
            **brow_data,
            spot_train_idx=spot_index[subject_index][0],
            spot_test_idx=spot_index[subject_index][1],
            recog_train_idx=recog_index[subject_index][0],
            recog_test_idx=recog_index[subject_index][1],
            dataset_name=dataset_name,
            emotion_class=emotion_class,
            k=k, k_p=k_p,
            spot_weight_path='/content/drive/MyDrive/Mean/MEAN_Weights/eyebrows/' + dataset_name + '/' + 'spot'+ '/s' + str(subject_count) + '.hdf5'
,
            recog_weight_path='/content/drive/MyDrive/Mean/MEAN_Weights/eyebrows/' + dataset_name + '/' + 'recog' + '/s' + str(subject_count) + '.hdf5'
,
            spot_recog_weight_path= '/content/drive/MyDrive/Mean/MEAN_Weights/eyebrows/' + dataset_name + '/' + 'spot_recog' + '/s' + str(subject_count) + '.hdf5'
,
            train=train
        )

        # ---------- Late Fusion ----------
        spot_prob_fused  = 80 * out_face['spot_prob']  + 20 * out_brow['spot_prob']
        recog_prob_fused = 70 * out_face['recog_prob'] + 30 * out_brow['recog_prob']

        # ---------- Spotting ----------
        preds, gt, total_gt_spot, metric_video, metric_final = spotting(
            spot_prob_fused,
            total_gt_spot,
            subject_count,
            0.55,
            metric_final,
            spot_multiple,
            k_p,
            final_samples,
            final_dataset_spotting,
            result_merge_micro,
            result_merge_micro_2,
            result_merge_macro,
            resul_merge_micro_gt
        )

        pred_spot_list.extend(preds)
        gt_spot_list.extend(gt)

        # ---------- Recognition ----------
        pred_list, gt_tp_list, _, _ = recognition(
            dataset_name,
            emotion_class,
            recog_prob_fused,
            preds,
            metric_video,
            final_emotions,
            subject_count,
            pred_list,
            gt_tp_list,
            out_face['y_test'],
            final_samples,
            [], [], spot_multiple, k, k_p,
            final_dataset_spotting
        )

    TP, FP, FN = sequence_evaluation(total_gt_spot, metric_final)
    asr_score, mae_score = apex_evaluation(pred_spot_list, gt_spot_list, k_p)

    return TP, FP, FN, metric_final, gt_tp_list, pred_list, asr_score, mae_score


def final_evaluation(TP_spot, FP_spot, FN_spot, dataset_name, expression_type, metric_final, asr_score, mae_score, spot_multiple, pred_list, gt_list, emotion_class, gt_tp_list):
    #Spotting
    precision = TP_spot/(TP_spot+FP_spot)
    recall = TP_spot/(TP_spot+FN_spot)
    F1_score = (2 * precision * recall) / (precision + recall)
    print('----Spotting----')
    print('Final Result for', dataset_name)
    print('TP:', TP_spot, 'FP:', FP_spot, 'FN:', FN_spot)
    print('Precision = ', round(precision, 4))
    print('Recall = ', round(recall, 4))
    print('F1-Score = ', round(F1_score, 4))
    print("COCO AP@[.5:.95]:", round(metric_final.value(iou_thresholds=np.round(np.arange(0.5, 1.0, 0.05), 2), mpolicy='soft')['mAP'], 4))
    print('ASR = ', round(asr_score, 4))
    print('MAE = ', round(mae_score, 4))

    #Check recognition accuracy if only correctly predicted spotting are considered
    #if(not spot_multiple):
    print('\n----Recognition (All)----')
    print('Predicted    :', pred_list)
    print('Ground Truth :', gt_tp_list)
    UF1, UAR = recognition_evaluation(dataset_name, emotion_class, gt_tp_list, pred_list, show=True)
    print('Accuracy Score:', round(accuracy_score(gt_tp_list, pred_list), 4))

    print('\n----Recognition (Consider TP only)----')
    gt_tp_spot = []
    pred_tp_spot = []
    for index in range(len(gt_tp_list)):
        if(gt_tp_list[index]!=-1):
            gt_tp_spot.append(gt_tp_list[index])
            pred_tp_spot.append(pred_list[index])
    print('Predicted    :', pred_tp_spot)
    print('Ground Truth :', gt_tp_spot)
    UF1, UAR = recognition_evaluation(dataset_name, emotion_class, gt_tp_spot, pred_tp_spot, show=True)
    print('Accuracy Score:', round(accuracy_score(gt_tp_spot, pred_tp_spot), 4))