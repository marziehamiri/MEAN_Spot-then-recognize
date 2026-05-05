import sys
import argparse
from face_crop import *
from load_images import *
from load_label import *
from load_excel import *
from feature_extraction import *
from feature_extraction2 import *
from prepare_training import *
from train_evaluate import *

from distutils.util import strtobool

## Note that the whole process will take a long time... please be patient
def main(config):

    # Define the dataset and expression to spot
    dataset_name = config.dataset_name
    train = config.train

    print(' ------ Spot-then-recognize m', dataset_name, '-------')
    
    # Load Images
    print('\n ------ Face detection and Croping Images ------')
    #face_crop(dataset_name) # Can comment this out after completed on the dataset specified and intend to try on another expression_type
    print("\n ------ Loading Images ------")
    images, subjects, subjectsVideos = load_images(dataset_name)
    
    # Load Ground Truth Label
    print('\n ------ Loading Excel ------')
    codeFinal = load_excel(dataset_name)
    print('\n ------ Loading Ground Truth From Excel ------')
    final_images, final_subjects, final_videos, final_samples, final_emotions = load_label(dataset_name, images, subjects, subjectsVideos, codeFinal) 

    #micro_filtering
    final_images_macro, final_subjects_macro, final_videos_macro, final_samples_macro, final_emotions_macro = load_label_macro(dataset_name, images, subjects, subjectsVideos, codeFinal) 
    
    result_merge_micro = merge_lists(final_subjects, final_videos)
    result_merge_micro_2 = merge_lists(final_subjects, final_videos)
    resul_merge_micro_gt = merge_lists(final_subjects, final_videos,final_samples)
    result_merge_macro = merge_lists( final_subjects_macro,final_videos_macro, final_samples_macro)
    

    # Set Parameters
    print('\n ------ Set k ------')
    k = set_k(dataset_name)
    print('\n ------ Set Emotion Class ------')
    emotion_class = set_emotion_class(dataset_name)
    print('\n ------ Set k_p ------') # k'
    k_p = cal_k_p(dataset_name, final_samples)

    # Feature Extraction & Pre-processing
    print('\n ------ Recognition Feature Extraction & Pre-processing ------')
    final_dataset_recognition = feature_extraction_recognition(dataset_name, final_images, final_samples)
    final_dataset_recognition2 = feature_extraction_recognition2(dataset_name, final_images, final_samples)
    

    print('\n ------ Spotting Feature Extraction & Pre-processing ------')
    final_dataset_spotting = feature_extraction_spotting(dataset_name, final_images, k)
    final_dataset_spotting2 = feature_extraction_spotting2(dataset_name, final_images, k)

    # Spotting Pseudo-labeling
    print('\n ------ Spotting Pseudo-Labeling ------')
    pseudo_y = spotting_pseudo_labeling(dataset_name, final_samples, final_dataset_spotting, k_p)
    
    # Recognition labeling
    print('\n ------ Recognition Pseudo-Labeling ------')
    spot_multiple, X, y, X1, y1, X2, y2, emotion_list = recognition_label(dataset_name, emotion_class, final_samples, final_emotions, final_dataset_spotting, final_dataset_recognition, pseudo_y)
    spot_multiple, Xx, yy, Xx1, yy1, Xx2, yy2, emotion_list = recognition_label(dataset_name, emotion_class, final_samples, final_emotions, final_dataset_spotting2, final_dataset_recognition2, pseudo_y)
    

    # LOSO splitting
    print('\n ------ Leave one Subject Out ------')
    groupsLabel, groupsLabel1 = loso_split(X, y, X1, y1, X2, y2, final_subjects, final_samples, final_dataset_spotting, final_emotions, emotion_list)
    groupsLabel2, groupsLabel3 = loso_split(Xx, yy, Xx1, yy1, Xx2, yy2, final_subjects, final_samples, final_dataset_spotting2, final_emotions, emotion_list)

    # Create directory if not exist
    create_directory(train, dataset_name)

    # Model Training & Evaluation
    print('\n ------ MEAN Training & Testing ------')
    #TP_spot, FP_spot, FN_spot, metric_final, gt_list, pred_list, gt_tp_list, asr_score, mae_score = train_test(X, y, X1, y1, X2, y2, dataset_name, emotion_class, groupsLabel, groupsLabel1, spot_multiple, final_subjects, final_emotions, final_samples, final_dataset_spotting, k, k_p, 'micro-expression',result_merge_micro,result_merge_micro_2,result_merge_macro,resul_merge_micro_gt, epochs_spot=30, epochs_recog=200, spot_lr=0.0005, recog_lr=0.0005, batch_size=32, ratio=5, p=0.55, spot_attempt=1, recog_attempt=1, train=train)
    #TP_spot, FP_spot, FN_spot, metric_final, gt_list, pred_list, gt_tp_list, asr_score, mae_score = train_test(Xx, yy, Xx1, yy1, Xx2, yy2, dataset_name, emotion_class, groupsLabel2, groupsLabel3, spot_multiple, final_subjects, final_emotions, final_samples, final_dataset_spotting2, k, k_p, 'micro-expression',result_merge_micro,result_merge_micro_2,result_merge_macro,resul_merge_micro_gt,X, y, X1, y1, X2, y2,groupsLabel, groupsLabel1,final_dataset_spotting, epochs_spot=30, epochs_recog=200, spot_lr=0.0005, recog_lr=0.0005, batch_size=32, ratio=5, p=0.55, spot_attempt=1, recog_attempt=1, train=train)
    face_data = {
    'X': X, 'y': y,
    'X1': X1, 'y1': y1,
    'X2': X2, 'y2': y2
    
}

    brow_data = {
    'X': Xx, 'y': yy,
    'X1': Xx1, 'y1': yy1,
    'X2': Xx2, 'y2': yy2
    
}

    TP, FP, FN, metric_final, gt_tp, pred, asr, mae = train_test_late_fusion(
    face_data=face_data,
    brow_data=brow_data,
    groupsLabel=groupsLabel,
    groupsLabel1=groupsLabel1,
    dataset_name=dataset_name,
    emotion_class=4,
    final_subjects=final_subjects,
    final_emotions=final_emotions,
    final_samples=final_samples,
    final_dataset_spotting=final_dataset_spotting,
    k=3,
    k_p=6,
    expression_type='micro',
    result_merge_micro=result_merge_micro,
    result_merge_micro_2=result_merge_micro_2,
    result_merge_macro=result_merge_macro,
    resul_merge_micro_gt=resul_merge_micro_gt,
    spot_multiple=True,
    train=False
)

    # Model Final Evaluation
    print('\n ------ MEAN Final Evaluation ------')
    gt_list = []
    final_evaluation(TP, FP, FN, dataset_name, 'micro-expression', metric_final, asr, mae, spot_multiple, pred, gt_list, emotion_class, gt_tp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--dataset_name', type=str, default='CASME_sq') 
    parser.add_argument('--train', type=strtobool, default=False) #Train or use pre-trained weight for prediction
    
    config = parser.parse_args()

    main(config)