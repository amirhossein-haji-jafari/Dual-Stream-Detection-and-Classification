experiments = [
    {
        'name': 'A_synergistic_original_nosmoothing',
        'fusion': 'synergistic',
        'data': 'original',
        'smoothing': False,
        'test_metrics': {
            'confusion_matrix': {
                'TN': 26,
                'FP': 8,
                'FN': 13,
                'TP': 59,
            },
            'accuracy': 0.8019,
            'precision': 0.8806,
            'recall': 0.8194,
            'f1_score': 0.8489,
            'auc': 0.7982,
            'npv': 0.6667,
            'misclassified_images': ['P107_R_CM_CC_resized.jpg', 'P299_R_CM_MLO_resized.jpg', 'P103_R_CM_MLO_resized.jpg',
                                     'P301_L_CM_MLO_resized.jpg', 'P320_L_CM_CC_resized.jpg', 'P145_L_CM_MLO_resized.jpg',
                                     'P145_L_CM_CC_resized.jpg', 'P99_R_CM_MLO_resized.jpg', 'P298_L_CM_CC_resized.jpg',
                                     'P265_R_CM_CC_resized.jpg', 'P298_L_CM_MLO_resized.jpg', 'P64_L_CM_CC_resized.jpg',
                                     'P98_L_CM_MLO_resized.jpg', 'P91_L_CM_MLO_resized.jpg', 'P83_L_CM_MLO_resized.jpg',
                                     'P98_L_CM_CC_resized.jpg', 'P301_L_CM_CC_resized.jpg', 'P118_L_CM_CC_resized.jpg',
                                     'P320_L_CM_MLO_resized.jpg', 'P83_L_CM_CC_resized.jpg', 'P99_R_CM_CC_resized.jpg',
                                     ]
        },
        'best_epoch': 1
    },
    {
        'name': 'B_synergistic_original_withsmoothing',
        'fusion': 'synergistic',
        'data': 'original',
        'smoothing': True,
        'test_metrics': {
            'confusion_matrix': {
                'TN': 26,
                'FP': 8,
                'FN': 13,
                'TP': 59,
            },
            'accuracy': 0.8019,
            'precision': 0.8806,
            'recall': 0.8194,
            'f1_score': 0.8489,
            'auc': 0.8190,
            'npv': 0.6667,
            'misclassified_images': ['P299_R_CM_MLO_resized.jpg', 'P103_R_CM_MLO_resized.jpg', 'P301_L_CM_MLO_resized.jpg',
                                     'P320_L_CM_CC_resized.jpg', 'P145_L_CM_MLO_resized.jpg', 'P99_R_CM_MLO_resized.jpg',
                                     'P98_L_CM_MLO_resized.jpg', 'P323_L_CM_CC_resized.jpg', 'P191_L_CM_MLO_resized.jpg',
                                     'P265_R_CM_MLO_resized.jpg', 'P263_L_CM_CC_resized.jpg', 'P11_R_CM_MLO_resized.jpg',
                                     'P98_L_CM_CC_resized.jpg', 'P301_L_CM_CC_resized.jpg', 'P118_L_CM_CC_resized.jpg',
                                     'P211_L_CM_MLO_resized.jpg', 'P320_L_CM_MLO_resized.jpg', 'P265_L_CM_MLO_resized.jpg',
                                     'P64_R_CM_CC_resized.jpg', 'P111_R_CM_CC_resized.jpg', 'P254_R_CM_MLO_resized.jpg'
                                     ]
        },
        'best_epoch': 4
    },
    {
        'name': 'C_conditional_original_withsmoothing',
        'fusion': 'conditional',
        'data': 'original',
        'smoothing': True,
        'test_metrics': {
            'confusion_matrix': {
                'TN': 25,
                'FP': 9,
                'FN': 10,
                'TP': 62,
            },
            'accuracy': 0.8208,
            'precision': 0.8732,
            'recall': 0.8611,
            'f1_score': 0.8671,
            'auc': 0.8599,
            'npv': 0.7143,
            'misclassified_images': ['P299_R_CM_MLO_resized.jpg', 'P191_L_CM_CC_resized.jpg', 'P301_L_CM_MLO_resized.jpg',
                                     'P320_L_CM_CC_resized.jpg', 'P145_L_CM_MLO_resized.jpg', 'P98_L_CM_MLO_resized.jpg',
                                     'P103_R_CM_CC_resized.jpg', 'P91_L_CM_MLO_resized.jpg', 'P98_L_CM_CC_resized.jpg',
                                     'P88_R_CM_MLO_resized.jpg', 'P64_L_CM_MLO_resized.jpg', 'P301_L_CM_CC_resized.jpg',
                                     'P29_L_CM_CC_resized.jpg', 'P211_L_CM_MLO_resized.jpg', 'P320_L_CM_MLO_resized.jpg',
                                     'P83_L_CM_CC_resized.jpg', 'P64_R_CM_CC_resized.jpg', 'P111_R_CM_CC_resized.jpg',
                                     'P77_R_CM_CC_resized.jpg'
                                     ]
        },
        'best_epoch': 12
    },
    {
        'name': 'D_synergistic_clean_nosmoothing',
        'fusion': 'synergistic',
        'data': 'clean',
        'smoothing': False,
        'test_metrics': {
            'confusion_matrix': {
                'TN': 45,
                'FP': 2,
                'FN': 7,
                'TP': 48,
            },
            'accuracy': 0.9118,
            'precision': 0.96,
            'recall': 0.8727,
            'f1_score': 0.9143,
            'auc': 0.9729,
            'npv': 0.8654,
            'misclassified_images': ['P322_L_CM_CC_resized.jpg', 'P88_R_CM_CC_resized.jpg', 'P205_L_CM_MLO_resized.jpg', 
                                     'P145_L_CM_MLO_resized.jpg', 'P145_L_CM_CC_resized.jpg', 'P119_R_CM_MLO_resized.jpg',
                                     'P323_L_CM_CC_resized.jpg', 'P83_L_CM_MLO_resized.jpg', 'P83_L_CM_CC_resized.jpg',
                                     ]
        },
        'best_epoch': 13
    },
    {
        'name': 'E_conditional_clean_nosmoothing',
        'fusion': 'conditional',
        'data': 'clean',
        'smoothing': False,
        'test_metrics': {
            'confusion_matrix': {
                'TN': 41,
                'FP': 6,
                'FN': 5,
                'TP': 50,
            },
            'accuracy': 0.8922,
            'precision': 0.8929,
            'recall': 0.9091,
            'f1_score': 0.9009,
            'auc': 0.9308,
            'npv': 0.8913,
            'misclassified_images': ['P145_L_CM_MLO_resized.jpg', 'P119_R_CM_MLO_resized.jpg', 'P323_L_CM_CC_resized.jpg',
                                     'P28_L_CM_MLO_resized.jpg', 'P125_L_CM_CC_resized.jpg', 'P11_R_CM_MLO_resized.jpg',
                                     'P217_L_CM_CC_resized.jpg', 'P73_L_CM_CC_resized.jpg', 'P217_L_CM_MLO_resized.jpg',
                                     'P265_L_CM_MLO_resized.jpg', 'P64_R_CM_CC_resized.jpg',
                                     ]
        },
        'best_epoch': 1
    },
    {
        'name': 'F_synergistic_clean_withsmoothing',
        'fusion': 'synergistic',
        'data': 'clean',
        'smoothing': True,
        'test_metrics': {
            'confusion_matrix': {
                'TN': None,
                'FP': None,
                'FN': None,
                'TP': None,
            },
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1_score': None,
            'auc': None,
            'npv': None,
            'misclassified_images': []
        }
    },
]