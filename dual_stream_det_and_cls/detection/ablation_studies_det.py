
import numpy as np

experiments = [
    {
        'name': 'A_synergistic_baseAugs_noclahe',
        'fusion': 'synergistic',
        'augmentations': ["_hflip_noise", "_hvflip_brightcont", '_resized', '_rotate20', "_symm_shit_blur"],
        'clahe': False,
        'test_metrics': {    
            "fold 0":{
                "minoverlap=0.0":{
                "AP@0": 0.7398,
                "F1": 0.69,
                "Recall": 0.5985,
                "Precision": 0.8229
                },
                "minoverlap=0.50":{
                "AP@50": 0.5728,
                "F1": 0.61,
                "Recall": 0.5227,
                "Precision": 0.7188
                },
                "AP@[.50:.05:.95]": 0.2384,
                'best_epoch': 25
            },
            "fold 1":{
            "minoverlap=0.0":{
                "AP@0": 0.7585,
                "F1": 0.73,
                "Recall": 0.6953,
                "Precision": 0.7607
                },
                "minoverlap=0.50":{
                "AP@50": 0.6668,
                "F1": 0.65,
                "Recall": 0.6250,
                "Precision": 0.6838
                },
                "AP@[.50:.05:.95]": 0.32,
                'best_epoch': 23
            },
            "fold 2":{
                "minoverlap=0.0":{
                "AP@0": 0.7981,
                "F1": 0.76,
                "Recall": 0.7789,
                "Precision": 0.7475
                },
                "minoverlap=0.50":{
                "AP@50": 0.6907,
                "F1": 0.66,
                "Recall": 0.6737,
                "Precision": 0.6465
                },
                "AP@[.50:.05:.95]": 0.3207,
                'best_epoch': 24
            },
            "fold 3":{
                "minoverlap=0.0":{
                "AP@0": 0.766,
                "F1": 0.69,
                "Recall": 0.5862,
                "Precision": 0.8252
                },
                "minoverlap=0.50":{
                "AP@50": 0.6462,
                "F1": 0.63,
                "Recall": 0.5379,
                "Precision": 0.7573
                },
                "AP@[.50:.05:.95]": 0.274,
                'best_epoch': 15
            },
            "fold 4":{
                "minoverlap=0.0":{
                "AP@0": 0.6881,
                "F1": 0.66,
                "Recall": 0.66,
                "Precision": 0.6667
                },
                "minoverlap=0.50":{
                "AP@50": 0.5943,
                "F1": 0.59,
                "Recall": 0.59,
                "Precision": 0.596
                },
                "AP@[.50:.05:.95]": 0.2577,
                'best_epoch': 15
            }
        },
        'mean_metrics': {
            'minoverlap=0.0': {
                'AP@0': 0.7501,
                'F1': 0.7060,
                'Recall': 0.6638,
                'Precision': 0.7646,
            },
            'minoverlap=0.50': {
                'AP@50': 0.6342,
                'F1': 0.6280,
                'Recall': 0.5899,
                'Precision': 0.6805,
            },
            'AP@[.50:.05:.95]': 0.2822
        },
         'std_metrics': {
            'minoverlap=0.0': {
                'AP@0': 0.0363,
                'F1': 0.0350,
                'Recall': 0.0701,
                'Precision': 0.0582,
            },
            'minoverlap=0.50': {
                'AP@50': 0.0442,
                'F1': 0.0256,
                'Recall': 0.0556,
                'Precision': 0.0560,
            },
            'AP@[.50:.05:.95]': 0.0332
        },
 
    },

    {
    'name': 'B_synergistic_baseAugs_elastic_noclahe',
    'fusion': 'synergistic',
    'augmentations': ["_hflip_noise", "_hvflip_brightcont", '_resized', '_rotate20_elastic', "_symm_shit_blur"],
    'clahe': False,
    'test_metrics': {    
        "fold 0":{
            "minoverlap=0.0":{
            "AP@0": 0.71,
            "F1": 0.64,
            "Recall": 0.5682,
            "Precision": 0.7426
            },
            "minoverlap=0.50":{
            "AP@50": 0.5466,
            "F1": 0.54,
            "Recall": 0.4773,
            "Precision": 0.6238
            },
            "AP@[.50:.05:.95]": 0.2334,
            'best_epoch': 17
        },
        "fold 1":{
        "minoverlap=0.0":{
            "AP@0": 0.7567,
            "F1": 0.75,
            "Recall": 0.7422,
            "Precision": 0.76
            },
            "minoverlap=0.50":{
            "AP@50": 0.6625,
            "F1": 0.66,
            "Recall": 0.6562,
            "Precision": 0.672
            },
            "AP@[.50:.05:.95]": 0.3249,
            'best_epoch': 23
        },
        "fold 2":{
            "minoverlap=0.0":{
            "AP@0": 0.8321,
            "F1": 0.79,
            "Recall": 0.8316,
            "Precision": 0.7524
            },
            "minoverlap=0.50":{
            "AP@50": 0.7048,
            "F1": 0.7,
            "Recall": 0.7368,
            "Precision": 0.6667
            },
            "AP@[.50:.05:.95]": 0.3464,
            'best_epoch': 21
        },
        "fold 3":{
            "minoverlap=0.0":{
            "AP@0": 0.7852,
            "F1": 0.68,
            "Recall": 0.5655,
            "Precision": 0.8542
            },
            "minoverlap=0.50":{
            "AP@50": 0.6425,
            "F1": 0.61,
            "Recall": 0.5034,
            "Precision": 0.7604
            },
            "AP@[.50:.05:.95]": 0.2495,
            'best_epoch': 15
        },
        "fold 4":{
            "minoverlap=0.0":{
            "AP@0": 0.7319,
            "F1": 0.68,
            "Recall": 0.64,
            "Precision": 0.7191
            },
            "minoverlap=0.50":{
            "AP@50": 0.6255,
            "F1": 0.61,
            "Recall": 0.58,
            "Precision": 0.6517
            },
            "AP@[.50:.05:.95]": 0.2781,
            'best_epoch': 21
        }
    },
    'mean_metrics': {
        'minoverlap=0.0': {
            'AP@0': 0.7632,
            'F1': 0.7080,
            'Recall': 0.6695,
            'Precision': 0.7657,
        },
        'minoverlap=0.50': {
            'AP@50': 0.6364,
            'F1': 0.624,
            'Recall': 0.5907,
            'Precision': 0.6749,
        },
        'AP@[.50:.05:.95]': 0.2865
    },
        'std_metrics': {
        'minoverlap=0.0': {
            'AP@0': 0.0426,
            'F1': 0.0542,
            'Recall': 0.1035,
            'Precision': 0.0464,
        },
        'minoverlap=0.50': {
            'AP@50': 0.0521,
            'F1': 0.0539,
            'Recall': 0.0962,
            'Precision': 0.0459,
        },
        'AP@[.50:.05:.95]': 0.0432
    },

},
    {
    'name': 'C_synergistic_baseAugs_elastic_gridshuffle_noclahe',
    'fusion': 'synergistic',
    'augmentations': ["_hflip_noise", "_hvflip_brightcont", '_resized', '_rotate20_elastic', "_symm_shit_blur", "_gridshuffle"],
    'clahe': False,
    'test_metrics': {    
        "fold 0":{
            "minoverlap=0.0":{
            "AP@0": 0.7088,
            "F1": 0.66,
            "Recall": 0.5606,
            "Precision": 0.8043
            },
            "minoverlap=0.50":{
            "AP@50": 0.5598,
            "F1": 0.59,
            "Recall": 0.5,
            "Precision": 0.7174
            },
            "AP@[.50:.05:.95]": 0.2049,
            'best_epoch': 13
        },
        "fold 1":{
        "minoverlap=0.0":{
            "AP@0": 0.7664,
            "F1": 0.73,
            "Recall": 0.7109,
            "Precision": 0.7521
            },
            "minoverlap=0.50":{
            "AP@50": 0.6737,
            "F1": 0.67,
            "Recall": 0.6484,
            "Precision": 0.686
            },
            "AP@[.50:.05:.95]": 0.3286,
            'best_epoch': 23
        },
        "fold 2":{
            "minoverlap=0.0":{
            "AP@0": 0.7959,
            "F1": 0.75,
            "Recall": 0.8,
            "Precision": 0.7037
            },
            "minoverlap=0.50":{
            "AP@50": 0.6897,
            "F1": 0.66,
            "Recall": 0.7053,
            "Precision": 0.6204
            },
            "AP@[.50:.05:.95]": 0.3470,
            'best_epoch': 18
        },
        "fold 3":{
            "minoverlap=0.0":{
            "AP@0": 0.7791,
            "F1": 0.72,
            "Recall": 0.6345,
            "Precision": 0.8364
            },
            "minoverlap=0.50":{
            "AP@50": 0.6586,
            "F1": 0.64,
            "Recall": 0.5655,
            "Precision": 0.7455
            },
            "AP@[.50:.05:.95]": 0.2853,
            'best_epoch': 18
        },
        "fold 4":{
            "minoverlap=0.0":{
            "AP@0": 0.7317,
            "F1": 0.69,
            "Recall": 0.63,
            "Precision": 0.759
            },
            "minoverlap=0.50":{
            "AP@50": 0.6176,
            "F1": 0.61,
            "Recall": 0.56,
            "Precision": 0.6747
            },
            "AP@[.50:.05:.95]": 0.2638,
            'best_epoch': 20
        }
    },
    'mean_metrics': {
        'minoverlap=0.0': {
            'AP@0': 0.7564,
            'F1': 0.71,
            'Recall': 0.6672,
            'Precision': 0.7711,
        },
        'minoverlap=0.50': {
            'AP@50': 0.6399,
            'F1': 0.634,
            'Recall': 0.5958,
            'Precision': 0.6888,
        },
        'AP@[.50:.05:.95]': 0.2859
    },
        'std_metrics': {
        'minoverlap=0.0': {
            'AP@0': 0.0318,
            'F1': 0.0316,
            'Recall': 0.0817,
            'Precision': 0.0456,
        },
        'minoverlap=0.50': {
            'AP@50': 0.0467,
            'F1': 0.0301,
            'Recall': 0.0723,
            'Precision': 0.0422,
        },
        'AP@[.50:.05:.95]': 0.0502
    },

},
    {
    'name': 'D_synergistic_baseAugs_elastic_gridshuffle_cutout_noclahe',
    'fusion': 'synergistic',
    'augmentations': ["_hflip_noise", "_hvflip_brightcont", '_resized', '_rotate20_elastic', "_symm_shit_blur", "_gridshuffle", "_cutout"],
    'clahe': False,
    'test_metrics': {    
        "fold 0":{
            "minoverlap=0.0":{
            "AP@0": 0.7277,
            "F1": 0.66,
            "Recall": 0.5379,
            "Precision": 0.8659
            },
            "minoverlap=0.50":{
            "AP@50": 0.5705,
            "F1": 0.58,
            "Recall": 0.4697,
            "Precision": 0.7561
            },
            "AP@[.50:.05:.95]": 0.2245,
            'best_epoch': 19
        },
        "fold 1":{
        "minoverlap=0.0":{
            "AP@0": 0.7752,
            "F1": 0.74,
            "Recall": 0.7422,
            "Precision": 0.7364
            },
            "minoverlap=0.50":{
            "AP@50": 0.6891,
            "F1": 0.68,
            "Recall": 0.6875,
            "Precision": 0.6822
            },
            "AP@[.50:.05:.95]": 0.3314,
            'best_epoch': 22
        },
        "fold 2":{
            "minoverlap=0.0":{
            "AP@0": 0.8031,
            "F1": 0.74,
            "Recall": 0.7053,
            "Precision": 0.7791
            },
            "minoverlap=0.50":{
            "AP@50": 0.7003,
            "F1": 0.66,
            "Recall": 0.6316,
            "Precision": 0.6977
            },
            "AP@[.50:.05:.95]": 0.3518,
            'best_epoch': 17
        },
        "fold 3":{
            "minoverlap=0.0":{
            "AP@0": 0.7897,
            "F1": 0.74,
            "Recall": 0.6345,
            "Precision": 0.8762
            },
            "minoverlap=0.50":{
            "AP@50": 0.6647,
            "F1": 0.66,
            "Recall": 0.5724,
            "Precision": 0.7905
            },
            "AP@[.50:.05:.95]": 0.2894,
            'best_epoch': 25
        },
        "fold 4":{
            "minoverlap=0.0":{
            "AP@0": 0.7346,
            "F1": 0.68,
            "Recall": 0.7,
            "Precision": 0.6667
            },
            "minoverlap=0.50":{
            "AP@50": 0.6412,
            "F1": 0.62,
            "Recall": 0.64,
            "Precision": 0.6095
            },
            "AP@[.50:.05:.95]": 0.2707,
            'best_epoch': 21
        }
    },
    'mean_metrics': {
        'minoverlap=0.0': {
            'AP@0': 0.7661,
            'F1': 0.712,
            'Recall': 0.664,
            'Precision': 0.7849,
        },
        'minoverlap=0.50': {
            'AP@50': 0.6532,
            'F1': 0.64,
            'Recall': 0.6002,
            'Precision': 0.7072,
        },
        'AP@[.50:.05:.95]': 0.2936
    },
        'std_metrics': {
        'minoverlap=0.0': {
            'AP@0': 0.0299,
            'F1': 0.0349,
            'Recall': 0.072,
            'Precision': 0.0791,
        },
        'minoverlap=0.50': {
            'AP@50': 0.0461,
            'F1': 0.0358,
            'Recall': 0.0748,
            'Precision': 0.0626,
        },
        'AP@[.50:.05:.95]': 0.045
    },

},
    {
    'name': 'E_synergistic_baseAugs_elastic_gridshuffle_cutoutMulnoise_noclahe',
    'fusion': 'synergistic',
    'augmentations': ["_hflip_noise", "_hvflip_brightcont", '_resized', '_rotate20_elastic', "_symm_shit_blur", "_gridshuffle", "_cutout_mulnoise"],
    'clahe': False,
    'test_metrics': {    
        "fold 0":{
            "minoverlap=0.0":{
            "AP@0": 0.7309,
            "F1": 0.68,
            "Recall": 0.5833,
            "Precision": 0.8191
            },
            "minoverlap=0.50":{
            "AP@50": 0.5535,
            "F1": 0.54,
            "Recall": 0.4621,
            "Precision": 0.6489
            },
            "AP@[.50:.05:.95]": 0.2205,
            'best_epoch': 15
        },
        "fold 1":{
        "minoverlap=0.0":{
            "AP@0": 0.7503,
            "F1": 0.72,
            "Recall": 0.7266,
            "Precision": 0.7099
            },
            "minoverlap=0.50":{
            "AP@50": 0.6274,
            "F1": 0.63,
            "Recall": 0.6328,
            "Precision": 0.6183
            },
            "AP@[.50:.05:.95]": 0.3127,
            'best_epoch': 24
        },
        "fold 2":{
            "minoverlap=0.0":{
            "AP@0": 0.8005,
            "F1": 0.76,
            "Recall": 0.7158,
            "Precision": 0.8095
            },
            "minoverlap=0.50":{
            "AP@50": 0.695,
            "F1": 0.68,
            "Recall": 0.6421,
            "Precision": 0.7262
            },
            "AP@[.50:.05:.95]": 0.3603,
            'best_epoch': 20
        },
        "fold 3":{
            "minoverlap=0.0":{
            "AP@0": 0.751,
            "F1": 0.7,
            "Recall": 0.6483,
            "Precision": 0.7705
            },
            "minoverlap=0.50":{
            "AP@50": 0.6477,
            "F1": 0.63,
            "Recall": 0.5793,
            "Precision": 0.6885
            },
            "AP@[.50:.05:.95]": 0.283,
            'best_epoch': 22
        },
        "fold 4":{
            "minoverlap=0.0":{
            "AP@0": 0.6906,
            "F1": 0.65,
            "Recall": 0.61,
            "Precision": 0.7011
            },
            "minoverlap=0.50":{
            "AP@50": 0.5904,
            "F1": 0.56,
            "Recall": 0.52,
            "Precision": 0.5977
            },
            "AP@[.50:.05:.95]": 0.2548,
            'best_epoch': 20
        }
    },
    'mean_metrics': {
        'minoverlap=0.0': {
            'AP@0': 0.7447,
            'F1': 0.7020,
            'Recall': 0.6568,
            'Precision': 0.7620,
        },
        'minoverlap=0.50': {
            'AP@50': 0.6228,
            'F1': 0.608,
            'Recall': 0.5673,
            'Precision': 0.6559,
        },
        'AP@[.50:.05:.95]': 0.2863
    },
        'std_metrics': {
        'minoverlap=0.0': {
            'AP@0': 0.0355,
            'F1': 0.0371,
            'Recall': 0.0566,
            'Precision': 0.0490,
        },
        'minoverlap=0.50': {
            'AP@50': 0.0484,
            'F1': 0.0511,
            'Recall': 0.0683,
            'Precision': 0.0466,
        },
        'AP@[.50:.05:.95]': 0.048
    },

},
{
    'name': 'F_synergistic_baseAugs_elastic_gridshuffle_cutout_clahe',
    'fusion': 'synergistic',
    'augmentations': ["_hflip_noise", "_hvflip_brightcont", '_resized', '_rotate20_elastic', "_symm_shit_blur", "_gridshuffle", "_cutout"],
    'clahe': True,
    'test_metrics': {    
        "fold 0":{
            "minoverlap=0.0":{
            "AP@0": 0.7177,
            "F1": 0.68,
            "Recall": 0.5985,
            "Precision": 0.79
            },
            "minoverlap=0.50":{
            "AP@50": 0.5892,
            "F1": 0.59,
            "Recall": 0.5152,
            "Precision": 0.68
            },
            "AP@[.50:.05:.95]": 0.2335,
            'best_epoch': 20
        },
        "fold 1":{
        "minoverlap=0.0":{
            "AP@0": 0.7352,
            "F1": 0.67,
            "Recall": 0.5547,
            "Precision": 0.8554
            },
            "minoverlap=0.50":{
            "AP@50": 0.6527,
            "F1": 0.63,
            "Recall": 0.5156,
            "Precision": 0.7952
            },
            "AP@[.50:.05:.95]": 0.2973,
            'best_epoch': 12
        },
        "fold 2":{
            "minoverlap=0.0":{
            "AP@0": 0.771,
            "F1": 0.73,
            "Recall": 0.7474,
            "Precision": 0.71
            },
            "minoverlap=0.50":{
            "AP@50": 0.6774,
            "F1": 0.65,
            "Recall": 0.6632,
            "Precision": 0.63
            },
            "AP@[.50:.05:.95]": 0.3369,
            'best_epoch': 24
        },
        "fold 3":{
            "minoverlap=0.0":{
            "AP@0": 0.8001,
            "F1": 0.75,
            "Recall": 0.6483,
            "Precision": 0.8785
            },
            "minoverlap=0.50":{
            "AP@50": 0.6701,
            "F1": 0.67,
            "Recall": 0.5862,
            "Precision": 0.7944
            },
            "AP@[.50:.05:.95]": 0.3028,
            'best_epoch': 22
        },
        "fold 4":{
            "minoverlap=0.0":{
            "AP@0": 0.7621,
            "F1": 0.71,
            "Recall": 0.66,
            "Precision": 0.7674
            },
            "minoverlap=0.50":{
            "AP@50": 0.6258,
            "F1": 0.62,
            "Recall": 0.58,
            "Precision": 0.6744
            },
            "AP@[.50:.05:.95]": 0.2693,
            'best_epoch': 20
        }
    },
    'mean_metrics': {
        'minoverlap=0.0': {
            'AP@0': 0.7572,
            'F1': 0.708,
            'Recall': 0.6418,
            'Precision': 0.8003,
        },
        'minoverlap=0.50': {
            'AP@50': 0.643,
            'F1': 0.632,
            'Recall': 0.572,
            'Precision': 0.7148,
        },
        'AP@[.50:.05:.95]': 0.288
    },
        'std_metrics': {
        'minoverlap=0.0': {
            'AP@0': 0.0286,
            'F1': 0.0299,
            'Recall': 0.0648,
            'Precision': 0.0608,
        },
        'minoverlap=0.50': {
            'AP@50': 0.0323,
            'F1': 0.0271,
            'Recall': 0.0548,
            'Precision': 0.0676,
        },
        'AP@[.50:.05:.95]': 0.0347
    },

},
{
    'name': 'dual_channel_synergistic_baseAugs_elastic_gridshuffle_cutout_noclahe',
    'fusion': 'synergistic',
    'augmentations': ["_hflip_noise", "_hvflip_brightcont", '_resized', '_rotate20_elastic', "_symm_shit_blur", "_gridshuffle", "_cutout"],
    'clahe': False,
    'test_metrics': {    
        "fold 0":{
            "minoverlap=0.0":{
            "AP@0": 0.5936,
            "F1": 0.43,
            "Recall": 0.2879,
            "Precision": 0.8636
            },
            "minoverlap=0.50":{
            "AP@50": 0.3885,
            "F1": 0.34,
            "Recall": 0.2273,
            "Precision": 0.6818
            },
            "AP@[.50:.05:.95]": 0.1544,
            'best_epoch': 12
        },
        "fold 1":{
        "minoverlap=0.0":{
            "AP@0": 0.6405,
            "F1": 0.64,
            "Recall": 0.5469,
            "Precision": 0.7609
            },
            "minoverlap=0.50":{
            "AP@50": 0.5432,
            "F1": 0.59,
            "Recall": 0.5078,
            "Precision": 0.7065
            },
            "AP@[.50:.05:.95]": 0.2417,
            'best_epoch': 20
        },
        "fold 2":{
            "minoverlap=0.0":{
            "AP@0": 0.748,
            "F1": 0.74,
            "Recall": 0.7263,
            "Precision": 0.7582
            },
            "minoverlap=0.50":{
            "AP@50": 0.5807,
            "F1": 0.6,
            "Recall": 0.5895,
            "Precision": 0.6154
            },
            "AP@[.50:.05:.95]": 0.2989,
            'best_epoch': 18
        },
        "fold 3":{
            "minoverlap=0.0":{
            "AP@0": 0.6411,
            "F1": 0.55,
            "Recall": 0.4,
            "Precision": 0.8788
            },
            "minoverlap=0.50":{
            "AP@50": 0.4796,
            "F1": 0.47,
            "Recall": 0.3448,
            "Precision": 0.7576
            },
            "AP@[.50:.05:.95]": 0.1899,
            'best_epoch': 12
        },
        "fold 4":{
            "minoverlap=0.0":{
            "AP@0": 0.6017,
            "F1": 0.62,
            "Recall": 0.54,
            "Precision": 0.72
            },
            "minoverlap=0.50":{
            "AP@50": 0.3858,
            "F1": 0.43,
            "Recall": 0.38,
            "Precision": 0.5067
            },
            "AP@[.50:.05:.95]": 0.1514,
            'best_epoch': 15
        }
    },
    'mean_metrics': {
        'minoverlap=0.0': {
            'AP@0': 0.645,
            'F1': 0.596,
            'Recall': 0.596,
            'Precision': 0.7963,
        },
        'minoverlap=0.50': {
            'AP@50': 0.4756,
            'F1': 0.486,
            'Recall': 0.4099,
            'Precision': 0.6536,
        },
        'AP@[.50:.05:.95]': 0.2073
    },
        'std_metrics': {
        'minoverlap=0.0': {
            'AP@0': 0.0551,
            'F1': 0.1029,
            'Recall': 0.1483,
            'Precision': 0.063,
        },
        'minoverlap=0.50': {
            'AP@50': 0.0791,
            'F1': 0.0985,
            'Recall': 0.1267,
            'Precision': 0.0865,
        },
        'AP@[.50:.05:.95]': 0.0562
    },

},
]


def calculate_stats_for_experiment(experiment):
    # Lists to store metrics from all folds
    metrics_0 = {
        'AP@0': [], 'F1': [], 'Recall': [], 'Precision': []
    }
    metrics_50 = {
        'AP@50': [], 'F1': [], 'Recall': [], 'Precision': []
    }
    ap_array = []

    # Collect metrics from all folds
    for fold in range(5):
        fold_data = experiment['test_metrics'][f'fold {fold}']
        
        # Collect metrics for minoverlap=0.0
        overlap_0 = fold_data['minoverlap=0.0']
        metrics_0['AP@0'].append(overlap_0['AP@0'])
        metrics_0['F1'].append(overlap_0['F1'])
        metrics_0['Recall'].append(overlap_0['Recall'])
        metrics_0['Precision'].append(overlap_0['Precision'])
        
        # Collect metrics for minoverlap=0.50
        overlap_50 = fold_data['minoverlap=0.50']
        metrics_50['AP@50'].append(overlap_50['AP@50'])
        metrics_50['F1'].append(overlap_50['F1'])
        metrics_50['Recall'].append(overlap_50['Recall'])
        metrics_50['Precision'].append(overlap_50['Precision'])
        
        # Collect AP@[.50:.05:.95]
        ap_array.append(fold_data['AP@[.50:.05:.95]'])

    # Calculate means
    experiment['mean_metrics']['minoverlap=0.0'] = {
        k: np.mean(v) for k, v in metrics_0.items()
    }
    experiment['mean_metrics']['minoverlap=0.50'] = {
        k: np.mean(v) for k, v in metrics_50.items()
    }
    experiment['mean_metrics']['AP@[.50:.05:.95]'] = np.mean(ap_array)

    # Calculate standard deviations
    experiment['std_metrics']['minoverlap=0.0'] = {
        k: np.std(v) for k, v in metrics_0.items()
    }
    experiment['std_metrics']['minoverlap=0.50'] = {
        k: np.std(v) for k, v in metrics_50.items()
    }
    experiment['std_metrics']['AP@[.50:.05:.95]'] = np.std(ap_array)

# Calculate stats for all experiments
for experiment in experiments:
    calculate_stats_for_experiment(experiment)

# Print results for verification
# ...existing code...

# Print results for verification
for experiment in experiments:
    print(f"\nExperiment: {experiment['name']}")
    
    print("\nMean Metrics:")
    print("--- IoU = 0.0 ---")
    print(f"AP@0: {experiment['mean_metrics']['minoverlap=0.0']['AP@0']:.4f}")
    print(f"F1: {experiment['mean_metrics']['minoverlap=0.0']['F1']:.4f}")
    print(f"Recall: {experiment['mean_metrics']['minoverlap=0.0']['Recall']:.4f}")
    print(f"Precision: {experiment['mean_metrics']['minoverlap=0.0']['Precision']:.4f}")
    
    print("\n--- IoU = 0.50 ---")
    print(f"AP@50: {experiment['mean_metrics']['minoverlap=0.50']['AP@50']:.4f}")
    print(f"F1: {experiment['mean_metrics']['minoverlap=0.50']['F1']:.4f}")
    print(f"Recall: {experiment['mean_metrics']['minoverlap=0.50']['Recall']:.4f}")
    print(f"Precision: {experiment['mean_metrics']['minoverlap=0.50']['Precision']:.4f}")
    
    print("\n--- Average Precision ---")
    print(f"AP@[.50:.05:.95]: {experiment['mean_metrics']['AP@[.50:.05:.95]']:.4f}")
    
    print("\nStd Metrics:")
    print("--- IoU = 0.0 ---")
    print(f"AP@0: {experiment['std_metrics']['minoverlap=0.0']['AP@0']:.4f}")
    print(f"F1: {experiment['std_metrics']['minoverlap=0.0']['F1']:.4f}")
    print(f"Recall: {experiment['std_metrics']['minoverlap=0.0']['Recall']:.4f}")
    print(f"Precision: {experiment['std_metrics']['minoverlap=0.0']['Precision']:.4f}")
    
    print("\n--- IoU = 0.50 ---")
    print(f"AP@50: {experiment['std_metrics']['minoverlap=0.50']['AP@50']:.4f}")
    print(f"F1: {experiment['std_metrics']['minoverlap=0.50']['F1']:.4f}")
    print(f"Recall: {experiment['std_metrics']['minoverlap=0.50']['Recall']:.4f}")
    print(f"Precision: {experiment['std_metrics']['minoverlap=0.50']['Precision']:.4f}")
    
    print("\n--- Average Precision ---")
    print(f"AP@[.50:.05:.95]: {experiment['std_metrics']['AP@[.50:.05:.95]']:.4f}")
    print("-" * 50)