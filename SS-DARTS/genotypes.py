from collections import namedtuple

Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    # 'sep_conv_3x3',
    # 'sep_conv_5x5',
    # 'dil_conv_3x3',
    # 'dil_conv_5x5',
    # 'conv_3x3',
    # 'conv_7x1_1x7',
    # 'double_sep_conv_3x3',
    # 'leaky_conv_3x3',
    # 'leaky_conv_5x5',
    'leaky_conv_7x1_1x7',
    'leaky_sep_conv_3x3',
    'leaky_sep_conv_5x5',
    'leaky_sep_conv2_3x3',
    'leaky_dil_conv_3x3',
    'leaky_dil_conv_5x5',
    'spectral_attention',
    'spatial_attention'


]

NASNet = Genotype(
    normal=[
        ("sep_conv_5x5", 1),
        ("sep_conv_3x3", 0),
        ("sep_conv_5x5", 0),
        ("sep_conv_3x3", 0),
        ("avg_pool_3x3", 1),
        ("skip_connect", 0),
        ("avg_pool_3x3", 0),
        ("avg_pool_3x3", 0),
        ("sep_conv_3x3", 1),
        ("skip_connect", 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ("sep_conv_5x5", 1),
        ("sep_conv_7x7", 0),
        ("max_pool_3x3", 1),
        ("sep_conv_7x7", 0),
        ("avg_pool_3x3", 1),
        ("sep_conv_5x5", 0),
        ("skip_connect", 3),
        ("avg_pool_3x3", 2),
        ("sep_conv_3x3", 2),
        ("max_pool_3x3", 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ("avg_pool_3x3", 0),
        ("max_pool_3x3", 1),
        ("sep_conv_3x3", 0),
        ("sep_conv_5x5", 2),
        ("sep_conv_3x3", 0),
        ("avg_pool_3x3", 3),
        ("sep_conv_3x3", 1),
        ("skip_connect", 1),
        ("skip_connect", 0),
        ("avg_pool_3x3", 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ("avg_pool_3x3", 0),
        ("sep_conv_3x3", 1),
        ("max_pool_3x3", 0),
        ("sep_conv_7x7", 2),
        ("sep_conv_7x7", 0),
        ("avg_pool_3x3", 1),
        ("max_pool_3x3", 0),
        ("max_pool_3x3", 1),
        ("conv_7x1_1x7", 0),
        ("sep_conv_3x3", 5),
    ],
    reduce_concat=[3, 4, 6],
)

DARTS_V1 = Genotype(
    normal=[
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 0),
        ("skip_connect", 0),
        ("sep_conv_3x3", 1),
        ("skip_connect", 0),
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 0),
        ("skip_connect", 2),
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ("max_pool_3x3", 0),
        ("max_pool_3x3", 1),
        ("skip_connect", 2),
        ("max_pool_3x3", 0),
        ("max_pool_3x3", 0),
        ("skip_connect", 2),
        ("skip_connect", 2),
        ("avg_pool_3x3", 0),
    ],
    reduce_concat=[2, 3, 4, 5],
)
DARTS_V2 = Genotype(
    normal=[
        ("sep_conv_3x3", 0),
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 0),
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 1),
        ("skip_connect", 0),
        ("skip_connect", 0),
        ("dil_conv_3x3", 2),
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ("max_pool_3x3", 0),
        ("max_pool_3x3", 1),
        ("skip_connect", 2),
        ("max_pool_3x3", 1),
        ("max_pool_3x3", 0),
        ("skip_connect", 2),
        ("skip_connect", 2),
        ("max_pool_3x3", 1),
    ],
    reduce_concat=[2, 3, 4, 5],
)


PC_DARTS_cifar = Genotype(
    normal=[
        ("sep_conv_3x3", 1),
        ("skip_connect", 0),
        ("sep_conv_3x3", 0),
        ("dil_conv_3x3", 1),
        ("sep_conv_5x5", 0),
        ("sep_conv_3x3", 1),
        ("avg_pool_3x3", 0),
        ("dil_conv_3x3", 1),
    ],
    normal_concat=range(2, 6),
    reduce=[
        ("sep_conv_5x5", 1),
        ("max_pool_3x3", 0),
        ("sep_conv_5x5", 1),
        ("sep_conv_5x5", 2),
        ("sep_conv_3x3", 0),
        ("sep_conv_3x3", 3),
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 2),
    ],
    reduce_concat=range(2, 6),
)
PC_DARTS_image = Genotype(
    normal=[
        ("skip_connect", 1),
        ("sep_conv_3x3", 0),
        ("sep_conv_3x3", 0),
        ("skip_connect", 1),
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 3),
        ("sep_conv_3x3", 1),
        ("dil_conv_5x5", 4),
    ],
    normal_concat=range(2, 6),
    reduce=[
        ("sep_conv_3x3", 0),
        ("skip_connect", 1),
        ("dil_conv_5x5", 2),
        ("max_pool_3x3", 1),
        ("sep_conv_3x3", 2),
        ("sep_conv_3x3", 1),
        ("sep_conv_5x5", 0),
        ("sep_conv_3x3", 3),
    ],
    reduce_concat=range(2, 6),
)

SS_DARTS = Genotype(
    normal=[
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 1),
        ('dil_conv_3x3', 0),
        ('dil_conv_5x5', 1),
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('spectral_attention', 0),
        ('spatial_attention', 1),
    ],
    normal_concat=[2, 3, 4, 5],  # Concatenate nodes 2, 3, 4, 5 in the normal cell
    reduce=[
        ('max_pool_3x3', 0),
        ('sep_conv_5x5', 1),
        ('avg_pool_3x3', 0),
        ('dil_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('spectral_attention', 1),
        ('dil_conv_3x3', 0),
        ('dil_conv_5x5', 1),
    ],
    reduce_concat=[2, 3, 4, 5]  # Concatenate nodes 2, 3, 4, 5 in the reduce cell
)

SS_DARTS2 = Genotype(
    normal=[
        ("leaky_sep_conv_5x5", 1),
        ("leaky_sep_conv_3x3", 0),
        ("leaky_sep_conv_5x5", 0),
        ("leaky_sep_conv_3x3", 0),
        ("avg_pool_3x3", 1),
        ("skip_connect", 0),
        ("avg_pool_3x3", 0),
        ('spectral_attention', 0),
        ('spatial_attention', 1),
        ("skip_connect", 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ("leaky_sep_conv_5x5", 1),
        ("leaky_dil_conv_3x3", 0),
        ("max_pool_3x3", 1),
        ("leaky_dil_conv_5x5", 0),
        ("avg_pool_3x3", 1),
        ("leaky_sep_conv_5x5", 0),
        ("skip_connect", 3),
        ('spectral_attention', 2),
        ("leaky_sep_conv_3x3", 2),
        ("max_pool_3x3", 1),
    ],
    reduce_concat=[4, 5, 6],
)

HSI = Genotype(
    normal=[
        ("sep_conv_5x5", 1),
        ("sep_conv_3x3", 0),
        ("sep_conv_5x5", 0),
        ("sep_conv_3x3", 0),
        ("avg_pool_3x3", 1),
        ("skip_connect", 0),
        ("avg_pool_3x3", 0),
        ("spectral_attention", 0),
        ("spatial_attention", 1),
        ("skip_connect", 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ("sep_conv_5x5", 1),
        ("sep_conv_7x7", 0),
        ("max_pool_3x3", 1),
        ("sep_conv_7x7", 0),
        ("avg_pool_3x3", 1),
        ("sep_conv_5x5", 0),
        ("skip_connect", 3),
        ("avg_pool_3x3", 2),
        ("spectral_attention", 2),
        ("max_pool_3x3", 1),
    ],
    reduce_concat=[4, 5, 6],
)


HSI_autocnn = Genotype(normal=[('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('max_pool_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 2), ('dil_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('skip_connect', 4)], reduce_concat=range(2, 6))
# HSI = Genotype(normal=[('leaky_dil_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('leaky_dil_conv_5x5', 0), ('skip_connect', 3), ('spectral_attention', 2), ('skip_connect', 3), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('leaky_sep_conv_3x3', 0), ('leaky_dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('spectral_attention', 1), ('leaky_dil_conv_5x5', 3), ('leaky_dil_conv_5x5', 2)], reduce_concat=range(2, 6))
# HSI = SS_DARTS2
Genotype(normal=[('leaky_conv_7x1_1x7', 0), ('leaky_conv_3x3', 1), ('max_pool_3x3', 2), ('leaky_dil_conv_5x5', 0), ('max_pool_3x3', 1), ('leaky_conv_7x1_1x7', 3), ('leaky_conv_7x1_1x7', 3), ('avg_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('leaky_conv_7x1_1x7', 0), ('max_pool_3x3', 1), ('leaky_dil_conv_3x3', 2), ('leaky_conv_5x5', 1), ('max_pool_3x3', 3), ('leaky_sep_conv_3x3', 1), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))


