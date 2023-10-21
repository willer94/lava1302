vertex = [
    [0.302610248327, -0.577637493610, 0.250902324915],
    [0.305925250053, -0.299487084150, 0.250224351883],

    [0.540874958038, 0.488110482693, 0.253669917583],
    [0.300494909286, 0.350195318460, 0.255064159632],

    [-0.540754258633, 0.486388295889, 0.251724690199],
    [-0.299998313189, 0.346072077751, 0.255891531706],

    [0.367236763239, -0.382238954306, 0.319449841976],
    [-0.366536468267, -0.379578888416, 0.322174757719],
    [-0.369265586138, 0.379880815744, 0.317924767733],
    [0.369070827961, 0.381840527058, 0.315168917179],

    [0.366340786219, -0.260322451591, 0.000237861823],
    [-0.365681171417, -0.260622560978, 0.000237783926],
    [-0.364047288895, 0.302447468042, 0.001630304847],
    [0.365129649639, 0.301783651114, 0.001181621221],

    [0.365370005369, -0.259557664394, 0.297277867794],
    [-0.366234153509, -0.259814143181, 0.295170307159],
    [-0.370678007603, 0.304547727108, 0.300576537848],
    [0.369063496590, 0.303832083941, 0.296431541443],
]

line_idx = [
    [0, 1],
    [2, 3],
    [4, 5],

    [6, 7],
    [7, 8],
    [8, 9],
    [9, 6],

    [10, 11],
    [11, 12],
    [12, 13],
    [13, 10],

    [10, 14],
    [11, 15],
    [12, 16],
    [13, 17],
]

W_PT = [
    [0.30531443, -0.57893653, 0.25084064],
    [0.54477770, 0.48960986, 0.25270429],
    [-0.54289737, 0.48885894, 0.25350052],
    [0.36662819, -0.38234623, 0.32212312],
    [0.36480841, 0.38159211, 0.31985739],
    [-0.36705289, 0.38095879, 0.32031161],
    [-0.36714840, -0.38153599, 0.32090666],
    [0.36735206, -0.26200437, 0.00172357],
    [0.36712000, 0.30142491, -0.00013418],
    [-0.36787140, 0.30158204, 0.00124829],
    [-0.36798065, -0.26210211, 0.00069993],
]

# bbox = [
#     [0.30531443, -0.57893653, 0.25084064],
#     [0.54477770, 0.48960986, 0.25270429],
#     [-0.54289737, 0.48885894, 0.25350052],
#     [0.36662819, -0.38234623, 0.32212312],
#     [0.36480841, 0.38159211, 0.31985739],
#     [-0.36705289, 0.38095879, 0.32031161],
#     [-0.36714840, -0.38153599, 0.32090666],
#     [0.36735206, -0.26200437, 0.00172357],
#     [0.36712000, 0.30142491, -0.00013418],
#     [-0.36787140, 0.30158204, 0.00124829],
#     [-0.36798065, -0.26210211, 0.00069993],
# ]

bbox = [
    [-0.55046200, -0.57799800, -0.08213980],
    [0.54975600, -0.57799800, -0.08213980],
    [-0.55046200, 0.50239000, -0.08213980],
    [-0.55046200, -0.57799800, 0.33220800],
    [0.54975600, 0.50239000, 0.33220800],
    [-0.55046200, 0.50239000, 0.33220800],
    [0.54975600, -0.57799800, 0.33220800],
    [0.54975600, 0.50239000, -0.08213980],
]


bbox_line_idx = [
    (0, 1),
    (1, 7),
    (7, 2),
    (2, 0),
    (3, 6),
    (6, 4),
    (4, 5),
    (5, 3),
    (0, 3),
    (1, 6),
    (7, 4),
    (2, 5)
]