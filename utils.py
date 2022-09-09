from pyts.image import GramianAngularField


# Function that converts a single row of data into an "image"
def grab_image_data(subset):
    gasf_transformer = GramianAngularField(method='summation')
    gasf_subset = gasf_transformer.transform(subset)

    return gasf_subset


NU_VALS = [0.01, 0.05, 0.10]
REP_DIMS = [16, 32, 64, 128]
K_VALS = [5, 10, 15, 20]
LR_VALS = [0.001]

ALL_COMBOS = []

for nu in NU_VALS:
    for rep_dim in REP_DIMS:
        for k in K_VALS:
            for lr in LR_VALS:
                ALL_COMBOS.append((nu, rep_dim, k, lr))

PERCENTAGE_TO_TRY = 0.20
