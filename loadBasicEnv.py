import os

# processed img shape
IMGSHAPE = 128

NET_SAVE_NAME = "vgg"

WORKDIR = "photo"
SAMPLE_DIR = f"./{WORKDIR}"
OUTPUT_DIR = f"./{WORKDIR}/processed"
OUTPUT_NUM = 4000
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

rawTags = []

if os.path.exists("{NET_SAVE_NAME}_DICT.txt"):
    with open(f"{NET_SAVE_NAME}_DICT.txt", "r") as f:
        data = f.read().split('mmm')
    DICT, NAMEDICT = map(eval, data)
else:
    filenames = [x for x in os.listdir(f"{OUTPUT_DIR}") if x.endswith("jpg")]
    for imgName in filenames:
        name, _ = imgName.split(".")
        tag, _ = name.split("_")
        if not tag in rawTags:
            rawTags.append(tag)

    DICT = {k: v for k, v in zip(rawTags, range(len(rawTags)))}
    NAMEDICT = {v: k for k, v in zip(rawTags, range(len(rawTags)))}
    print('loading data')

    with open(f"{NET_SAVE_NAME}_DICT.txt", "w") as f:
        f.write(str(DICT))
        f.write("mmm")
        f.write(str(NAMEDICT))
