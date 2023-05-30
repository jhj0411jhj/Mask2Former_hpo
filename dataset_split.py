import json
import random
import os
import shutil

# 分割前训练集的标签与图片路径
file = open('datasets/ytvis_2019/train.json', 'r')
in_dir = "datasets/ytvis_2019/train/JPEGImages"

data = json.load(file)
train_val_test_ratio = 4  # train:(val+test) = 4:1
video_list = data['videos']
annotations_list = data['annotations']


random.seed(1)
val_test_idx = random.sample(range(0, len(video_list)), int(len(video_list) / (train_val_test_ratio + 1)))
val_idx = val_test_idx[:len(val_test_idx)//2]
test_idx = val_test_idx[len(val_test_idx)//2:]

# 输出路径
out_dir_train = "datasets/ytvis_2019_new/train/JPEGImages"
out_dir_val = "datasets/ytvis_2019_new/valid/JPEGImages"
out_dir_test = "datasets/ytvis_2019_new/test/JPEGImages"

if not os.path.exists(out_dir_train):
    os.makedirs(out_dir_train)

if not os.path.exists(out_dir_val):
    os.makedirs(out_dir_val)

if not os.path.exists(out_dir_test):
    os.makedirs(out_dir_test)

train_data = []
train_anno = []
val_data = []
val_anno = []
test_data = []
test_anno = []

for idx in range(len(video_list)):
    if idx in val_idx:  # val
        val_data.append(video_list[idx])
        video_id = video_list[idx]['id']
        for anno in annotations_list:
            if anno['video_id'] == video_id:
                val_anno.append(anno)

        filenames = video_list[idx]['file_names']
        for filename in filenames:
            file_path = os.path.join(in_dir, filename)
            file_dir = filename.split('/')[0]
            print(file_dir)
            if not os.path.exists(os.path.join(out_dir_val, file_dir)):
                os.makedirs(os.path.join(out_dir_val, file_dir))
            shutil.copy(file_path, os.path.join(out_dir_val, filename))

    elif idx in test_idx:  # test
        test_data.append(video_list[idx])
        video_id = video_list[idx]['id']
        for anno in annotations_list:
            if anno['video_id'] == video_id:
                test_anno.append(anno)

        filenames = video_list[idx]['file_names']
        for filename in filenames:
            file_path = os.path.join(in_dir, filename)
            file_dir = filename.split('/')[0]
            print(file_dir)
            if not os.path.exists(os.path.join(out_dir_test, file_dir)):
                os.makedirs(os.path.join(out_dir_test, file_dir))
            shutil.copy(file_path, os.path.join(out_dir_test, filename))

    else:
        train_data.append(video_list[idx])
        video_id = video_list[idx]['id']
        for anno in annotations_list:
            if anno['video_id'] == video_id:
                train_anno.append(anno)

        filenames = video_list[idx]['file_names']
        for filename in filenames:
            file_path = os.path.join(in_dir, filename)
            file_dir = filename.split('/')[0]
            print(file_dir)
            if not os.path.exists(os.path.join(out_dir_train, file_dir)):
                os.makedirs(os.path.join(out_dir_train, file_dir))
            shutil.copy(file_path, os.path.join(out_dir_train, filename))

print("Creating Annotation...")

# train
data['videos'] = train_data
data['annotations'] = train_anno
train_file = open('datasets/ytvis_2019_new/train.json', 'w')
train_file.write(json.dumps(data))
train_file.close()

# val
data['videos'] = val_data
data['annotations'] = val_anno
val_file = open('datasets/ytvis_2019_new/valid.json', 'w')
val_file.write(json.dumps(data))
val_file.close()

# test
data['videos'] = test_data
data['annotations'] = test_anno
test_file = open('datasets/ytvis_2019_new/test.json', 'w')
test_file.write(json.dumps(data))
test_file.close()

os.rename("datasets/ytvis_2019", 'datasets/ytvis_2019_bak')
os.rename("datasets/ytvis_2019_new", 'datasets/ytvis_2019')

print("Complete!")

