import os
import random
import shutil

if __name__ == "__main__":

    split = 0.8  # training set ratio
    origin_data_root = r"E:\Education\Master_Course\Image_process\group_project\dataset\flower102\jpg"
    train_data_root = r"E:\Education\Master_Course\Image_process\group_project\dataset\flower102\train"
    test_data_root = r"E:\Education\Master_Course\Image_process\group_project\dataset\flower102\test"

    origin_data_list = os.listdir(origin_data_root)
    train_data_list = random.sample(origin_data_list, int(len(origin_data_list) * split))
    test_data_list = [i for i in origin_data_list if i not in train_data_list]
    # test_data_list = list(set(origin_data_list) - set(train_data_list))

    print(f"training set: {len(train_data_list)}")
    print(f"test set: {len(test_data_list)}")

    try:
        os.mkdir(train_data_root)
        os.mkdir(test_data_root)
    except FileExistsError:
        pass

    for file in train_data_list:
        shutil.copy(os.path.join(origin_data_root, file), os.path.join(train_data_root, file))

    for file in test_data_list:
        shutil.copy(os.path.join(origin_data_root, file), os.path.join(test_data_root, file))

    print("split finished")



