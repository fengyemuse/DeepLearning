import os
import shutil

src_dir = r'D:\work\Deeplearning_Data\cats_dogs\train'
base_dir = r'D:\work\Deeplearning_Data\cats_dogs\cats_vs_dogs_data'


def image_copy(copy_target='cats', dirs=None):
    if dirs is None:
        dirs = [train_cats_dir, validation_cats_dir, test_cats_dir]
    fnames = ['{0}.{1}.jpg'.format(copy_target, i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dirs[0], fname)
        shutil.copyfile(src, dst)
    fnames = ['{0}.{1}.jpg'.format(copy_target, i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dirs[1], fname)
        shutil.copyfile(src, dst)
    fnames = ['{0}.{1}.jpg'.format(copy_target, i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dirs[2], fname)
        shutil.copyfile(src, dst)


if __name__ == '__main__':
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        train_dir = os.path.join(base_dir, 'train')
        os.makedirs(train_dir)
        validation_dir = os.path.join(base_dir, 'validation')
        os.makedirs(validation_dir)
        test_dir = os.path.join(base_dir, 'test')
        os.makedirs(test_dir)
        train_cats_dir = os.path.join(train_dir, 'cats')
        os.makedirs(train_cats_dir)
        train_dogs_dir = os.path.join(train_dir, 'dogs')
        os.makedirs(train_dogs_dir)
        validation_cats_dir = os.path.join(validation_dir, 'cats')
        os.makedirs(validation_cats_dir)
        validation_dogs_dir = os.path.join(validation_dir, 'dogs')
        os.makedirs(validation_dogs_dir)
        test_cats_dir = os.path.join(test_dir, 'cats')
        os.makedirs(test_cats_dir)
        test_dogs_dir = os.path.join(test_dir, 'dogs')
        os.makedirs(test_dogs_dir)

        image_copy(copy_target='cat', dirs=[train_cats_dir, validation_cats_dir, test_cats_dir])
        image_copy(copy_target='dog', dirs=[train_dogs_dir, validation_dogs_dir, test_dogs_dir])
