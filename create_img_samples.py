import argparse
import sys

# my libraries
import utils
import image_processing as img_p


def main():
    path = utils.getParentDir()
    policy_dir = utils.joinPath(path, 'policies')
    bkg_dir = utils.joinPath(path, 'backgrounds')
    save_dir = utils.joinPath(path, 'dataset')
    classification_dir = utils.joinPath(save_dir, 'classification')
    imgs_dir = utils.joinPath(save_dir, 'images')

    # get argumennts
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-dir", type=str, default=policy_dir,
                        help="directory of documents")
    parser.add_argument("--background-dir", type=str, default=bkg_dir,
                        help="directory of backgrounds")
    parser.add_argument("--save-path", type=str, default=save_dir,
                        help='path to save images')
    args = parser.parse_args()

    policy_imgs = utils.filesInDir(args.policy_dir, 'jpg')
    bkg_imgs = utils.filesInDir(args.background_dir, 'jpg')
    save_imgs = utils.filesInDir(args.save_path, 'jpg')

    num_bkgs = len(bkg_imgs)
    num_policies = len(policy_imgs)
    num_samples = 15
    rand_bkgs = utils.getRandomRange(0, num_bkgs - 1, num_samples)
    rand_policies = utils.getRandomRange(0, num_policies - 1, num_samples)

    for i in range(num_samples):
        img_f = img_p.readImg(policy_imgs[rand_policies[i]])
        img_b = img_p.readImg(bkg_imgs[rand_bkgs[i]])
        dataset_img, class_img = img_p.createImg(img_f, img_b)
        img_name = 'img_' + str(i) + '.jpg'
        full_img_path = utils.joinPath(imgs_dir, img_name)
        full_class_path = utils.joinPath(classification_dir, img_name)
        img_p.saveImg(full_img_path, dataset_img)
        img_p.saveImg(full_class_path, class_img)

    # img_p.showImg(res)


if __name__ == '__main__':
    main()
