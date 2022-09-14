import argparse

# my libraries
import utils
import image_processing as img_p


def main():
    path = utils.getParentDir()
    policy_dir = utils.joinPath(path, 'policies')
    bkg_dir = utils.joinPath(path, 'backgrounds')
    save_dir = utils.joinPath(path, 'images')
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

    img_f = img_p.readImg(policy_imgs[0])
    img_b = img_p.readImg(bkg_imgs[0])
    res = img_p.createImg(img_f, img_b)
    img_p.showImg(res)


if __name__ == '__main__':
    main()
