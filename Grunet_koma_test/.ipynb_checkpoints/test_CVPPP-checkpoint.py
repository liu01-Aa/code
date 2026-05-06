from os import listdir
from os.path import join
import model as vo
from deepcoloring.data import ssReader, test_DATA
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '../..')
import torch
from torchvision import transforms
import numpy as np
from PIL import  Image
from utils import _get_ins_seg_masks, process_sem_nbj, visualization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dir_in = "data/"
def evalution(test_dataloader, model, max_n_objects=20, data_test_names=None):
    model.eval()
    print("Model loaded")
    sem_pred = []
    ins_pred = []
    cout = 0
    for batch in test_dataloader:
        source, images_= batch

        # torch.Size([1, 530, 500, 3]) (b, h, w,c)

        images_ = images_.cuda()

        sem_pred_, ins_pred_, n_objects_ = model(images_)

        ins_pred_ = ins_pred_.detach().cpu().numpy()
        sem_pred_, n_objects_ = process_sem_nbj(sem_pred_, n_objects_, max_n_objects)

        sem_pred.append(sem_pred_)
        ins_pred.append(ins_pred_)

        p_sem_pred, ins_masks, ins_color_imgs = _get_ins_seg_masks(sem_pred, ins_pred, n_objects_)
        ins_masks = Image.fromarray(np.uint8(ins_masks[0]))
        ins_masks = ins_masks.resize(size=(500, 530), resample=Image.NEAREST)
        # ins_masks = Image.fromarray(np.uint8(ins_color_imgs[0]))

        # p_sem_pred = np.asarray(p_sem_pred[0]) * 255
        #
        # p_sem_pred = Image.fromarray(np.uint8(p_sem_pred))




        # size = (38, 23, 538, 553)
        # ins_masks = ins_masks.crop(size)
        # # ins_masks = np.asarray(ins_masks)
        # ins_masks.save(dir_in+data_test_names[cout]+'_ins_label.png')
        #
        # p_sem_pred = p_sem_pred.crop(size)
        # p_sem_pred.save(dir_in+data_test_names[cout]+'_seg_label.png')


        #
        np.savetxt(data_test_names[cout]+'.txt', ins_masks, fmt='%d', delimiter=' ')
        # print("{},  {}".format(data_test_names[cout], cout))
        # print(ins_masks[256][256])
        # print(ins_masks[200][200])
        #
        # visualization(p_sem_pred, ins_masks, data_test_names[cout])
        # exit()


        cout = cout + 1
        print(cout)
        sem_pred = []
        ins_pred = []

if __name__ == "__main__":

    names = ['plant003', 'plant004', 'plant009', 'plant014', 'plant019', 'plant023', 'plant025', 'plant028', 'plant034',
     'plant041', 'plant056', 'plant066', 'plant074', 'plant075', 'plant081', 'plant087', 'plant093', 'plant095',
     'plant097', 'plant103', 'plant111', 'plant112', 'plant117', 'plant122', 'plant125', 'plant131', 'plant136',
     'plant140', 'plant150', 'plant155', 'plant157', 'plant158', 'plant160']
    basepath = "../Dataset/CVPPP/A1/test dataset/"

    t_rgb = sorted([join(basepath, f) for f in listdir(basepath) if f.endswith('.png')])

    # transforms.RandomRotation((30, 60)),
    transform_test = transforms.Compose(
        [transforms.Resize((448, 448)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ])


    test_data = test_DATA(image_list=t_rgb, transform=transform_test, use_cache=True)

    net = vo.GrsUnet(out_chans=20).cuda()
    net.load_state_dict(torch.load("Grsunet.th"))
    net.eval()



    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    evalution(test_dataloader, net, max_n_objects=20, data_test_names=names)

