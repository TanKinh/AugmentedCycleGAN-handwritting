import os
import torch

from edges2shoes_data import DataLoader, load_edges2shoes, AlignedIterator, UnalignedIterator
from PIL import Image, ImageFont, ImageDraw
from model import StochCycleGAN, AugmentedCycleGAN
from functools import reduce
from torch.autograd import Variable
import util
from scipy.misc import imresize
from options import TestLineOption
import json
import torchvision.transforms as transforms
import torchvision.utils as vutils
import pickle as pkl
import random
import numpy as np

def draw_single_char(ch, font, canvas_size=128, x_offset=26, y_offset=36):
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img.convert('RGB')

def save_image(result_dir, image, image_name, aspect_ratio=1.0):
    im = util.tensor2im(image)
    save_path = os.path.join(result_dir, image_name)
    h, w, _ = im.shape
    if aspect_ratio > 1.0:
        im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
    if aspect_ratio < 1.0:
        im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
    util.save_image(im, save_path)
    print('save in path: ', save_path)

def visualize_multi(opt, real_A, model, name='multi_test.png'):
    """Generate multi fake B from real_A and multi Z_B, format image |read_A|multi_fake_B """
    size = real_A.size()
    # all samples in real_A share the same prior_z_B
    multi_prior_z_B = Variable(real_A.data.new(opt.num_multi,
        opt.nlatent, 1, 1).normal_(0, 1).repeat(size[0],1,1,1), volatile=True)
    multi_fake_B = model.generate_multi(real_A.detach(), multi_prior_z_B)
    multi_fake_B = multi_fake_B.data.cpu().view(
        size[0], opt.num_multi, size[1], size[2], size[3])
    vis_multi_image = torch.cat([real_A.data.cpu().unsqueeze(1), multi_fake_B], dim=1) \
        .view(size[0]*(opt.num_multi+1),size[1],size[2],size[3])
        
    save_path = os.path.join(opt.res_dir, name)
    print('print image in path ', save_path)
    vutils.save_image(vis_multi_image.cpu(), save_path,
        normalize=True, range=(-1,1), nrow=opt.num_multi+1)
    i = 0
    print(multi_fake_B[0].size())
    print(type(multi_fake_B[0].numpy()))
    # n_choice = random.randint(0, opt.num_multi)
    # k_b = random.sample(multi_fake_B[0], 3)
    k_b = multi_fake_B[0].numpy()
    print('size : ', k_b.shape)
    print(multi_fake_B[0][0].size())
    # np_b = np.random.choice(k_b, size=(1, 64, 64))
    choice = random.randint(0, opt.num_multi - 1)
    # l_choice = []        
    # for _ in range(1):
    #     l_choice.append(random.randint(0, opt.num_multi -1))
    # print('list choice : ', choice)
    # for ch in l_choice:
    #     i_dir =  opt.res_dir + '/' + 'im_k%d.png'%i
    #     print('print image in path ', i_dir)
    #     vutils.save_image(multi_fake_B[0][ch].cpu(), i_dir, normalize=True, range=(-1, 1), nrow=1)
    #     i += 1
    i_dir =  opt.res_dir + '/' + 'im_k%d.png'%choice
    vutils.save_image(multi_fake_B[0][choice].cpu(), i_dir, normalize=True, range=(-1, 1), nrow=1)
    return multi_fake_B[0][choice]

def get_transform(opt):
    transform_list = [transforms.Resize([64, 64], Image.BICUBIC)]
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def gen_line(text, opt):
    result_img_names = []

    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display

    font = ImageFont.truetype(opt.font, size=opt.font_size)

    print('check point dir ',opt.chk_path)
    # extract expr_dir from chk_path
    expr_dir = os.path.dirname(opt.chk_path)
    opt_path = os.path.join(expr_dir, 'opt.pkl')
    opt.__dict__.update(parse_opt_file(opt_path))
    opt.expr_dir = expr_dir

    # create results directory (under expr_dir)
    res_path = os.path.join(opt.expr_dir, opt.res_dir)
    opt.res_dir = res_path
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    epochs = opt.which_epoch.split(',')
    for epoch in range(10):
        results = []
        inputs = []
        opt.which_epoch = epoch
        model = AugmentedCycleGAN(opt, testing=True)
        model.load(opt.chk_path)

        count = 0
        for ch in text:
            img = draw_single_char(ch, font, x_offset=opt.offset, y_offset=opt.offset)

            transform = get_transform(opt)

            img = transform(img)

            if opt.input_nc == 1:  # RGB to gray
                tmp = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
                img = tmp.unsqueeze(0)

            prior_z_B = Variable(img.new(img.size()[0], opt.nlatent, 1, 1).normal_(0, 1), volatile=True)

            img = img.reshape([1, img.shape[0], img.shape[1], img.shape[2]])
            #inputs.append(img)

            size = img.size()
            img = img.cuda()
            prior_z_B = prior_z_B.cuda()

            multi_prior_z_B = Variable(img.new(opt.num_multi, opt.nlatent, 1, 1).normal_(0, 1).repeat(size[0], 1, 1, 1), volatile=True)
            count += 1
            multi_fake_B = visualize_multi(opt, img, model, name='line' + str(count) +  str(epoch) + '.png')
            results.append(multi_fake_B)

        result = reduce((lambda x, y: torch.cat((x, y), -1)), results)
        # input_img = reduce((lambda x, y: torch.cat((x, y), -1)), inputs)

        # # result_img_name = file_name
        result_img_name = 'result_' + str(epoch) + '_' + text +  '.png'
        # input_img_name = 'input_' + opt.name + '_' + text + '.png'
        # result_img_names.append(result_img_name)
        
        save_image(opt.res_dir, result, result_img_name, aspect_ratio=1.0)
        # save_image(opt.results_dir, input_img, input_img_name, aspect_ratio=opt.aspect_ratio)

    # return {'input': input_img_name, 'result': result_img_names}

def parse_opt_file(opt_path):

    def parse_val(s):
        if s == 'None':
            return None
        if s == 'True':
            return True
        if s == 'False':
            return False
        if s == 'inf':
            return float('inf')
        try:
            f = float(s)
            # special case
            if '.' in s:
                return f
            i = int(f)
            return i if i == f else f
        except ValueError:
            return s

    opt = None
    # print(type(opt_path))
    opt_path = (opt_path.replace('\\','/'))
    with open(opt_path, 'rb') as f:
        if opt_path.endswith('pkl'):
            opt = pkl.load(f)
        else:
            opt = dict()
            for line in f:
                if line.startswith('-----'):
                    continue
                k,v = line.split(':')
                opt[k.strip()] = parse_val(v.strip())
    return opt

if __name__ == '__main__':
    # text = input("Input text: ")
    text = '博多駅前'
    print(str('他'))
    opt = TestLineOption().parse()
    gen_line(text, opt)

    # --chk_path checkpoints/FOLDER/latest --metric visual --which_epoch 50 --offset 6 --res_dir vis_dir