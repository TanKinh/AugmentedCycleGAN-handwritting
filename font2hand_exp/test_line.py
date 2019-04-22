import os
import torch

from font2hand_data import DataLoader, load_font2hand, AlignedIterator, UnalignedIterator
from PIL import Image, ImageFont, ImageDraw
from model import StochCycleGAN, AugmentedCycleGAN
from functools import reduce
from torch.autograd import Variable
import json

def draw_single_char(ch, font, canvas_size=128, x_offset=26, y_offset=36):
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img.convert('RGB')

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
    vutils.save_image(vis_multi_image.cpu(), save_path,
        normalize=True, range=(-1,1), nrow=opt.num_multi+1)

def gen_line(text, opt):
    result_img_names = []

    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display

    font = ImageFont.truetype(opt.font, size=opt.font_size)

    epochs = opt.which_epoch.split(',')
    for epoch in epochs:
        results = []
        inputs = []
        opt.which_epoch = epoch
        model = AugmentedCycleGAN(opt, testing=True)
        model.load(opt.chk_path)

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

            multi_prior_z_B = Variable(img.new(opt.num_multi, opt.nlatent, 1, 1).normal_(0, 1).repeat(size[0], 1, 1, 1), volatile=True)
            
            visualize_multi(opt, img, model, name='line%d.png' %i)
            results.append(model.fake_B)

        result = reduce((lambda x, y: torch.cat((x, y), -1)), results)
        input_img = reduce((lambda x, y: torch.cat((x, y), -1)), inputs)

        # result_img_name = file_name
        result_img_name = 'result_' + opt.name + '_' + str(epoch) + '_' + text +  '.png'
        input_img_name = 'input_' + opt.name + '_' + text + '.png'
        result_img_names.append(result_img_name)

        save_image(opt.results_dir, result, result_img_name, aspect_ratio=opt.aspect_ratio)
        save_image(opt.results_dir, input_img, input_img_name, aspect_ratio=opt.aspect_ratio)

    return {'input': input_img_name, 'result': result_img_names}

if __name__ == '__main__':
    text = input("Input text: ")
    print(text)