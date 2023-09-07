from model import available_control_net, run_control_net
from utils import *




init_img = get_an_image_resize('./IU.png', (560, 480))
pose_img = get_images_resize('./open_pose_images', hw=(560, 480))
images = split_by_batch_size(pose_img, 3)



for i in pose_img:
    images = run_control_net()
