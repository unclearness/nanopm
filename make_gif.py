from PIL import Image, ImageDraw
import os

def make_gif(root_dir, prefix, ext):
  images = []

  im_files = sorted([x for x in os.listdir(root_dir) if x.startswith(prefix) and x.endswith(ext)])

  for im_file in im_files:
    path = os.path.join(root_dir, im_file)
    im = Image.open(path)
    images.append(im)

  images[0].save(prefix + '.gif',
                save_all=True, append_images=images[1:], optimize=False, duration=200, loop=0)


if __name__ == "__main__":
  make_gif('win_build', 'nnf', 'jpg')
  make_gif('win_build', 'distance', 'jpg')
  make_gif('win_build', 'recon', 'jpg')