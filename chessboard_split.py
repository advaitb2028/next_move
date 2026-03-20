import cv2
import numpy as np
import os

def get_tiles(img_file, save):

  if save:
    img = cv2.imread(img_file)
  else:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

  height, width, _ = img.shape
  tile_height = height // 8
  tile_width = width // 8

  if not os.path.exists("pieces_dataset"):
    os.makedirs("pieces_dataset")

  count = 0
  all_tiles = []

  for row in range(8):
    for column in range(8):
      y_start = row * tile_height
      y_end = (row + 1) * tile_height
      x_start = column * tile_width
      x_end = (column + 1) * tile_width

      tile = img[y_start:y_end, x_start:x_end]
      bnw_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

      all_tiles.append(bnw_tile)

      if save == True:
        tile_name = f"{os.path.basename(img_file)}_tile_{count}.jpg"
        cv2.imwrite(os.path.join("pieces_dataset", tile_name), bnw_tile)

      count += 1

  if save == True:
    print(f"{count} images saved. Process completed")

  return all_tiles

  # END FUNCTION

def main():
  for raw_board in os.listdir("images"):
    if raw_board.lower().endswith(('.png', '.jpg', '.jpeg')):
      get_tiles(f"images/{raw_board}", True)

if __name__ == "__main__":
  """
  for every file in raw_images, split and save into new folder
  decide if making grayscale here or in model input
  and then need to manually label all images
  ALSO, where to SAVE?
  """
  main()
