from pycocotools.coco import COCO
import requests
import os

def download_images(coco, category_names, download_dir, limit=100):
    catIds = coco.getCatIds(catNms=category_names)
    imgIds = coco.getImgIds(catIds=catIds)
    imgIds = imgIds[:limit]
    imgs = coco.loadImgs(imgIds)

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    for img in imgs:
        img_url = img['coco_url']
        img_filename = os.path.join(download_dir, img['file_name'])
        print(f"Downloading {img_filename}...")
        
        try:
            r = requests.get(img_url, stream=True)
            if r.status_code == 200:
                with open(img_filename, 'wb') as f:
                    for chunk in r:
                        f.write(chunk)
            else:
                print(f"Failed to download {img_url}")
        except Exception as e:
            print(f"Error downloading {img_url}: {str(e)}")

def mainDownloadFunction():
    dataDir = '/Users/caseytirrell/Downloads/annotations'
    dataType = 'train2017'
    annFile = f'{dataDir}/instances_{dataType}.json'

    coco = COCO(annFile)

    category_names = [
        'building',
        'person',
        'bicycle',
        'car',
        'tree',
        'bus',
        'backpack',
        'ball',
        'kite',
        'chair',
        'teddy bear',
        'book',
        'clock',
        'scissors',
        'toys'
    ]
    download_dir = "/Users/caseytirrell/Documents/CS370-Assignments/Assignments 2"
    download_images(coco, category_names, download_dir, limit=100)

if __name__ == "__main__":
    mainDownloadFunction()
