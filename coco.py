# get ms-coco dataset
# make sure at least 24GB storage available
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("coco-2017")
print("COCO-2017 downloaded successfully!")