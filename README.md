I'll release latest code and training pipeline as I return school at request, cause codes are left there, this repo is just a backup commit.

## Rotated-Yolov3
Rotating target detection implemented with yolov3.

Not good enough yet, reach only Hmean 70 on ICDAR15 dataset. 

I'll not keep updating here, but PRs are welcomed. Better detector for rotation object detection will be published in my repo as soon as possible(that's why I deprecated ryolo). 

## Support 
* SEBlock  
* CUDA RNMS  
* riou loss  
* Inception module  
* DCNv2  
* ORN  
* SeparableConv
* Mish/Swish
* GlobalAttention

## Notice  
Feel free to contact me if you have any question when use this code, cause maybe I don't know either.(too long the last time I make modification on it, and I don't think yolo is a good choice for arbitrary orientation object detection.)  
I'll release a stronger detector later.  


## Some Results
<div align=center><img  src="https://github.com/ming71/rotate-yolo/blob/master/1.jpg"/></div>
<div align=center><img  src="https://github.com/ming71/rotate-yolo/blob/master/2.jpg"/></div>

## Q&A

Following questions are frequently mentioned. And if you have something unclear, don't doubt and contact me via  opening issues. 

* Q: How can I obtain  `icdar_608_care.txt`?

  A: `icdar_608_care.txt` sets the initial anchors generated via kmeans, you need to run `kmeans.py` refer to my implemention  [here](https://github.com/ming71/toolbox/blob/master/kmeans.py). You can also check `utils/parse_config.py` for more details.

* Q: How to train the model on my own dataset?

  A: This ryolo implemention is based on this [repo](https://github.com/ultralytics/yolov3),  training and evaluation pipeline are the same as that one do.

* Q: Where is ORN codes?

  A: I'll release the whole codebase as I return school, and this [repo](https://github.com/ming71/CUDA/tree/master/ORN) may help.

