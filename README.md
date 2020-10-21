## Rotated-Yolov3

Rotaion object detection implemented with yolov3.

---

Hello, the no-program [ryolov3](https://github.com/ming71/yolov3-polygon) is available now. Although not so many tricks are attached like this repo, it still achieves good results, and is friendly for beginners to learn, have a good luck.

## Update

The latest code has been uploaded, unfortunately, due to my negligence, I incorrectly modified some parts of the code and did not save the historical version last year, which made it hard to reproduce the previous high performance. It is tentatively that there are some problems in the loss calculation part. 

But I found from the experimental results left last year that yolov3 is suitable for rotation detection. After using several tricks (attention, ORN, Mish, and etc.), it have achieved good performance. More previous experiment results can be found [here](https://github.com/ming71/rotate-yolo/blob/master/experiment).

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

## Detection Results

The detection results from rotated yolov3 left over last year:

<div align=center><img  src="https://github.com/ming71/rotate-yolo/blob/master/demo.png"/></div>

## Q&A

Following questions are frequently mentioned. And if you have something unclear, don't doubt and contact me via  opening issues. 

* Q: How can I obtain  `icdar_608_care.txt`?

  A: `icdar_608_care.txt` sets the initial anchors generated via kmeans, you need to run `kmeans.py` refer to my implemention  [here](https://github.com/ming71/toolbox/blob/master/kmeans.py). You can also check `utils/parse_config.py` for more details.

* Q: How to train the model on my own dataset?

  A: This ryolo implemention is based on this [repo](https://github.com/ultralytics/yolov3),  training and evaluation pipeline are the same as that one do.

* Q: Where is ORN codes?

  A: I'll release the whole codebase as I return school, and this [repo](https://github.com/ming71/CUDA/tree/master/ORN) may help.

* Q: I cannot reproduce the result you reported(80 mAP for hrsc and 0.7 F1 for IC15).
* A: Refer to my reply [here](https://github.com/ming71/rotate-yolov3/issues/14#issuecomment-663328130). This is only a backup repo, the overall model is no problem, but **direct running does not necessarily guarantee good results**, cause it is not the latest version, and some parameters may have problems, you need to adjust some details and parameter settings yourself. 
  I will upload the complete executable code as soon as I return to school in September (if lucky).

## In the end
There is no need or time to maintain the codebase to reproduce the previous performance. If you are interested in this work, you are welcome to fix the bugs in this codebase, and the trained models are available [here](https://pan.baidu.com/s/1EXhyGSiuUIPnkZ7cwpfCbQ) with extracted code `5noq` . I'll reimplement the rotation yolov4 or yolov5 if time permitting  in the future.