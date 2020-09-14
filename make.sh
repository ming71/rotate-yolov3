#!/usr/bin/env bash
cd model/layer/ORN
./make.sh

cd ../../layer/DCNv2
./make.sh

cd ../../../utils/nms
./make.sh
