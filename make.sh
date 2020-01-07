#!/usr/bin/env bash
cd model/layer/ORN
./build.sh

cd ../../layer/DCNv2
./build.sh

cd ../../../utils/nms
./build.sh
