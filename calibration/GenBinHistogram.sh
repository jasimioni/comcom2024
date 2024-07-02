#!/bin/bash

# Gera todos os gráficos dos bins para os modelos calibrados e não calibrados

for network in AlexNet MobileNet
do
    for month in 01 02 03 04 05 06 07 08 09 10 11 12
    do
        ./GenBinHistogramChart.py --savefolder non_calibrated/${network}WithExits/ \
                                  --datafile non_calibrated/${network}WithExits/2016_${month}.csv \
                                  --title "Non Calibrated ${network} ${month}"
        for calibration in 01 02 03
        do
            ./GenBinHistogramChart.py --savefolder calibrated/${network}WithExits/${calibration}/ \
                                      --datafile calibrated/${network}WithExits/${calibration}/2016_${month}.csv \
                                      --title "Calibrated ${calibration} ${network} ${month}"
        done
    done
done
