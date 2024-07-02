#!/bin/bash

export PYTHONUNBUFFERED=1

# Utiliza o dataset de 01 02 03 para calibrar os modelos, calculando
# a temperatura a ser aplicada aos logits. Isso gera um modelo novo
# ajustado aos dados daquele mês. A ideia é ter 3 meses diferentes pra
# Verificar o comportamento usando:
# 01 - Mês de treinamento
# 02 - Mês de validação (+/-)
# 03 - Mês adicional pra ver

# Automaticamente executa a inferência do modelo e salva os resultados
# com o tempo de execução igual a zero. Depois dá pra adicionar os tempos
# do raspberry usando o AddTimesToCalibratedCsv.py

for MONTH in 01 02 03
do

    AN_TRAINED_MODEL="../trained_models/AlexNetWithExits_epoch_19_90.1_91.1"
    AN_DST_DIR="calibrated/AlexNetWithExits/${MONTH}/"
    [[ -d $AN_DST_DIR ]] || mkdir -p $AN_DST_DIR

    ./Calibrate2exits.py --batch-size 1000 --model alexnet --trained-model ${AN_TRAINED_MODEL}.pth \
                         --calibrated-model-savefile ${AN_TRAINED_MODEL}_${MONTH}.pth --dataset ../MOORE/2016_${MONTH}.csv \
                         --savefolder $AN_DST_DIR | tee $AN_DST_DIR/calibration.log

    MN_TRAINED_MODEL="../trained_models/MobileNetV2WithExits_epoch_19_89.7_90.9" 
    MN_DST_DIR="calibrated/MobileNetWithExits/${MONTH}/"
    [[ -d $MN_DST_DIR ]] || mkdir -p $MN_DST_DIR

    ./Calibrate2exits.py --batch-size 250 --model mobilenet --trained-model ${MN_TRAINED_MODEL}.pth \
                         --calibrated-model-savefile ${MN_TRAINED_MODEL}_${MONTH}.pth --dataset ../MOORE/2016_${MONTH}.csv \
                         --savefolder $MN_DST_DIR | tee $MN_DST_DIR/calibration.log

done
