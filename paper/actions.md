## Teste tradicional
    
### ACC mensal alexnet sem saída multipla ao longo do tempo
### ACC mensal mobilenet sem saída multipla ao longo do tempo
    /home/jasimioni/ppgia/greencomputing2024/paper/evaluation/ChartErrorRate
    
    TPR / TNR ao invés de FPR / FNR
    
### Custo computacional alexnet sem saída multipla em desktop e dispositivo
### Custo computacional mobilenet sem saída multipla em desktop e dispositivo
    /home/jasimioni/ppgia/greencomputing2024/paper/evaluation/proc_rate.txt
    
### Custo energético alexnet sem saída multipla em dispositivo
### Custo energético mobilenet sem saída multipla em dispositivo
    /home/jasimioni/ppgia/greencomputing2024/paper/evaluation/EnergyConsumption.py
    
    Por que utilizamos essas duas redes? Achar uma justificativa da escolha, com base na literatura.
    AlexNet - Referência
    MobileNet - Uso especial para dispositivos de baixo poder

## Proposta
    P1 - Early exits (não é novidade, vamos focar em ter ganho computacional e energetivo, acurácia é secundário, o que queremos é não ter impacto na acc)
    P2 - Calibracao das confianças nas saídas (não é novidade, vamos apenas fazer com que tenha relação da conf com acc)
    P3 - Selecao multi-objetivo REJ vs ACC (novidade, )
    P4 - Atualizacao baseado nas rejs (novidade)
    P5 - Implementacao de uma arquitetura para offloading (novidade)

## Avaliacao da Proposta

### P2 - Avaliar calibracao e mostrar que faz com que a confiança tenha relação com ACC
    GRAFICO BARRA COM BINS ACC E CONFIANCA COM E SEM CALIBRACAO
    /home/jasimioni/ppgia/greencomputing2024/paper/evaluation/CalibrationComparison/
    
    => Usar Janeiro com o algoritmo naive (aproximaçõe sucessivas)
        /home/jasimioni/ppgia/greencomputing2024/calibration/calibrated/noob_algo/AlexNetWithExits/01
        /home/jasimioni/ppgia/greencomputing2024/calibration/calibrated/noob_algo/MobileNetWithExits/01

### P3 - Avaliar seleção multi-objetivo 
    GRAFICO LINHA COM TEMPO vs ACC em fevereiro para alexnet e mobilenet com 2 saidas
    /home/jasimioni/ppgia/greencomputing2024/nsga2/GenFig3ParetoFig4AccFig5Comp.py
    /home/jasimioni/ppgia/greencomputing2024/nsga2/paretocomp_calibrated.pdf

### P1 – Avaliar acc mensal sem atualização usando os pontos de operação definidos anteriormentes
    Fizemos a análise com F1 - posso mudar pra ACC
    /home/jasimioni/ppgia/greencomputing2024/nsga2/f1comp_calibrated.pdf

### P4 – Avaliar acc mensal com atualização mensal usando os eventos rejeitados da avaliação anterior
    Não funciona:
    /home/jasimioni/ppgia/greencomputing2024/reinforced/gradual-train.log

### P5 – Avaliar custo computacional, latência, custo energético da proposta fazendo offloading para cloud, comparar várias zonas na nuvem
    Custo computacional e tempo de execução:
    /home/jasimioni/ppgia/greencomputing2024/paper/evaluation/mq/side_by_side.html

    Custo Energético
    Converter para matplotlib
    https://docs.google.com/spreadsheets/d/1mz9FeGGzptKERec5ZN3vaJsfVdjiKhgSIJhfVWJSnDo/edit?gid=0#gid=0

