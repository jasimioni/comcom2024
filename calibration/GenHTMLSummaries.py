#!/usr/bin/env python3

with open('Compare1exit.html', 'w') as file:
    for network in [ 'AlexNet', 'MobileNet' ]:
        for m in range(12):
            month = f'{m+1:02d}'
            name = f"{network}_2016_{month}_1exit_total"
            print(f"<img width='400px' src='pre_calibration/{network}/{name}.png'>", file=file)
            print(f"<img width='400px' src='calibrated/{network}/Calibrated_{name}.png'><br />", file=file)

with open('Compare2exit.html', 'w') as file:
    for network in [ 'AlexNet', 'MobileNet' ]:
        for m in range(12):
            month = f'{m+1:02d}'
            name = f"{network}EE_2016_{month}_exit_1_total"
            print(f"<img width='400px' src='pre_calibration/{network}EE/{name}.png'>", file=file)
            print(f"<img width='400px' src='calibrated/{network}WithExits/Calibrated_{name}.png'>", file=file)
            name = f"{network}EE_2016_{month}_exit_2_total"
            print(f"<img width='400px' src='pre_calibration/{network}EE/{name}.png'>", file=file)
            print(f"<img width='400px' src='calibrated/{network}WithExits/Calibrated_{name}.png'><br />", file=file)