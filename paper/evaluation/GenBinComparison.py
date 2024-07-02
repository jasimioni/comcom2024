#!/usr/bin/env python3

import os

calibrated_base_folder = '../../../calibration/calibrated'
non_calibrated_base_folder = '../../../calibration/non_calibrated'

width = '450px'

for calibration in [ '01', '02', '03' ]:
    report_file = f'CalibrationComparison/{calibration}.html'
    
    content = f'''
    <html>
        <head>
            <title>Calibration Comparison - Base Month {calibration}</title>
            <style>
                table {{
                    border-collapse: collapse;
                    width: 100%;
                }}
                th, td {{
                    border: 1px solid black;
                    padding: 8px;
                    text-align: center;
                }}
                .c1 {{
                    background-color: #f2f2f2;
                }}
                .c2 {{
                    background-color: #e6ffe6;
                }}
                .c3 {{
                    background-color: #222222;
                }}
                .c4 {{
                    background-color: #000000;
                }}
                h1 {{
                    text-align: center;
                }}
            </style>
        <body>
    '''   
    content += f'<h1>Calibration {calibration}</h1>'
    content += f'<table>'
    for month in range(1, 13):
        month = f'{month:02d}'
        content += f'<tr><td colspan="4"><h2>2016 - {month}</h2></td></tr>'
        for network in [ 'AlexNetWithExits', 'MobileNetWithExits' ]:
            content += '<tr>'
            content += f'<td class="c1"><img width="{width}" src="{non_calibrated_base_folder}/{network}/2016_{month}_exit_1.png"></td>'
            content += f'<td class="c1"><img width="{width}" src="{calibrated_base_folder}/{network}/{calibration}/2016_{month}_exit_1.png"></td>'
            content += f'<td class="c2"><img width="{width}" src="{non_calibrated_base_folder}/{network}/2016_{month}_exit_2.png"></td>'
            content += f'<td class="c2"><img width="{width}" src="{calibrated_base_folder}//{network}/{calibration}/2016_{month}_exit_2.png"></td>'
            content += '</tr>'
    
    content += '''
        </body>
    </html>
    '''
    
    with open(report_file, 'w') as f:
        f.write(content)    
            
            
            


