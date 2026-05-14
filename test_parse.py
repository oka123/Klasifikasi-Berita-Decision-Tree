import pandas as pd
import csv

content = '''Class Index,Title,Description
3,Fears for T N pension after talks,Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul.
4,The Race is On: Second Private Team Sets Launch Date for Human Spaceflight (SPACE.com),"SPACE.com - TORONTO, Canada -- A second\\team of rocketeers competing for the  #36
4,Ky. Company Wins Grant to Study Peptides (AP),"AP - A company founded by a chemistry researcher at the University of Louisville won a grant to develop a method of producing better peptides, which are short chains of amino acids, the building blocks of proteins."
'''
lines = content.splitlines()

parsed_rows = []
for line in lines:
    if not line.strip():
        continue
    try:
        row = next(csv.reader([line], delimiter=','))
        parsed_rows.append(row)
    except Exception as e:
        print('Error parsing:', line, '->', e)
        parsed_rows.append(line.split(','))

if len(parsed_rows) > 0:
    header = parsed_rows[0]
    num_cols = len(header)
    valid_data = []
    for row in parsed_rows[1:]:
        if len(row) == num_cols:
            valid_data.append(row)
        elif len(row) > num_cols:
            valid_data.append(row[:num_cols-1] + [','.join(row[num_cols-1:])])
        else:
            valid_data.append(row + [None] * (num_cols - len(row)))
    
    df = pd.DataFrame(valid_data, columns=header)
    print(df)
