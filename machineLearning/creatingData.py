import numpy as np
import pandas as pd


humans = ['John', 'Caterina', 'Arturo', 
          'Norberto', 'Giorgio', 'Federica',
          'Vincenza', 'Saarbruecken', 'Cologne',
          'Constance', 'Freiburg', 'Karlsruhe'
         ]

n= len(humans)
data = {'Height': np.random.normal(175, 10, n),
        'Weight': np.random.normal(65, 6, n),
        'Age': np.random.normal(25, 20, n)
       }
df = pd.DataFrame(data=data, index=humans)
print(df)