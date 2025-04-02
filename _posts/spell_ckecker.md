We write a short and simple code for spell checking. For a given word, we simply check the distance between it and the words in NLTK corpus. Either Jaccard or Edit distance can be used to measure the distances.


```python
import nltk
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
nltk.data.path.append("assets/")
nltk.download('words')

from nltk.corpus import words
correct_spellings = words.words()

from nltk.util import ngrams
from nltk.metrics import distance
```

    [nltk_data] Downloading package words to
    [nltk_data]     /Users/mohsenkarkheiran/nltk_data...
    [nltk_data]   Package words is already up-to-date!



```python
def spell_check(Word, dist_type, ng):

    
    recoms = []
    
    def sec(e):
        return e[1]
    
    
    ls = [ws for ws in correct_spellings if ws[0]==Word[0]]
    label1 = set(ngrams(Word,ng))
        
    dists = []
        
    for ws in ls:
        label2 = set(ngrams(ws,ng))
        if dist_type == 'jaccard':
            d = distance.jaccard_distance(label1, label2)  
        elif dist_type == 'edit':
            d = distance.edit_distance(Word, ws)
            
        dists = dists + [(ws, d)]
        
    dists.sort(key=sec)
        
        
            
    return dists[0][0], dists[1][0], dists[2][0]
    
```


```python
spell_check('gooshh', dist_type = 'edit', ng = 3)
```




    ('goolah', 'goose', 'goosish')




```python
spell_check('gooshh', dist_type = 'jaccard', ng = 3)
```




    ('goose', 'goosy', 'goosery')




```python

```
