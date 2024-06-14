## File Description:  
  
- **data**: This folder contains the dataset we collected ourselves.  
- **attack**: This folder contains codes related to basic model training, including data collection scripts, data processing, model building, and model training.  
- **evaluation**: This folder contains experimental codes for attack and defense methods based on noise filtering.  
- **attack_aug**: This folder contains experimental codes for attack methods based on noise augmentation.  
  
## Basic Environment:  
  
- Anaconda 4.5.11, Jupyter Notebook with Python 3.7.0.  
- TensorFlow 2.10.0 as the backend server and Keras 2.10.0 as the frontend terminal.  
  
Additionally, there are basic packages such as numpy, matplotlib, pandas, pyautogui, and dpkt, which you can install using pip.  
  
## Dataset Description:  
  
The dataset is located in the **data** directory and is in .csv format. You can load them using the following code:  
  
```python
import pandas as pd
df = pd.read_csv('../data.csv', header=None)
```

Each row in the dataset represents a traffic instance, where the first item in each row is the index, the second item is the number of repetitions, and the subsequent items are traffic data.

## Notes:  
  
- The path in the code may not match your path, please pay attention to replace it.
