# How to Import Modules in Python

### init.py

you need to ensure that Python treats your directories as packages. Here's a step-by-step guide:


Add an `__init__.py` to each dir you want to use as a module for import

```
root
|-- src/
    |-- service.py
|-- utils/
    |-- models.py
    |-- __init__.py
```

then use absolute imports `from utils import models`

from utils import models





### Relative imports

navigate to the desired dir based on the calling file's location in the project structure. Out of app/ into utils/ then get models.py

```
# app/main.py
from ..utils import models  # Use double dots to go up one level

# Access functions or classes from models.py:
my_model = models.SomeModel()
result = models.some_function(data)
```