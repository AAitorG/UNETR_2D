<h1 align="center">UNETR_2D and YNETR_2D</h1>
<p align="center">2D implementation in Tensorflow2-Keras of UNETR [1], and YNETR a new proposed architecture, for EM image segmentation.</p>
 
---
 
## Install
By running the following commands in bash, you will create and start using a new python environment called 'tutorial-env' (you can change it). Then, you will clone this repository and install the dependencies, specified in [requirements.txt](requirements.txt).
 
```Bash
python3 -m venv tutorial-env
source tutorial-env/bin/activate
git clone https://github.com/AAitorG/UNETR_2D.git UNETR_2D
cd UNETR_2D/
python3 -m pip install -r requirements.txt
```
 
## Usage
There are two different ways to work:
-   ```Bash
    python3 main.py
    ```
    With this script, it is easier to run multiple experiments, and repetitions. By default, the best hyperparameters are set (without SSL pre-text training). This script generates a .csv with every run details and results.
 
-   [quickstart.ipynb](quickstart.ipynb) (also accessible in [colab](https://colab.research.google.com/github/AAitorG/UNETR_2D/blob/main/quickstart.ipynb))
   
    With this notebook, it is easier to do a single run. The default parameters used in the notebook correspond to UNETR-2D Mini. Although the best hyperparameters have not been selected by default, due to the high time consumption, another configuration has been selected that makes use of OneCycle and fewer epochs, which is faster, and also achieves very good results.
 
Both approaches have a 'Parameters' section where preferred parameters can be set before starting the training. For “basic” changes, modifying values of this section should be enough.
 
In order to train for reconstruction, you just need to change the loss function to 'mse' and specify which image alterations you want to use.
 
With an interesting performance, proposed **YNETR_2D** architecture can also be used just by setting it in the 'Parameters' section.
 
Regarding the dataset, in `quickstart.ipynb` the Lucchi dataset is downloaded and used by default. But in `main.py` you must specify in the 'Parameters' section the path of the dataset you want to use. Keep in mind that every dataset must be organized as follows:
```
    data/
        |-- train/
        |    |-- x/
        |    |      training-0001.png
        |    |      ...
        |    |-- y/
        |    |      training_groundtruth-0001.png
        |    |        ...
        |-- test/
        |    |-- x/
        |    |      testing-0001.png
        |    |      ...
        |    |-- y/
        |    |      testing_groundtruth-0001.png
        |    |      ...
 
```
## License
    MIT
 
## References
```
[1] Hatamizadeh, A., Tang, Y., Nath, V., Yang, D., Myronenko, A., Landman, B., ... & Xu, D. (2022). Unetr: Transformers for 3d medical image segmentation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 574-584).
```
