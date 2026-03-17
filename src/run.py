import os
os.chdir(r"C:\Users\angel\Desktop\Proyectos_privados\PROYECTOS\EasyMoney_Capstone")
from train_model import *
from utils import *


def main():
    # Entrenar
    name_model = "DecisionTreeClassifier"
    scoring = 'f1'


    param_grid = {
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 10, 50],
        'min_samples_leaf': [1, 5, 10]
    }


    metrics = train_model(
        name_model, scoring, param_grid, target="sales_account", act=True)



if __name__ == "__main__":
    main()
