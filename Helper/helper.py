import numpy as np
import matplotlib.pyplot as plt

def plot_boundary(data, dt1, dt2 ):
    y_train = data['Outcome']
    x1_min, x1_max = data['Glucose'].min() - 1, data['Glucose'].max() + 1
    x2_min, x2_max = data['DiabetesPedigreeFunction'].min() - 1, data['DiabetesPedigreeFunction'].max() + 1
    x1_x, x2_x = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                         np.arange(x2_min, x2_max, 0.1))

    y_pred_dta1 = dt1.predict(np.c_[x1_x.ravel(), x2_x.ravel()]).reshape(x1_x.shape)
    y_pred_dat2 = dt2.predict(np.c_[x1_x.ravel(), x2_x.ravel()]).reshape(x1_x.shape)

    fig, (ax1,ax2) = plt.subplots(1,2, figsize=[15,6])

    ax1.contourf(x1_x, x2_x, y_pred_dta1, alpha=0.2,cmap="Spectral")
    ax1.scatter(data['Glucose'][y_train==1], data['DiabetesPedigreeFunction'][y_train==1],marker=".",color="red",label="Positive")
    ax1.scatter(data['Glucose'][y_train==0], data['DiabetesPedigreeFunction'][y_train==0],marker=".",color="green",label="Negative")

    ax1.set_xlabel("Glucose")
    ax1.set_ylabel("DiabetesPedigreeFunction")
    ax1.set_title("Decision Tree with max_depth=2")
    ax1.legend()

    ax2.contourf(x1_x, x2_x, y_pred_dat2, alpha=0.2,cmap="Spectral")
    ax2.scatter(data['Glucose'][y_train==1], data['DiabetesPedigreeFunction'][y_train==1],marker=".",color="red",label="Positive")
    ax2.scatter(data['Glucose'][y_train==0], data['DiabetesPedigreeFunction'][y_train==0],marker=".",color="green",label="Negative")

    ax2.set_xlabel("Glucose")
    ax2.set_ylabel("DiabetesPedigreeFunction")
    ax2.set_title("Decision Tree with max_depth=10")
    ax2.legend()
    plt.show()