import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Plots Palette and theme
sns.set_theme()
sns.set_palette('husl')

def yhat_vs_y(y_hat,y, save=False, filename='yhat-vs-y'): 
    '''
    Error visualization y_hat vs y_true.
    '''
    plt.title("Yhat vs Y")
    plt.plot(y_hat, y,"o")
    plt.plot(y,y)
    plt.xlabel("y")
    plt.ylabel("yhat")
    
    if save == True:
        plt.savefig(f'{filename}.png')
    
    plt.show()


def error_vs_y(y_hat,y, save=False, filename='error-vs-y'):
    '''
    Error = y_true - y_hat
    '''
    error = y - y_hat
    plt.title("Error vs Y")
    plt.plot(y, error,"o")
    plt.axhline(y=0)
    plt.xlabel("y")
    plt.ylabel("error")
    
    if save == True:
        plt.savefig(f'{filename}.png')
        
    plt.show()

def error_distribution(y_hat,y, save=False, filename='error-distribution'):
    '''
    Error Histrogram
    Error = y_true - y_hat
    '''
    error = y - y_hat
    plt.hist(error, bins=100)
    plt.title("Error Distribution")
    plt.xlabel("error")
    
    if save == True:
        plt.savefig(f'{filename}.png')
    
    plt.show()

def confusion_matrix_plot(y_hat, y, save=False, filename='confusion-matrix'):
    '''
    Confusion Matrix Plot
    '''
    cm = confusion_matrix(y, y_hat)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save:
        plt.savefig(f'{filename}.png')

    plt.show()