import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Plots Palette and Theme
sns.set_theme()
sns.set_palette('husl')

def date_var_plot(df, var, save=False, filename=None):
    '''
    Plots a temporal line chart for a given variable in a DataFrame.
    Parameters:
        df (DataFrame): The data containing a 'date' column.
        var (str): The variable to plot on the Y-axis.
        save (bool): If True, saves the plot as an image.
        filename (str): Name of the file to save (required if save=True).
    '''
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='date', y=var)
    plt.title(f'{var} per Day')
    plt.xlabel('Date')
    plt.ylabel(var)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        if filename is None:
            filename = f"{var}_per_day.png"
        else:
            filename = f"{filename}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()


def var_per_day(var,name = 'Your Var', save=False, filename=None):
    '''
    Plots a time series for a given variable in a DataFrame.
        Parameters:
        df (DataFrame): The data containing a 'date' column.
        var (str): The variable to plot on the Y-axis.
        save (bool): If True, saves the plot as an image.
        filename (str): Name of the file to save (required if save=True).
    '''
    plt.figure(figsize=(14, 6))
    sns.lineplot(x=var.index, y=var.values)
    plt.title(f'Number of {name} per day')
    plt.xlabel('Date')
    plt.ylabel(name)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        if filename is None:
            filename = f"{var}_per_day.png"
        else:
            filename = f"{filename}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()

def vars_per_day_multivar(df, vars_list, names=None, save=False, filename=None):
    '''
    Plots multiple time series on the same line chart.
    
    Parameters:
        df (DataFrame): Must contain a datetime index or a 'date' column.
        vars_list (list): List of column names (strings) to plot.
        names (list): Optional list of display names for the variables.
        save (bool): If True, saves the plot as an image.
        filename (str): File name to save (only used if save=True).
    '''
    plt.figure(figsize=(14, 6))

    # Use custom names if provided, else fallback to column names
    if names is None:
        names = vars_list

    for var, name in zip(vars_list, names):
        sns.lineplot(data=df, x=df.index, y=df[var], label=name)

    plt.title('Variables per Day')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend(title='Variables')
    plt.tight_layout()

    if save:
        if filename is None:
            filename = "variables_per_day.png"
        else:
            filename = f"{filename}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()


def var_per_day_per_year(df, name_var, save=False, filename=None):
    '''
    Plots a time series per year for a given variable in a DataFrame.
        Parameters:
        df (DataFrame): The data containing a 'date' (Dtype = Datetime) column.
        var (str): The variable to plot on the Y-axis.
        save (bool): If True, saves the plot as an image.
        filename (str): Name of the file to save (required if save=True).
    '''

    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    years = df.date.dt.year.unique()

    for i, year in enumerate(years):
        data = df[df['date'].dt.year == year]
        sns.lineplot(data=data, x='date', y=name_var, ax=ax[i])
        ax[i].set_title(f'Number of users per day - {year}')
        ax[i].set_xlabel('Date')
        ax[i].set_ylabel(name_var)
        ax[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()

    if save:
        if filename is None:
            filename = f"{var}_per_day.png"
        else:
            filename = f"{filename}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    plt.show()
