import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from IPython.display import display
import json

def load_display_data():
    conn = sqlite3.connect('experiments.db')
    df = pd.read_sql_query("SELECT * FROM experiments", conn)

    # Horizontal bar plot of accuracies, sorted by accuracy descending
    df_sorted = df.sort_values('Accuracy', ascending=True)
    conn.close()
    
    plt.figure(figsize=(10, max(2, len(df_sorted) * 0.8)))  # Adjust height for better visibility
    df_sorted['Accuracy'].plot(kind='barh')
    plt.xlabel('Accuracy')
    plt.yticks(list(range(len(df_sorted))), df_sorted['id'])
    plt.ylabel('ID')
    plt.title('Accuracy of Experiments (Sorted by Accuracy)')
    plt.show()
    
    return df_sorted
    
def display_details(df, id):
    if df[df['id'] == id].empty:
        print(f"Experiment with id {id} not found.")
        return
    
    df_selected = df[df['id'] == id].T
    df_selected.columns = ['value']

    train_losses_str = df_selected.loc['train_losses'].values[0]
    valid_losses_str = df_selected.loc['valid_losses'].values[0]
    
    if train_losses_str is None or valid_losses_str is None:
        print("Loss data is missing for this experiment.")
        return
    
    train_losses = json.loads(train_losses_str)
    valid_losses = json.loads(valid_losses_str)

    # Plot the losses
    x = range(1, len(train_losses) + 1)
    plt.figure(figsize=(5, 5))
    plt.plot(x, train_losses, label='Train Loss')
    plt.plot(x, valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()

    display(df_selected.drop(labels=['train_losses', 'valid_losses'], axis=0))