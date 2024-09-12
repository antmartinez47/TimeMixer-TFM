import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import numpy as np
import plotly.graph_objects as go
import random

def parallel_coord_plot(csv_file, suptitle=None, path=None, configs=100):
    df = pd.read_csv(csv_file)
    if "smac" in csv_file:
        return
        # Group by 'id' and select the row with the highest 'budget' for each group
        df = df.loc[df.groupby('id')['budget'].idxmax()]
        # Reset the index
        df.reset_index(drop=True, inplace=True)
        df = df[[i for i in df.columns if i.startswith("config")] + ["cost"]]
        df.columns = df.columns.str.replace('config_', '', regex=False)
        df.columns = df.columns.str.replace('cost', 'valid_loss', regex=False)
    else:
        df = df[[i for i in df.columns if i.startswith("config")] + ["best_valid_loss"]]
        df.columns = df.columns.str.replace(r'config[/_]', '', regex=True)
        df.columns = df.columns.str.replace('best_', '', regex=False)
    hps = sorted([i for i in df.columns if not i.startswith("valid")])
    if "d_ff" in hps:
        hps.remove("d_ff")
    if "moving_avg" in hps:
        hps.remove("moving_avg")
    random_configs = np.random.choice(list(range(df.shape[0])), size=configs)
    dimensions = []
    for col in hps:
        if df[col].dtype == "object":
            unique = df[col].unique()
            mapping = dict(zip(unique, range(len(unique))))
            df[col] = df[col].apply(lambda x: mapping[x])
            dimensions.append(dict(tickvals=list(mapping.values()), label=col, values=df[col].iloc[random_configs], ticktext=list(mapping.keys())))
        else:
            dimensions.append(dict(label=col, values=df[col].iloc[random_configs]))
            
    # Create the parallel coordinate plot
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(color = df['valid_loss'].iloc[random_configs], colorscale = 'Viridis', showscale = True, colorbar=dict(title='Cost')),
            dimensions=dimensions
        )
    )
    # Update the layout
    fig.update_layout(
        title=suptitle,
        plot_bgcolor = 'white',
        paper_bgcolor = 'white'
    )
    if path is not None:
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.write_image(path, width=1900, height=800)

def plot_hyperparameters(data_dict, suptitle=None, path=None):
    num_plots = None
    fig, axes = None, None

    for index, (label, csv_file) in enumerate(data_dict.items()):
        df = pd.read_csv(csv_file)
        if "smac" in csv_file and len(data_dict) == 1:
            return
        elif "smac" in csv_file:
            continue
            df = df.loc[df.groupby('id')['budget'].idxmax()]
            df.reset_index(drop=True, inplace=True)
            df = df[[i for i in df.columns if i.startswith("config")] + ["cost"]]
            df.columns = df.columns.str.replace('config_', '', regex=False)
        else:
            df = df[[i for i in df.columns if i.startswith("config")] + ["best_valid_loss"]]
            df.columns = df.columns.str.replace('config/', '', regex=False)
            df.columns = df.columns.str.replace('best_valid_loss', 'cost', regex=False)

        hyperparameters = sorted([col for col in df.columns if col != 'cost'])

        if "d_ff" in hyperparameters:
            hyperparameters.remove("d_ff")
        if "moving_avg" in hyperparameters:
            hyperparameters.remove("moving_avg")

        if num_plots is None:
            num_plots = len(hyperparameters)
            num_cols = 3
            num_rows = math.ceil(num_plots / num_cols)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
            axes = axes.flatten()

        if len(data_dict) <= 1:
            label = None

        for i, column in enumerate(hyperparameters):
            num_unique_values = df[column].nunique()
            if num_unique_values < 10 or df[column].dtype == 'object':
                # Different color for each box
                unique_values = df[column].unique()
                palette = sns.color_palette("husl", len(unique_values))
                sns.boxplot(x=column, y='cost', data=df, ax=axes[i], palette=palette, fliersize=0)
            else:
                # Random color for scatter plot
                random_color = sns.color_palette("husl", 100)[random.randint(0, 99)]
                sns.scatterplot(x=column, y='cost', data=df, ax=axes[i], color=random_color, label=label)

            axes[i].set_title(f'{column}', fontsize=14)  # Add title to each subplot
            axes[i].grid(True)  # Add gridlines

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

    if len(data_dict) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', title='Algorithm', bbox_to_anchor=(1.15, 1))
        for i in range(len(axes)):
            if axes[i].get_legend() is not None:
                axes[i].get_legend().remove()

    fig.suptitle(suptitle, fontsize=25)
    plt.tight_layout(pad=1.5)
    if path is not None:
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(path, bbox_inches='tight')  # Ensure the legend and title are not cut off
    


def plot_cost_trajectories(data_files, default_value=None, smooth=True, window_size=25, path=None):
    """
    Plots the evolution of cost over wall clock time for multiple datasets using line plots.
    Includes a secondary x-axis showing the cumulative number of configurations evaluated so far.
    Automatically selects the plotting method based on the presence of 'smac' in the file path.

    Parameters:
    data_files (dict): A dictionary where the key is the legend name and 
                       the value is the path to the corresponding CSV file.
    smooth (bool): Whether to smooth the cost values. Default is True.
    window_size (int): The size of the rolling window for smoothing. Default is 20.

    Returns:
    None: Displays the plot.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.set(style="whitegrid")

    # Track the cumulative configuration counts
    config_counts = []

    # Loop through each file in the provided dictionary
    for legend_name, file_path in data_files.items():
        if 'smac' in file_path:
            # Execute SMAC specific processing
            data = pd.read_csv(file_path)
            # Group by 'id' and select the row with the highest 'budget' for each group
            data = data.loc[data.groupby('id')['budget'].idxmax()]
            # Reset the index
            data.reset_index(drop=True, inplace=True)

            # Convert start_time and end_time to datetime format if they are not already
            data['start_time'] = pd.to_datetime(data['start_time'], unit='s')
            data['end_time'] = pd.to_datetime(data['end_time'], unit='s')

            # Calculate the wall clock time (in seconds) relative to the first start time
            wall_clock_time = (data['end_time'] - data['start_time'].min()).dt.total_seconds()

            # Calculate the cumulative count of configurations evaluated
            cumulative_configs = range(1, len(data) + 1)

            # Store the wall clock time and cumulative configurations for mapping
            config_counts.append((wall_clock_time, cumulative_configs, legend_name))

            if smooth:
                # Apply a rolling minimum to smooth the cost values
                data['cost'] = data['cost'].rolling(window=window_size, min_periods=1, center=False).min()

            # Plot the cost evolution
            sns.lineplot(data=data, x=wall_clock_time, y='cost', label=legend_name, ax=ax1)

        else:
            # Execute Tuning specific processing
            data = pd.read_csv(file_path)
            # Convert 'date' column to datetime format
            data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d_%H-%M-%S')

            # Calculate the wall clock time (in seconds) relative to the first timestamp
            start_time = data['date'].min()
            wall_clock_time = (data['date'] - start_time).dt.total_seconds()

            # Calculate the cumulative count of configurations evaluated
            cumulative_configs = range(1, len(data) + 1)

            # Store the wall clock time and cumulative configurations for mapping
            config_counts.append((wall_clock_time, cumulative_configs, legend_name))

            if smooth:
                # Apply a rolling minimum to smooth the cost values
                data['best_valid_loss'] = data['best_valid_loss'].rolling(window=window_size, min_periods=1, center=False).min()

            # Plot the cost evolution
            sns.lineplot(data=data, x=wall_clock_time, y='best_valid_loss', label=legend_name, ax=ax1)
            
    if default_value is not None:
        ax1.axhline(default_value, c="k", label="Default cost")

    # Create a secondary x-a xis on the top to show cumulative configurations
    ax2 = ax1.twiny()

    # Combine data for proper tick alignment
    combined_times = []
    combined_configs = []

    for times, configs, _ in config_counts:
        combined_times.extend(times)
        combined_configs.extend(configs)

    # Set limits and ticks
    ax1.set_xlim(min(combined_times), max(combined_times))
    ax2.set_xlim(ax1.get_xlim())  # Sync with primary x-axis

    # Create tick positions and labels for the secondary x-axis
    xticks = ax1.get_xticks()
    xticks = [tick for tick in xticks if tick <= max(combined_times)]  # Remove out-of-range ticks
    xticklabels = [int(round(next((c for t, c in zip(combined_times, combined_configs) if t >= tick), 0)))
                   for tick in xticks]

    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels)
    ax2.set_xlabel('Number of Configurations')

    # Set labels and titles for the primary axis
    ax1.set_xlabel('Wall Clock Time (s)')
    ax1.set_ylabel('Cost')
    ax1.legend(title='')
    ax1.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    if path is not None:
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(path)

    # Show the plot
    
