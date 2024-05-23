import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import base64
from io import BytesIO
import subprocess

def display_stats_and_histogram(gdf, column_name : str , bins = 20):
    """
    Display descriptive statistics and histogram of a column in a GeoDataFrame.

    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame containing the data.
    column_name (str): The name of the column to display the statistics and histogram for.

    Returns:
    None

    """
    # Descriptive statistics with styling
    styled_stats = (gdf[column_name].describe()
                    .to_frame()
                    .style.format("{:.2f}")  # Adjust the precision to 2 decimal places
                    .set_caption(f"Descriptive Statistics of {column_name} after filtering"))

    # Save the histogram plot to a file
    plt.figure(figsize=(6, 4))  # Smaller size
    gdf[column_name].hist(bins=bins, color='skyblue', edgecolor='black')  # Nicer look with fewer bins
    plt.title(f'Histogram of {column_name} after filtering')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.grid(False)  # Turn off the grid for a cleaner look

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close()

    # Combine the styled table and histogram in HTML for side-by-side display
    html_content = f"""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div style="flex: 1;">{styled_stats.to_html()}</div>
        <div style="flex: 1; margin-left: 10px;">
            <img src="data:image/png;base64,{encoded_image}" alt="Histogram" style="max-width: 100%;">
        </div>
    </div>
    """

    # Display the combined content
    display(HTML(html_content))

def get_runtime_mode():
    try:
        # Check if `nvidia-smi` command runs successfully
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return 'gpu'
        else:
            return 'cpu'
    except FileNotFoundError:
        # `nvidia-smi` is not found
        return 'cpu'

