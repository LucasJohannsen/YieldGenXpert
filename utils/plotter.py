import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

class RFPlotter:
    def __init__(self, gpu_result, poly_gdf, buffer_size=4.5, marker_size=45.9, outer_linewidth=10.5, inner_linewidth=2):
        self.RF = gpd.GeoDataFrame(gpu_result, geometry=gpd.points_from_xy(gpu_result.x, gpu_result.y))
        self.poly_gdf = poly_gdf
        self.buffer_size = buffer_size
        self.marker_size = marker_size
        self.outer_linewidth = outer_linewidth
        self.inner_linewidth = inner_linewidth
        self._prepare_data()

    def _prepare_data(self):
        # Ensure the CRS matches between RF and poly_gdf
        self.RF = self.RF.set_crs(self.poly_gdf.crs, allow_override=True)
        # Create an outward buffer around the polygon
        self.buffered_poly_gdf = self.poly_gdf.buffer(self.buffer_size)

    def plot(self):
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 10))
        palette = plt.get_cmap('Set1')
        fig.patch.set_facecolor('#F0F0F0')
        ax.set_facecolor('#F0F0F0')

        ax.ticklabel_format(style='plain', axis='y', useOffset=False)
        ax.grid(False)

        # Plot the RF data points
        sc2 = self.RF.plot(ax=ax, column='z', cmap='viridis', markersize=self.marker_size, marker='s')

        # Plot the buffered polygon border on top
        self.buffered_poly_gdf.boundary.plot(ax=ax, color='#453781FF', linewidth=self.outer_linewidth, alpha=1)

        # Plot the actual polygon boundary inside to make it look clean
        self.poly_gdf.boundary.plot(ax=ax, color='black', linewidth=self.inner_linewidth, alpha=1)

        # Colorbar settings
        norm = Normalize(vmin=self.RF['z'].min(), vmax=self.RF['z'].max())
        cbar_ax = fig.add_axes([0.16, 0.855, 0.7, 0.01])
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), cax=cbar_ax, orientation='horizontal')
        fig.text(0.08, 0.85, 'Yield t/ha', ha='center', va='center', fontsize=14)

        plt.show()

