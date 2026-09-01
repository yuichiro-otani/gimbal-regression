from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt


def draw_map(
    column,
    title,
    gdf,
    *,
    file_path=None,
    dpi=200,
    show=True,
    cmap="viridis",
    markersize=10,
    alpha=1.0,
    legend=True,
    basemap=False,
    provider="CartoDB.Positron",
    carto_api_key=None,
    n_ticks=6,
    vmin=None,
    vmax=None,
):
    """
    Plot a GeoDataFrame colored by `column`.

    Parameters
    ----------
    column : str
        Column in `gdf` to plot.
    title : str
        Figure title.
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with geometry and CRS.
    file_path : str or None, default=None
        Output file path for saving the figure.
    dpi : int, default=200
        Output resolution when saving.
    show : bool, default=True
        Whether to display the figure.
    cmap : str, default="viridis"
        Matplotlib colormap name.
    markersize : float, default=10
        Marker size for point geometries.
    alpha : float, default=1.0
        Marker transparency.
    legend : bool, default=True
        Whether to display a legend/colorbar.
    basemap : bool, default=False
        Whether to add a contextily basemap.
    provider : str, default="CartoDB.Positron"
        Basemap provider path under `contextily.providers`.
    carto_api_key : str or None, default=None
        CARTO Basemap API key. Required when a CartoDB provider is used.
        The key is used only for the tile request and is not stored by GR.    
    n_ticks : int, default=6
        Number of longitude/latitude ticks.
    vmin, vmax : float or None, default=None
        Color scale limits for numeric plotting.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """
    if gdf is None or len(gdf) == 0:
        raise ValueError("gdf is empty.")
    if getattr(gdf, "crs", None) is None:
        raise ValueError("gdf.crs is None. Please set CRS, e.g. EPSG:4326.")
    if column not in gdf.columns:
        raise ValueError(f"Column '{column}' not found in gdf.")

    cx = None
    if basemap:
        try:
            import contextily as _cx
        except ImportError as exc:
            raise ImportError("contextily is required when basemap=True.") from exc
        cx = _cx

    g = gdf.to_crs(epsg=3857) if basemap else gdf.to_crs(epsg=4326)

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_title(title)

    plot_kwargs = dict(
        ax=ax,
        column=column,
        cmap=cmap,
        markersize=markersize,
        alpha=alpha,
        legend=legend,
    )
    if legend:
        plot_kwargs["legend_kwds"] = {"shrink": 0.67, "orientation": "vertical"}
    if vmin is not None:
        plot_kwargs["vmin"] = vmin
    if vmax is not None:
        plot_kwargs["vmax"] = vmax

    g.plot(**plot_kwargs)

    if basemap:
        src = cx.providers.CartoDB.Positron
    
        if isinstance(provider, str):
            try:
                node = cx.providers
                for part in provider.split("."):
                    node = getattr(node, part)
                src = node
            except Exception as exc:
                raise ValueError(
                    f"Unknown basemap provider: {provider}"
                ) from exc
    
        # CARTO basemaps require an API key.
        if isinstance(provider, str) and provider.startswith("CartoDB."):
            if not carto_api_key:
                raise ValueError(
                    "A CARTO Basemap API key is required when using a "
                    "CartoDB provider. Pass carto_api_key='YOUR_KEY'."
                )
    
            src = src.copy()
            separator = "&" if "?" in src["url"] else "?"
            src["url"] = f"{src['url']}{separator}key={carto_api_key}"
    
        cx.add_basemap(
            ax,
            source=src,
            crs=g.crs,
            reset_extent=False,
        )
        from pyproj import Transformer

        to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        xt = np.linspace(xmin, xmax, n_ticks)
        yt = np.linspace(ymin, ymax, n_ticks)

        y_mid = 0.5 * (ymin + ymax)
        x_mid = 0.5 * (xmin + xmax)

        xlabels = [f"{to_ll.transform(x, y_mid)[0]:.4f}" for x in xt]
        ylabels = [f"{to_ll.transform(x_mid, y)[1]:.4f}" for y in yt]

        ax.set_xticks(xt)
        ax.set_xticklabels(xlabels)
        ax.set_yticks(yt)
        ax.set_yticklabels(ylabels)

    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")

    if file_path:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        fig.savefig(file_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def results_to_gdf(df, lon_col="lon", lat_col="lat", crs="EPSG:4326"):
    """
    Convert a DataFrame with longitude and latitude columns to a GeoDataFrame.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("geopandas is required for results_to_gdf.") from exc

    if lon_col not in df.columns or lat_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{lon_col}' and '{lat_col}'.")

    return gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=crs,
    )
