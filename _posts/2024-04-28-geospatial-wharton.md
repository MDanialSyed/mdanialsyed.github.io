---
layout: post
title: "Geospatial Data Science for Evaluating Land Protection Policies"
subtitle: "Using high-resolution geospatial data to measure the real-world impact of environmental policies"
background: '/img/posts/geospatial-wharton/cover4.jpg'
---

My first project as a researcher in the Wharton School's Business Economics and Public Policy department was a geospatial economics study. It asked: to what extent have EU land protection policies contributed to the continent's greening and biodiversity?

I had worked with geospatial data before, applying convolutional neural networks to satellite images to predict poverty in Malawi, but the scale of this project created new challenges. The datasets totaled about 100 million data points, which required large-scale data processing and automated workflows. This post shares practical lessons for anyone starting out with large geospatial datasets.

## Project Overview

Land protection has become a central response to deforestation and urban sprawl. The Kunming-Montreal Global Biodiversity Framework, agreed at COP15 in 2022, commits signatories to protecting 30% of the world's land and water by 2030, and the EU has set a 30% target for its own territory. By then, Europe had already protected about 26% of its land.

To assess whether Europe's protected areas have improved vegetation and biodiversity, we needed granular datasets that would:

1. Record the protection status of each parcel of land.
2. Measure the greening of that land over time.
3. Describe other geophysical characteristics of the land.

## Data Collection

The raw data came from several sources. Our primary measure of greenness, the Normalized Difference Vegetation Index (NDVI), was sourced from Google Earth Engine. Agronomic data such as soil suitability came from the European Soil Data Centre (ESDAC), protected-site coverage from the European Environment Agency (EEA), and national boundary shapefiles from Eurostat. Most of these data are freely available online (ESDAC requires a request and approval), so the analysis can be reproduced.

#### Tip 1: Organize geospatial data from the start

Tabular data can usually be kept in a simple two-folder setup (raw and processed). Geospatial data are harder to track as they pass through several layers of a pipeline, especially when they come from several sources and each needs its own processing in ArcGIS or R.

For example, I started with Eurostat national boundaries for Scandinavia in shapefile format. The shapefile had to be split into separate countries in ArcGIS and stored locally. Further down the pipeline, each file was converted from shapefile to GeoJSON and then to CSV. Raw, intermediate and final files are easily misplaced, particularly across many countries, which becomes a problem when you need to return to an earlier step and change the processing.

Three practices help: set up an organized folder structure from the beginning, use a file naming convention such as `_raw`, `_clean` and `_final` for each processing stage, and document what was done at each step. This was the folder structure for the project:

![Diagram of the project folder structure: final datasets, scripts, country datasets and shapefiles, and raster and vector data](/img/posts/geospatial-wharton/workspace_structure.png)

## Data Challenges

The unit of observation was a 300 m × 300 m grid cell that stays constant over time. The grid divides each country into evenly spaced cells, each identified by the coordinates of its centroid. Grids were generated for the 32 countries in the Common Database on Designated Areas (CDDA).

Building the datasets on these grids raised several challenges. NDVI was constructed from Landsat images from 1985 to 2019, but dense cloud cover in some months obscured the images, so we used a rolling-window algorithm to select the images with the clearest view of each country's vegetation.

Mapping vector data, mostly categorical data such as climate zones and biogeographical regions, in ArcGIS Pro was slow when done manually for larger countries with several million grid cells. Instead of repeating this by hand for 32 countries, I built an ArcGIS toolbox to automate it at the country level.

#### Tip 2: Automate repetitive workflows

A rule from my ***Effective Programming for Economists*** course at the University of Bonn: if code has to run more than once, write a function and use `lapply` or a `for` loop. This made a large difference in building these datasets. Two examples:

**1. ArcGIS Pro toolbox and ArcPy**

Many ArcGIS tutorials focus on manual steps through the software's menus. Manual processing has two drawbacks. First, it invites human error: selecting the wrong attribute for a spatial join, setting the wrong coordinate reference system or missing a parameter. Second, repeating the same processes across many countries is time-consuming and multiplies the risk of error.

The solution is to automate sequences of spatial processes with an ArcGIS toolbox or with ArcPy scripts. ArcGIS ModelBuilder represents geoprocessing workflows as directed acyclic graphs (DAGs), which show the processing steps applied to each country's shapefile and make it easy to check their order: for example, repairing the protected-sites shapefile to fix topology errors and then spatially joining it to the country shapefile. The DAGs also serve as documentation.

![ArcGIS ModelBuilder diagram of the geoprocessing steps applied to each country's shapefile](/img/posts/geospatial-wharton/DAG.png)

**2. Functions for raster processing in R**

Several of the inputs were multi-band rasters that record a geophysical process over time. For example, rainfall history for every grid cell from 1985 to 2019 required extracting the corresponding band from the rainfall raster for each year. This snippet is from an early version of the raster extraction script:

```r
rainfall1985 <- raster("C:/Users/me/Desktop/ForestTransition/rain/rain_1985.tif")
rainfall1986 <- raster("C:/Users/me/Desktop/ForestTransition/rain/rain_1985.tif")
rainfall1987 <- raster("C:/Users/me/Desktop/ForestTransition/rain/rain_1987.tif")
```

The second line maps the 1985 raster onto the variable for 1986. The error was buried among some 1,600 lines of code and took many hours to trace. A loop or function that removes repeated lines avoids this kind of mistake:

```r
for (year in rainfall_years) {
  raster_path <- paste0("../data_Raster/GIS/eu_rain_ERA_5/rain_", year, ".tif")
  raster_name <- paste0("rainfall", year)
  assign(raster_name, raster(raster_path))
  country_grids[,paste0("rain",year)] <- exact_extract(eval(as.symbol(paste0("rainfall",year))),country_grids,'mean')
}
```

The loop replaced 25 lines of code and removed the hard-coding that caused the error. The same approach applies to any raster that needs repetitive processing, here extracting a raster to a country's grid.

## Geospatial Data Construction

The data pipeline involved several scripts, which are documented step by step in the project's replication package. The most demanding was the raster extraction script, which maps raster values onto a shapefile (in effect, converting a TIFF file into CSV columns).

Extracting raster data, which stores mostly continuous measures such as NDVI and rainfall on a grid, was computationally intensive even for smaller countries such as Switzerland. R ran out of memory on a local machine ("vector memory exhausted"), so I moved the process to Wharton's high-performance computing (HPC) cluster.

#### Tip 3: Use high-performance computing

Wharton's HPC cluster cut the raster extraction for a single country from several days to a few hours. Some of the same ideas also help without access to HPC:

- **Array jobs:** multi-core processing in R with the `parallel` package and `mclapply` handled most of the Monte Carlo simulations in my master's thesis. Geospatial processing, however, needed more memory as well as more processing power. I split the raster extraction into 12 independent parts and ran them simultaneously as an array job, which made the workflow about four times faster. Merging the outputs back into one dataset took only a few minutes.
- **UNIX command-line tools:** these improve efficiency with large datasets, with or without HPC.
  - **File manipulation:** `grep`, `awk` and `sed` search, filter and transform data. For example, `sed` can apply the same correction across several R scripts at once.
  - **Batch processing:** shell scripts chain commands into data processing pipelines. In this project, I ran batches of R scripts for batches of countries and then merged the components into a dataset for each country.
  - **Remote access:** SSH connects securely to a remote server or cloud platform, so computationally intensive tasks can run on machines with more resources than a local computer.

## Results

The project produced 32 datasets, one for each country in the sample, with 172 variables on forest cover, agronomic characteristics and spatial information: about 100 million data points in total. Using a staggered difference-in-differences design, we found that protected-area policies had no meaningful effect on either vegetation cover or economic activity, measured by nightlights, across countries and time periods. The results suggest that the land selected for protection was not under immediate threat, which limited the policies' impact.

The findings point to a gap between policy goals and outcomes on the ground: protecting more land matters less than protecting the areas that need intervention to prevent degradation. More generally, the project shows how large volumes of high-resolution geospatial data, together with careful data management and high-performance computing, can be used to evaluate the real-world impact of environmental policies.
