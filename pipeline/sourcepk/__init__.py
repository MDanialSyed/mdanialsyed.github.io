"""Source: Pakistan data pipeline.

Downloads economic data from official publishers, keeps every source file
exactly as received, turns the files into series with full lineage, checks
them, and writes only the publishable values to the website's data folder.

Numbers are produced by code only. Nothing in this package reads news or
accepts a hand-typed value.
"""

__version__ = "0.1.0"
