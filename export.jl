import PlutoSliderServer

PlutoSliderServer.github_action(".";
  Export_cache_dir="pluto_state_cache",
  Export_baked_notebookfile=false,
  Export_baked_state=false,
  Export_create_pluto_featured_index=true,
  # more parameters can go here
)
