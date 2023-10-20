

x_axis_vars <- c(  "Bed Slope Angle" = "bed_slope_angle" ,
                   "Bed Width"="bed_width" , 
                     "Fuel Clump Size"="fuel_clump_size",
                   "Fuel Depth"=   "fuel_depth",
                       "Fuel Gap Size" ="fuel_gap_size",
                      "Fuel Loading"= "fuel_loading",
                  "Ignition Depth"  =  "ignition_depth",
                     "Particle Diameter" =  "particle_diameter",
                    "Particle Moisture"  = "particle_moisture",
                   "Wind Amplitude"  = "wind_amplitude_rel_mean",
                   "Mean Wind Speed" = "wind_mean", 
                    "Wind Period" =  "wind_period",
                   "Wind Type"   ="wind_type"
                  
)

y_axis_vars <- c(
  "Spread Rate" = "ros",
  "Flame Zone Depth" = "fzd",
  "Flame Length" = "flength"
  
)


paramNames <- c(
  "bed_slope_angle",
  "bed_width",
  "fuel_clump_size",
  "fuel_depth",
  "fuel_gap_size",
  "fuel_loading",
  "ignition_depth",
  "particle_diameter",
  "particle_moisture",
  "wind_amplitude_rel_mean",
  "wind_mean",
  "wind_period",
  "wind_type", 
  "xvar", "yvar"
)

xparamnames <- paramNames[1:13]


yNames <- c("ros", "fzd", "flamelength")
