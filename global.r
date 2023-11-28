

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

min_bed_slope_angle <- 0
max_bed_slope_angle <- 30
min_bed_width <-  1
max_bed_width <- 50
min_fuel_clump_size <- 0.5
max_fuel_clump_size <- 2
min_fuel_depth <- 0.05
max_fuel_depth <-  1 
min_fuel_gap_size <- 0.1
max_fuel_gap_size <- 0.5
min_fuel_loading <- 0.05
max_fuel_loading <- 3
min_ignition_depth <- 0.1
max_ignition_depth  <- 4
min_particle_diameter <- 0.001
max_particle_diameter <- 0.005
min_particle_moisture <- 2
max_particle_moisture<- 35
min_wind_amplitude_rel_mean  <- 0.2
max_wind_amplitude_rel_mean  <- 1
min_wind_mean <- 1
max_wind_mean <- 10
min_wind_period <- 1
max_wind_period  <- 5

ycolnames <- c( "ros", "fzd", "flength")
ycolnames.verbose <- c( "Rate of Spread", "Flame Zone Depth", "Flame Length")

test <- ""


paramNames.verbose <- c("Bed Slope Angle",
                        "Bed Width", 
                        "Fuel Clump Size",
                        "Fuel Depth",
                        "Fuel Gap Size",
                        "Fuel Loading",
                        "Ignition Depth",
                        "Particle Diameter",
                        "Particle Moisture",
                        "Wind Amplitude",
                        "Mean Wind Speed", 
                        "Wind Period",
                        "Wind Type")


nlevs <- 5
npoints <- 100

