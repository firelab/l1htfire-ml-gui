library(shiny)
library(shinyjs)
ui<-fluidPage(
  titlePanel("L1HTFIRE-ML"),
  fluidRow(
    br(),
    column(width = 4, offset = 0,
           print("The L1HTFire model (Linear 1-Dimensional Heat Transfer) is a physics-based wildland fire spread model created by the USFS Missoula Fire Sciences Laboratory.
This is a Machine Learning (ML) surrogate of L1HTFire and displays 3 output variables: fire spread rate, flame length, and flame zone depth. 
Google Research created the ML version using 575 million training runs of the physical model. The ML surrogate model is hosted here for demonstration purposes only in order to illustrate
how wildland fire behavior results from the complex relationships between environmental and fire variables. A full description of this model can be found in Wildland Fire Behaviour (Finney, McAllister, Grumpstrup & Forthofer, 2021)")),
    column(width = 2, offset = 1,
           img(src='Logo_of_the_United_States_Forest_Service.svg.png', align = "center", height="20%", width="20%")),
    column(width = 2, offset = 0,
           img(src='Google_2015_logo.svg.png', align = "center", height="50%", width="50%")),
  ),
  br(),
  br(),
  br(),
  br(),
  column(5,
         fluidRow(
           column(width = 6, offset = 2,
                  useShinyjs(),
                  selectInput("xvar", "X-axis variable", selectvars, width =  "300px",  selected = "wind_mean"),
                  selectInput("yvar", "Y-axis variable", y_axis_vars, width = "300px",  selected = "ros"),
                  selectInput("levvar", "Level variable", selectvars, width ="300px",  selected = "particle_moisture"),
                  fluidRow(
                    column(width = 6, offset = 0,
                           
                           radioButtons("gap_type", "Fuel Continuity", c("Continuous", "Discontinuous")),
                           sliderInput( "fuel_clump_size", "Fuel Clump Size (meters) :", min = min_fuel_clump_size, max = max_fuel_clump_size, value =1.0, step = 0.1),
                           sliderInput( "fuel_gap_size", "Fuel Gap Size (meters) :", min = min_fuel_gap_size, max =max_fuel_gap_size, value =0.15, step = 0.1),
                           radioButtons("wind_type", "Wind Case", c("Constant Wind", "Sine Wind")),
                           sliderInput("wind_mean", "Mean Wind Speed (m/s)", min = min_wind_mean, max = max_wind_mean, value = 3.0, step = 0.1),
                           sliderInput("wind_amplitude_rel_mean", "Wind Amplitude rel. to mean (Fraction)", min = min_wind_amplitude_rel_mean, max = max_wind_amplitude_rel_mean, value = 1.0, step = 0.1),
                           sliderInput("wind_period", "Wind Period (s)", min = min_wind_period, max = max_wind_period, value = 3.0, step = 0.1),
                    ),
                  )
           ),
           column(width = 3, offset =  0,
                  sliderInput("bed_slope_angle", "Bed Slope Angle (in Degrees):", min = min_bed_slope_angle, max = max_bed_slope_angle,  value = 15),
                  sliderInput("bed_width", "Bed width (meters) :", min = min_bed_width, max = max_bed_width, value = 25, step = 1),
                  sliderInput( "fuel_depth", "Fuel Depth (meters)", min = min_fuel_depth, max = max_fuel_depth, value = 0.5, step = 0.1),
                  sliderInput("fuel_loading", "Fuel Loading (kg/m^3)", min = min_fuel_loading, max = max_fuel_loading, value = 1.0, step = 0.1),
                  sliderInput("ignition_depth", "Ignition Depth (meters)", min = min_ignition_depth, max = max_ignition_depth, value = 1.0, step = 0.1),
                  sliderInput("particle_diameter", "Particle Diameter (mm):", min = min_particle_diameter, max = max_particle_diameter, value = 3.5, step = 0.5),
                  sliderInput("particle_moisture", "Particle Moisture (%)", min = min_particle_moisture, max = max_particle_moisture, value = 2, step = 1),
                 
           ),
         ),
         uiOutput("bla1_ui"),  # here just for defining your ui
         uiOutput("bla2_ui")
  ),
  column(6,
         plotOutput("a_distPlot", height = "700px")
  )
)