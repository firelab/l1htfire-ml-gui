library(tensorflow)
library(keras)

#setwd(here())
# paramNames <- c("start_capital", "annual_mean_return", "annual_ret_std_dev",
#                 "annual_inflation", "annual_inf_std_dev", "monthly_withdrawals", "n_obs",
#                 "n_sim")

#use_virtualenv("G:\\tensorflow\\venv")
#use_virtualenv("~/.virtualenvs/spreadenv/")
#use_virtualenv("C:\\Users\\Isaac Grenfell\\Documents\\.virtualenvs\\r-reticulate")
#use_virtualenv("C:\\Users\\Isaac Grenfell\\Documents\\.virtualenvs\\r-tensorflow")
#use_virtualenv("C:\\Users\\Isaac Grenfell\\Documents\\.virtualenvs\\spreadenv")
#use_virtualenv("C:\\Users\\Isaac Grenfell\\Documents\\.virtualenvs\\spreadgui")

#setwd("I:\\workspace\\spread-model\\modelProtocolBuffers")

#setwd("G:\\tensorflow\\modelProtocolBuffers")
#setwd("G:\\tensorflow\\modelProtocolBuffers")

#needed to set virtual environment on server for shiny user
#use_virtualenv("/home/natalie/.virtualenvs/r-tensorflow")



#new_model <- load_model_tf('modelProtocolBuffers/no_gap', compile = FALSE)
new_model_combined <- load_model_tf('modelProtocolBuffers/combined', compile = FALSE)

nmc <- compile(new_model_combined)

# 
# paramNames <- c("bed_slope_angle", "bed_width", "fuel_depth",
#                 "fuel_loading", "ignition_depth", "particle_diameter", "particle_moisture",
#                 "wind_mean", "xvar", "yvar")




# ###For testing
# bed_slope_angle = 0
# bed_width = 50
# fuel_clump_size = 1.0
# fuel_depth = 0.5
# fuel_gap_size = 0.15
# fuel_loading = 1.0
# ignition_depth = 1.0
# particle_diameter = 0.0035
# particle_moisture = 2.0
# wind_amplitude_rel_mean = 100
# wind_mean = 3.0
# wind_period = 5.0
# wind_type = 0
# xvar = "bed_slope_angle"
# yvar  = "ros"
# levvar ="bed_width"

##End for testing

predict_spread <- function(bed_slope_angle = 0,
                           bed_width = 50,
                           fuel_clump_size = 1.0, 
                           fuel_depth = 0.5, 
                           fuel_gap_size = 0.15,
                           fuel_loading = 1.0,
                           ignition_depth = 1.0,
                           particle_diameter = 0.0035, 
                           particle_moisture = 2.0, 
                           wind_amplitude_rel_mean = 100, 
                           wind_mean = 3.0, 
                           wind_period = 5.0, 
                           wind_type = 0,
                           xvar = "bed_slope_angle", yvar  = "ros", levvar ="bed_width" )
{
  
  if(wind_type == 0)
  {
    wind_amplitude_rel_mean <- 0
    wind_period  <- 0
    
  }
  
  
  predvec.raw  <- c(bed_slope_angle,
                    bed_width,
                    fuel_clump_size,
                    fuel_depth,
                    fuel_gap_size,
                    fuel_loading,
                    ignition_depth,
                    particle_diameter,
                    particle_moisture,
                    wind_amplitude_rel_mean,
                    wind_mean,
                    wind_period,
                    wind_type
                    )
  
  xcolnum <- which(xvar == paramNames)
  ycolnum <- which(yvar == ycolnames)
  levcolum <- which(levvar == paramNames)
  #-------------------------------------
  # Inputs
  #-------------------------------------


  min_x_vec <- c(min_bed_slope_angle, 
                 min_bed_width, 
                 min_fuel_clump_size,
                 min_fuel_depth, 
                 min_fuel_gap_size,
                 min_fuel_loading, 
                 min_ignition_depth, 
                 min_particle_diameter, 
                 min_particle_moisture, 
                 min_wind_amplitude_rel_mean,
                 min_wind_mean,
                 min_wind_period
  )
  
  
  max_x_vec <- c(max_bed_slope_angle, 
                 max_bed_width, 
                 max_fuel_clump_size,
                 max_fuel_depth, 
                 max_fuel_gap_size,
                 max_fuel_loading, 
                 max_ignition_depth, 
                 max_particle_diameter, 
                 max_particle_moisture, 
                 max_wind_amplitude_rel_mean,
                 max_wind_mean,
                 max_wind_period
  )
  
  levseq <- seq(0, 1, length = nlevs)
  
  #-------------------------------------
  # Prediction
  #-------------------------------------
  # 
  # predvec <- c(degrees.scale,
  #              bed_width.scale,
  #              fuel_depth.scale,
  #              fuel_loading.scale,
  #              ignition_depth.scale,
  #              particle_diameter.scale,
  #              particle_moisture.scale,
  #              wind_mean.scale)
  
  
  #x <- t(as.matrix(cbind(predvec)))
  x <- t(as.matrix(cbind(predvec.raw)))
  
  xrep <- rbind(rep(x , npoints))
  xrep <- matrix(xrep, nrow = npoints, byrow = TRUE)
  #colnames(xrep) <- x_axis_vars
 # xparamnames <- names(x_axis_vars)
  xcolnum <- which(paramNames == xvar)

  tempxvals <- seq(min_x_vec[xcolnum], max_x_vec[xcolnum], length = npoints)
  xrep[,xcolnum] <- tempxvals
  nreps <- length(levseq)
  xrep.list <- vector("list", nreps)
  
  startseq <- rep(NA, nreps)
  endseq <- rep(NA, nreps)
  
  inc <- 0
  
  
  tempxvals.lev <- levseq
  denominator.lev <- max_x_vec[levcolum] - min_x_vec[levcolum]
  tempxvals.lev <- tempxvals.lev * denominator.lev
  tempxvals.lev <- tempxvals.lev  + min_x_vec[levcolum]
  temp.levs.x <- tempxvals.lev
  
  
  for(i in 1:nreps)
  {
    startseq[i] <- inc * npoints + 1
    endseq[i] <- (inc + 1) * npoints
    tempmat <- xrep
    tempmat[,levcolum] <- rep(temp.levs.x[i], dim(xrep)[1])
    xrep.list[[i]] <- tempmat
    inc <- inc + 1
    
  }
  xrep.all <- do.call("rbind", xrep.list)
  
  
  #xrep[,xcolnum] <- seq(0, 30, length = 10)
  
  #print(xrep[,xcolnum])
  colnames(xrep.all) <- paramNames.verbose
  pred.output <- nmc %>% predict(xrep.all)
  #pred.output <- nmc %>% predict(x)
  
  #pred.output
  
  ###Normalize output
  # numerator <- pred.output[,1] -  0.0
  # denominator <- 50.710646000000004 -  0.0
  # flamelength <-  pred.output[,1] * 50.710646000000004
  # numerator <- pred.output[,2] -  0.0
  # denominator <- 67.77206 -  0.0
  # fzd <- pred.output[,2] *67.77206
  # numerator <- pred.output[,3] -  0.0
  # denominator <- 783.45548 -  0.0
  # ros <-pred.output[,3] * 783.45548
  # 
  flamelength <- pred.output[,1]
  fzd <- pred.output[,2]
  ros <- pred.output[,3]
  outmat<- cbind(ros, flamelength, fzd )
  ycolnum <- which(ycolnames == yvar)
  ycollist <- vector("list", nreps)
  
  for(i in 1:nreps)
  {
    ycollist[[i]] <- outmat[,ycolnum][startseq[i]:endseq[i]]
    
  }
  
  ycolmat <- do.call("cbind", ycollist)
  
  colnames(outmat) <- ycolnames
  outlist <- vector("list", 7)
  outlist[[1]] <- xrep[,xcolnum]
  #  outlist[[2]] <- outmat[,ycolnum]
  outlist[[2]] <- ycolmat
  outlist[[3]] <- xcolnum
  outlist[[4]] <- ycolnum
  outlist[[5]] <- tempxvals
  outlist[[6]] <- temp.levs.x
  outlist[[7]] <- levcolum
  
  ###For debugging purposes
  # fout <- ("debug.txt")
  # outmat.debug <- cbind(xrep.all, ycolmat[,1])
  # write.table(outmat.debug, fout, quote = F, row.names = FALSE)
  
  return(outlist)
}

plot_nav <- function(nav) {
  
  
  layout(matrix(c(1,1)))
  
  palette(c("black", "grey50", "grey30", "grey70", "#d9230f"))
  
  colorBlindBlack8  <- c("#000000", "#E69F00", "#56B4E9", "#009E73", 
                         "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
  
  
  startseq <- seq(1,41, length = npoints)
  endseq <- seq(10, 50,length=npoints)
  
  tempx <- nav[[1]]
  xcoltemp <- nav[[3]]
  ycoltemp <- nav[[4]]
  orig.x  <-  nav[[5]]
  
  ymattemp <-  nav[[2]]
  legvals <- nav[[6]]
  levcolnum <- nav[[7]]
  #ymmattemp <-  outmat[,ycolnum]
  # ymattemp <- matrix(ymattemp, nrow = 10, byrow = TRUE)
  matplot(orig.x, ymattemp, type=  "n", xlab = paramNames.verbose[xcoltemp], ylab = ycolnames.verbose[ycoltemp])
  matlines(orig.x, ymattemp, col =colorBlindBlack8, lwd = 2)
  legend("topleft",  legend = legvals, lty = 1:nlevs, col = colorBlindBlack8[1:nlevs], lwd = 2, title = paramNames.verbose[levcolnum])
  
  grid()
  
}         
                  
function(input, output, session) {
  navA <- reactive({predict_spread(bed_slope_angle = input$bed_slope_angle,
                                   bed_width =  input$bed_width,
                                   fuel_clump_size = input$fuel_clump_size,
                                   fuel_depth=  input$fuel_depth, 
                                   fuel_gap_size = input$fuel_gap_size,
                                   fuel_loading=  input$fuel_loading, 
                                   ignition_depth=    input$ignition_depth, 
                                   particle_diameter=  input$particle_diameter, 
                                   particle_moisture=  input$particle_moisture, 
                                   wind_amplitude_rel_mean = input$wind_amplitude_rel_mean,
                                   wind_mean=input$wind_mean, 
                                   wind_period = input$wind_period,
                                   wind_type = input$wind_type,
                                   xvar=  input$xvar, 
                                   yvar= input$yvar,
                                   levvar = input$levvar
  )
    
    
  })
  output$a_distPlot <- renderPlot({
    
    plot_nav(navA())  }
  )
  # 
  # observeEvent(
  #   input$xvar,
  #   {
  #     updateSelectInput(
  #       inputId = "xvar",
  #       choices = setdiff(paramNames[1:8], isolate(input$levvar)),
  #       selected = isolate(input$xvar)
  #     )
  #   }
  # )
  # observeEvent(
  #   input$levvar,
  #   {
  #     updateSelectInput(
  #       inputId = "levvar",
  #       choices = setdiff(paramNames[1:8], isolate(input$xvar)),
  #       selected = isolate(input$levvar)
  #     )
  #   }
  # )



}

