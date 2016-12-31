# Tools to assist plotting in R

library(ggplot2)

# ggplot theme from:
# https://shiring.github.io/machine_learning/2016/11/27/flu_outcome_ML_post?imm_mid=0eb8e0&cmp=em-data-na-na-newsltr_20161214


my_theme <- function(base_size = 12, base_family = "sans"){
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      axis.text = element_text(size = 12),
      axis.text.x = element_text(angle = 45, vjust = 0.5, hjust = 0.5),
      axis.title = element_text(size = 14),
      panel.grid.major = element_line(color = "grey"),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "aliceblue"),
      strip.background = element_rect(fill = "lightgrey", color = "grey", size = 1),
      strip.text = element_text(face = "bold", size = 12, color = "black"),
      legend.position = "bottom",
      legend.justification = "top", 
      legend.box = "horizontal",
      legend.box.background = element_rect(colour = "grey50"),
      legend.background = element_blank(),
      panel.border = element_rect(color = "grey", fill = NA, size = 0.5)
    )
}

# my ggplot2 plotting functions

plot_line <- function(data=vaccReported, xvar,yvar, alpha=1,add_color=TRUE,
                      color_factor="school_year",add_smooth=FALSE,
                      method="auto",smooth_color_factor="none")
{
  
  library(ggplot2)    
  
  p <- ggplot(data, aes_string(x=xvar,y=yvar))      
  
  if(add_color) {
    p <- p + geom_line(alpha = alpha, aes_string(color = color_factor)) 
  } else {
    p <- p + geom_line(alpha = alpha)
  }  
  
  if(smooth_color_factor !="none" & add_smooth) {
    p <- p + geom_smooth(method=method, aes_string(color=smooth_color_factor))
  }
  
  if(add_smooth){
    p <- p + geom_smooth(method=method)
  }
  
  p
}


plot_bivar_box <- function(data=vaccReported, xvar,yvar, alpha=1,add_color=FALSE,
                           color_factor="school_year",add_jitter=FALSE,
                           jitter_alpha=1,jitter_color,box_color)
{  
  library(ggplot2)  
  
  p <- ggplot(data, aes_string(x=xvar,y=yvar))      
  
  if(add_color) {
    p <- p + geom_boxplot(alpha = alpha, color=box_color,
                          aes_string(color = color_factor)) 
  } else {
    p <- p + geom_boxplot(alpha = alpha,color=box_color)
  }  
  
  if(add_jitter) {
    p <- p + geom_jitter(alpha=jitter_alpha,color=jitter_color)
  }
  
  p
}

plot_scatter <- function(data=vaccReported, xvar,yvar, alpha=1,add_color=TRUE,
                         color_factor="school_year",add_smooth=FALSE,
                         method="auto",smooth_color_factor="none")
{
  
  library(ggplot2)    
  
  p <- ggplot(data, aes_string(x=xvar,y=yvar))      
  
  if(add_color) {
    p <- p + geom_point(alpha = alpha, aes_string(color = color_factor)) 
  } else {
    p <- p + geom_point(alpha = alpha)
  }  
  
  if(smooth_color_factor !="none" & add_smooth) {
    p <- p + geom_smooth(method=method, aes_string(color=smooth_color_factor))
  }
  
  if(add_smooth){
    p <- p + geom_smooth(method=method)
  }
  
  
  p
}



iter_plot_line <- function(names,data=vaccReported, xvar, alpha=1,add_color=TRUE,
                           color_factor="school_year",add_smooth=FALSE,
                           method="auto",smooth_color_factor="none"){
  for(name in names){
    print(plot_line(data, xvar,yvar = name, alpha,add_color,
                    color_factor,add_smooth,method,smooth_color_factor))
  }
}

iter_plot_box <- function(names,data=vaccReported, xvar, alpha=1,
                          add_color=FALSE,color_factor="school_year",
                          add_jitter=FALSE,jitter_alpha=1,jitter_color,
                          box_color){
  for(name in names){
    print(plot_bivar_box(data, xvar,yvar=name, alpha,
                         add_color, color_factor,
                         add_jitter,jitter_alpha,jitter_color,
                         box_color))
  }
}

iter_plot_scatter <- function(names,data=vaccReported, xvar, alpha=1,add_color=TRUE,
                              color_factor="school_year",add_smooth=FALSE,
                              method="auto",smooth_color_factor="none"){
  for(name in names){
    print(plot_scatter(data, xvar,yvar = name, alpha,add_color,
                       color_factor,add_smooth,method,smooth_color_factor))
  }
}