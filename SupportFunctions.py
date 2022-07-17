
def annotate_bar_perc(plot, n_rows, text_size=14, text_pos=(0,8), prec=2):
    """
    Function that annotates a stacked matplotlib barplot with percentage labels.
    
    """
    
    # Annotate the Bar-plots with category-percentage
    conts = plot.containers # Get containers, for each class of Transported one (True/False)
    
    for i in range(len(conts[0])): 
        height = sum([cont[i].get_height() for cont in conts]) # Calculate height of bar
        text = f"{height/n_rows*100:.{prec}f}%" # Create Annotation

        # Add text for every big bar
        plot.annotate(text,
                    (conts[0][i].get_x() + conts[0][i].get_width() / 2, height),  # Position xy
                    ha='center', # Centering
                    va='center',  
                    size=text_size, 
                    xytext=text_pos, # Coordinates of Text, Coord-System is defined by textcoords
                    textcoords='offset points') # Coord-System (So from the annotated position + xytext)
    return plot