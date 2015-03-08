from pandas import *
from ggplot import *

def plot_weather_data(turnstile_weather):
    ''' 
    plot_weather_data is passed a dataframe called turnstile_weather. 
    Use turnstile_weather along with ggplot to make another data visualization
    focused on the MTA and weather data we used in Project 3.
    
    Make a type of visualization different than what you did in the previous exercise.
    Try to use the data in a different way (e.g., if you made a lineplot concerning 
    ridership and time of day in exercise #1, maybe look at weather and try to make a 
    histogram in this exercise). Or try to use multiple encodings in your graph if 
    you didn't in the previous exercise.
    
    You should feel free to implement something that we discussed in class 
    (e.g., scatterplots, line plots, or histograms) or attempt to implement
    something more advanced if you'd like.

    Here are some suggestions for things to investigate and illustrate:
     * Ridership by time-of-day or day-of-week
     * How ridership varies by subway station
     * Which stations have more exits or entries at different times of day
       (You can use UNIT as a proxy for subway station.)

    If you'd like to learn more about ggplot and its capabilities, take
    a look at the documentation at:
    https://pypi.python.org/pypi/ggplot/
     
    You can check out the link 
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
    to see all the columns and data points included in the turnstile_weather 
    dataframe.
     
   However, due to the limitation of our Amazon EC2 server, we are giving you a random
    subset, about 1/3 of the actual data in the turnstile_weather dataframe.
    '''
    # PLOT2 - Ridership by day of the week
  
    data = turnstile_weather.copy()
    data['Weekday'] = pandas.to_datetime(turnstile_weather.DATEn).apply(lambda d: d.strftime('%w %A'))
    entries_grouped = data[['Weekday','ENTRIESn_hourly']].groupby(['Weekday'], as_index = False).sum()
    
    # Create a new dataframe with columns: weekdays and entries
    weekdays = entries_grouped['Weekday']
    entries = entries_grouped['ENTRIESn_hourly']
    entries_weekday = pandas.concat([weekdays, entries], axis = 1)

    plot = ggplot(entries_weekday, aes('Weekday' , 'ENTRIESn_hourly')) \
           + geom_bar(aes(weight = 'ENTRIESn_hourly'), fill = 'blue', stat = "bar") \
           + xlab('Day of Week') + ylab('Entries') + ggtitle('Ridership by Day of Week') \
           + scale_x_continuous(labels = ('Sunday', 'Monday', 'Tuesday', \
                                          'Wednesday', 'Thursday', 'Friday', 'Saturday'))
    
    return plot
