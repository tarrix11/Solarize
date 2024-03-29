#!/usr/bin/env python
# coding: utf-8

# In[121]:


# NUMBER 1

''' 
THIS IS THE MAIN CODE 
gets effective irradiance - still gotta find optimal angles 
'''

# Plot each year Individually to see what the data looks like 
# Plot their average best fit lines too 
# finds all the missing rows too and replace into new array 

# Notes 
# 2001 unsuable 
# 2002
# 2003
# 2004
# 2007
# 2010
# 2019 - 7/9/2019 - 7/20/2019 there were no rows for these data  

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from datetime import *



# DATFRAME with Irradiance of all years with Index being the hour (contains NaNs to be replaced with averages)
all_years = pd.DataFrame() 

plots = 4
year_begin = 2015
plt.figure(figsize=(15, 10))

# i can probably make this clearer by using dict or sth
year_dataframes = [] # this to combine the dataframes that the loop iterates through so we can call each element of the dataframe array to get each years data

# INITIAL VISUALIZATION
# Iterate through the number of plots (years) that we want
for i in range(0,plots):    
    fileName = "C:/Users/tartn/Documents/Solarize_Data/Kew_Gardens/midas-open_uk-radiation-obs_dv-202107_greater-london_00723_kew-gardens_qcv-1_2020.csv"
    current_year = year_begin + i
    fileName = fileName.replace('2020',str(current_year))
    print(fileName)
    df = pd.read_csv(fileName, skiprows=list(range(0, 75)))
    # when reading this csv files, switch the '\' to '/' forward slashes
    # since file data frame is uneven, we need to skip the first 0-75 rows to access the real data 
    year = 'data' + str(current_year)
    
    print(year)
    
    year = pd.DataFrame(columns=['Time','Irradiance','Effective Irradiance']) 
    year['Time'] = df['ob_end_time']
    year['Irradiance'] = df['glbl_irad_amt']
    year.drop(df.tail(1).index,inplace=True) # remove the last 'end' value 

    

    # iterate through the entire'Time' column 
    # subtract the current row with the starting date 
    # get the number of seconds elapsed and divide by 3600 to get hour 
    
    for index, row in year.iterrows():
        year.loc[index,'Hours'] = (datetime.strptime(row['Time'],'%Y-%m-%d %H:%M:%S') - datetime.strptime(year.iloc[0]['Time'],'%Y-%m-%d %H:%M:%S')).total_seconds() / 3600
        
        
    
    
    # since irradiance is in KJ/m2 per hour, we need to divide it down to W/m2
    year['Irradiance'] = (year['Irradiance']*1000)/3600
 
    
    sns.scatterplot(x = year['Hours'],y = year['Irradiance'],data = year,s=2)

    

    
    # DATA PREPROCESSING 
    # Check if the next hour is correct and not skipped 
    # Print all the rows that were skipped 
    missing_rows = []
 
    last_hour = year['Hours'].iloc[-1] # 8759 in 2005
    
    
    
    for i in range(len(year['Hours'])):
        if i > 0:
            if  (year.iloc[i,3] - year.iloc[i-1,3]) > 1:  #get each row of the 'Hours' Column 
                print('Hour:', end=" ")
                print(i-1),
                print('Skips: ', end=" "),
                print(year.iloc[i,3] - year.iloc[i-1,3] , end=" ")
                print('hours after this ')

                start = int(year.iloc[i-1,3])
                end = int(year.iloc[i,3])

                missing_rows.append(list(range(start,end))) # THESE ARE ALL THE SKIPPED ROWS 

                # for the missing range, get a create new df with the missing hours
          
        
        
    # Creates a new Dataframe to work on with just Irradiance and Hour
    new_df = pd.concat([year['Irradiance'], year['Hours']], axis=1)
    
    hours_in_year = 8760
    
    # Fills all the missing hours and puts in a NaN for the corresponding Irradiance 
    for h in range(0,hours_in_year):
        if (h > 0) and (h < hours_in_year):
            if (new_df.iloc[h,1] - new_df.iloc[h-1,1]) > 1:
                new_row_df = pd.DataFrame({'Irradiance': [np.nan], 'Hours': [h]}, index=[h])
                new_df = pd.concat([new_df.loc[:h-1], new_row_df, new_df.loc[h:]]).reset_index(drop=True)
                
    # make the new dataFrame with all the irradiances and 8760 hours, then replace them with the averages 
    # if is not NaN sum up and average out for each hour for the 10 years 
    
    all_years[str(current_year)] = new_df['Irradiance']
    

               
        
# After looping for every year to get a new data frame with Irradiances and Nans
# iterate through each year column, then if NaN, replace if Avg of that entire row thats not NaN 

# find the average by counting all non NaNs and dividing by that count for each row 
averages = []
for index1, row1 in all_years.iterrows():
    total = 0
    nums = 0
    for column1, value1 in row1.iteritems():
    
        if pd.isna(value1):   # if no value avaib
            pass
        else: 
            nums += 1
            total += value1
            
    row_average = total/nums
    averages.append(np.round(row_average,2))

all_years['Averages'] = pd.DataFrame(averages)
        
            
        
        

# go through each row of each col, if NaN, replace with the Average value from the same index position 
for col2 in all_years.columns:
    
    for row2, value2 in all_years[col2].iteritems():
        
        if pd.isna(value2):
            
            all_years.loc[row2, col2] = all_years['Averages'][row2]
 


            



##### Fitting an Average best fit line, instead of top/bottom boundary ##################
    # Fit a polynomial curve to the data
    # these numbers may seem low but it takes into account the night time as well 
    # I think i can use this code to fill in the data for now 
    total_rows = len(year) 
    hours = year.iloc[0:total_rows,3].values
    irradiance = year.iloc[0:total_rows,1].values

    coefficients = np.polyfit(hours , irradiance,12)
    curve_fit = np.poly1d(coefficients)

    # Calculate corresponding y values for the curve
    y_curve = curve_fit(hours)

    # Draw the curved bounding line
    
    sns.scatterplot(x = hours,y = y_curve,data = year, s = 2)
    
    print(current_year, missing_rows)
    
    
plt.figure(figsize=(15, 10))   
 
hours1 = range(len(all_years['Averages'].values))
sns.scatterplot(x = year['Hours'],y = all_years['Averages'],data = year,s=2)
coefficients1 = np.polyfit( hours1, all_years['Averages'].values,6)
curve_fit1 = np.poly1d(coefficients1)

# Calculate corresponding y values for the curve
y_curve1 = curve_fit1(hours1) # fit hours1 into a formula that produces the y values 

# Draw the curved bounding line

sns.scatterplot(x = hours1,y = y_curve1 ,data = all_years, s = 2)
plt.xlabel('Average Solar Irradiance W/m2')
plt.ylabel('Time (days)')







############# SUN ANGLE CALCULATIONS ##########################

from pysolar.solar import *
from datetime import *
from math import *
import matplotlib.pyplot as plt
import calendar


#Coventry Co-ordinates
latitude = 52.424
longitude = -1.536

hours_total = 24
days_total = 29
months_total = 12
year = 2019
month = 6
day = 15
cosRatios = []

for months in range(months_total):
    _, last_day = calendar.monthrange(year, months+1)
    for days in range(1, last_day + 1):
        for hour in range(hours_total):
            date = datetime(year, months+1, days, hour, 13, 1, 130320, tzinfo= timezone.utc)

            #print(hour)

            Sun_altitude = get_altitude(latitude, longitude, date)
            zenith = (90-Sun_altitude) 
            zenith_rad = radians(zenith)
            #print("Zenith deg : %f" % (zenith))

            azimuth = get_azimuth(latitude, longitude, date)
            azimuth_rad = radians(180-azimuth)
            #print("Azimuth deg: %f" % (180 - azimuth)) # since in pysolar the the 0th degree refers to the noth pole which in the formula is in reference to the south pole 
            # therefore we subtract subtract 180 by that angle to get its reference from the south pole 

            altitude_deg = get_altitude(latitude, longitude, date)
            radiation.get_radiation_direct(date, altitude_deg)
            # print("Hours: %d, Radiation : %f" % (hour,radiation.get_radiation_direct(date, altitude_deg)))

            tilt_rad = radians(45)
            surfaceAzimuth_rad = radians(-10)

            # Ratio cos(theta) from the Lamberts formula 
            # Negative value means sun is behind earth can be equated to zero 
            cosTheta = cos(zenith_rad)*cos(tilt_rad)+(sin(zenith_rad)*sin(tilt_rad)*cos(azimuth_rad - surfaceAzimuth_rad))
            #print(cosTheta)
            #print(' ')
            cosRatios.append(cosTheta)

x = range(len(cosRatios))

#plt.figure(figsize=(16, 12))  
#plt.plot(x,cosRatios)
len(cosRatios) # now we can assume these numbers do not change as every year the earth revolves back to the same place 

# replace all negative cosine ratios with zero cuz thats when the sun is behind the earth 

cosRatioDf = pd.DataFrame(cosRatios)
cosRatio = cosRatioDf.applymap(lambda x: max(x, 0))

all_years['cosRatio'] = cosRatio


# FINAL EFFECTIVE IRRADIANCE
# multiply cos()ratios by the recieved average irradiance 
## Numbers may seem low but thats cuz its averaged out for 24 hours, not the 6-10 hours of sunlight we get

all_years['EffectiveAverage'] = all_years['Averages']*all_years['cosRatio']
sns.scatterplot(x=np.arange(0,len(all_years['EffectiveAverage'])),y = all_years['EffectiveAverage'].values, s = 2, marker='s')


coefficients3 = np.polyfit( hours1, all_years['EffectiveAverage'].values,6)
curve_fit3 = np.poly1d(coefficients3)

# Calculate corresponding y values for the curve
y_curve3 = curve_fit3(hours1) # fit hours1 into a formula that produces the y values 

# Draw the curved bounding line

sns.scatterplot(x = hours1,y = y_curve3 ,data = all_years, s = 2)
plt.title('Before and After Considering Effective Angle')



############ Day Average over all years #########################
# Create a new DataFrame for the averaged values
plt.figure(figsize=(15, 10))   
averaged_df = pd.DataFrame()
hoursInDay = 24
daysInYear = 365
combineDays = 24
# Calculate the average for each chunk and append to the new DataFrame
# for each 24 hour blocks of data, average them to get energy produce in WattHours of each day
for i in range(0, len(all_years), hoursInDay):
    block = all_years.iloc[i : i + combineDays]
    averaged_value = block['EffectiveAverage'].mean()
    averaged_df = averaged_df.append({'day_average': averaged_value}, ignore_index=True)
    
len(averaged_df)
sns.scatterplot(x=np.arange(1,daysInYear+1),y=averaged_df['day_average'].values,s=4)
plt.title('All Years Days Average')
plt.xlabel('Days')
plt.ylabel('Average Solar Irradiance (Watt-Hour/m2)')

coefficients4 = np.polyfit(np.arange(1,daysInYear+1),averaged_df['day_average'].values,6)
curve_fit4 = np.poly1d(coefficients4)

# Calculate corresponding y values for the curve
y_curve4 = curve_fit4(np.arange(1,daysInYear+1)) # fit hours1 into a formula that produces the y values 

# Draw the curved bounding line

sns.scatterplot(x = np.arange(1,daysInYear+1),y = y_curve4 ,data = averaged_df, s = 4)


# In[122]:


# NUMBER 2

# Split RIDGE INTO Random TRAIN AND TEST AND MSE ERROR shown
import numpy as np
from math import *
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Generate some sample data
x1 = np.array(range(1, 366))
y1 = averaged_df['day_average'].values

# Reshape x to a 2D array (required for scikit-learn)
x_reshaped = x1.reshape(-1, 1)

# Create polynomial features (degree=6 for a 6th-degree polynomial)
poly = PolynomialFeatures(degree=6)
X_poly = poly.fit_transform(x_reshaped)

# Scale features
scaler = StandardScaler() # standardize feature by removing mean and scale to unit variance
X_poly_scaled = scaler.fit_transform(X_poly)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_poly_scaled, y1, test_size=0.2, random_state=None)

# Increase regularization strength % 0.0005 gives great results 
alpha = 0.0005


# Create and fit the Ridge Regression model on the training data
ridge_model = Ridge(alpha=alpha)
ridge_model.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred_test = ridge_model.predict(X_test)

# Plot the data and the fitted curve
x_range = np.linspace(min(x1), max(x1), 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
x_range_scaled = scaler.transform(x_range_poly)
y_pred = ridge_model.predict(x_range_scaled)

plt.scatter(x1, y1, label='Test Data', s=5)  # Set the size with the 's' parameter
plt.plot(x_range, y_pred, color='red', label=f'Fitted Curve (Ridge Regression, alpha={alpha})')
plt.xlabel('day')
plt.ylabel('Energy (Watt-Hour)')
plt.legend()
plt.show()
mse_test = mean_squared_error(y_test, y_pred_test)
error = np.sqrt(mse_test)
print(error)
print(ridge_model.coef_)

# test and train in diff colors 
# plot predicted vs true data should be straight line


# In[124]:


# Number 3

# SCALE THIS BY THE AVERAGE TEMP RISE IN UK / YEAR?? for multiplying each year??
# we can take the average between every 2 rows of data as the daily mean then calculate using the formula the multiplier efficiency of the panel 
# only a few years are available 
# some years dont have all the data 

# RUN MAIN CODE FIRST 
current_year = 2015
plots = 4
for i in range(0,plots):
    fileNameT = "C:/Users/tartn/Documents/Solarize_Data/KewTemperatures/midas-open_uk-daily-temperature-obs_dv-202308_greater-london_00723_kew-gardens_qcv-1_2017.csv"
    fileNameT = fileNameT.replace('2020',str(current_year))
    temps = pd.read_csv(fileNameT, skiprows=list(range(0, 90)))

    # extract min and max of each time 
    temperatures = pd.DataFrame()
    temperatures['time'] = temps.iloc[:,0]
    temperatures['max'] = temps.iloc[:,8]
    temperatures['min'] = temps.iloc[:,9]


    tempAvgDouble = []
    for itemp in range(0,len(temperatures['time'])):
        tempAvgDouble.append((temperatures['max'][itemp]+ temperatures['min'][itemp])/2)

    temperatures['averages'] = tempAvgDouble

    temperatures = temperatures.iloc[:-1] # remove the last NaN row
    temperatures.head()

    dayTemps = []
    # finding average temp every 9am=9pm (i think this is the best i can do cuz thats the only available data and night time 9pm-9am is mostly irrelevant)
    for jtemp in range(0,floor(len(temperatures['time'])/2)):

        dayTemps.append(temperatures['averages'][2*jtemp+1])

    tempsAvg = pd.DataFrame()
    colT = 'temperatures' + str(current_year)
    tempsAvg[colT] = dayTemps

    daysInYear = 365
    tempsAvg = tempsAvg.head(daysInYear)


    # Set parameters from panel model 
    initialEff = 21.8*0.985
    degradeRate = 0.25
    TC = -0.29
    standardTestTemp = 25

    # these will change as time progresses 
    yearCount = 5
    currentTemp = 5 # for row in year 

    #EfficiencyFormula =initialEff + TC*(currentTemp-standardTestTemp) - (yearCount-1)*(degradeRate)

    efficiency = []
    averaged_df = averaged_df.head(daysInYear)
    predicted_irradiance = averaged_df

    tempsAvg['finalEfficiency'] = np.round(initialEff + TC*(tempsAvg[colT]-standardTestTemp) - (yearCount-1)*(degradeRate),2)/100

    totalEnergy = pd.DataFrame()
    totalEnergy['energy'] = averaged_df['day_average']*tempsAvg['finalEfficiency'] 


    x = np.array(range(0, len(tempsAvg[colT].values)))
    y = totalEnergy['energy'].values
    plt.scatter(x, y,s=5,label='Output')
    plt.xlabel('day')
    plt.ylabel('Effective Energy (WH) per m^2')
    plt.title('Energy recieved VS Panel Output Energy')

    # NOTE that this uses the average irradiance and ridge regression and temperature data from 2017


    ### this is for comparing with ridge, RUN RIDGE CODE in prev block first!
    plt.scatter(x1, y1, label='Test Data', s=5)  # Set the size with the 's' parameter
    plt.plot(x_range, y_pred, color='red', label=f'Fitted Curve (Ridge Regression, alpha={alpha})')

    ###
    plt.legend()
    
    current_year += 1
    print(current_year)


# In[125]:


# Number 4

# effective output with temperature data but not too good cuz missing data

# RUN NUMBER 1 & 2 First 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import floor

# make sure main code is initialized then Ridge code before running this

start_year = 2000
plots = 20

# Set parameters from panel model
initialEff = 21.8 * 0.985
degradeRate = 0.25
TC = -0.29
standardTestTemp = 25

# Create subplots
fig, axes = plt.subplots(plots, 1, figsize=(10, 6 * plots))

# TEMPS DAY AVG OF ALL YEARS
all_temps = pd.DataFrame()

for i in range(plots):
    current_year = start_year + i  # Increment the year for each iteration
    fileNameT = "C:/Users/tartn/Documents/Solarize_Data/KewTemperatures/midas-open_uk-daily-temperature-obs_dv-202308_greater-london_00723_kew-gardens_qcv-1_2020.csv"
    fileNameT = fileNameT.replace('2020', str(current_year))
    temps = pd.read_csv(fileNameT, skiprows=list(range(0, 90)))

    # Extract min and max of each timestep
    temperatures = pd.DataFrame()
    temperatures['time'] = temps.iloc[:, 0]
    temperatures['max'] = temps.iloc[:, 8]
    temperatures['min'] = temps.iloc[:, 9]

    tempAvgDouble = []
    for itemp in range(0, len(temperatures['time'])):
        tempAvgDouble.append((temperatures['max'][itemp] + temperatures['min'][itemp]) / 2)

    temperatures['averages'] = tempAvgDouble

    # Interpolate missing values
    # WRITE THE CODE to convert time to intervals 
    

    dayTemps = []
    # Finding average temp every 9am-9pm
    # only appends the day and not the night so skip half of it
    for jtemp in range(0, floor(len(temperatures['time']) / 2)): 
        dayTemps.append(temperatures['averages'][2 * jtemp + 1])

    tempsAvg = pd.DataFrame()
    colT = 'temperatures' + str(current_year)
    
    # Makes sure column size uniformity, replace missing values with NaN 
    for days in range(daysInYear - len(dayTemps)):
        dayTemps.append(np.nan)
    
    tempsAvg[colT] = dayTemps
    
    daysInYear = 365 # doesnt take leap years for uniformity
    tempsAvg = tempsAvg.head(daysInYear)

   
     
    #fill in the temosAvg col to be 365 
    ### code 
    all_temps[colT] = tempsAvg[colT]

    # These will change as time progresses
    yearCount = i

    # TEMPERATURE COEFFICIENT FORMULA TO FIND EFFECTIVE EFFICIENCY
    tempsAvg['finalEfficiency'] = np.round(
        initialEff + TC * (tempsAvg[colT] - standardTestTemp) - (i) * (degradeRate), 2) / 100

    totalEnergy = pd.DataFrame()
    totalEnergy['energy'] = averaged_df['day_average'] * tempsAvg['finalEfficiency']

    # Plotting on the i-th subplot
    ax = axes[i]

    x = np.array(range(0, len(tempsAvg[colT].values)))
    y = totalEnergy['energy'].values[:len(x)]  # Ensure y has the same length as x
    ax.scatter(x, y, s=5, label=' Panel Energy Output')
    ax.set_xlabel('day')
    ax.set_ylabel('Effective Energy (WH) per m^2')
    ax.set_title(f'Energy received VS Panel Output Energy - {current_year}')

    # NOTE that this uses the average irradiance and ridge regression (AVERAGE IRRADIANCE )
    ax.scatter(x1, y1, label='Test Data', s=5)  # Set the size with the 's' parameter
    ax.plot(x_range, y_pred, color='red', label=f'Fitted Total Average Irradiance Curve (Ridge Regression, alpha={alpha})')

    ax.legend()

    print(current_year)
    print(fileNameT)

plt.tight_layout()
plt.show()




# cant see the output in some cuz the temperature data isnt there 

# NOW PUT THE FIXED DATA INTO THE PLOTS!!! -> Label each df properly and see what went wrong maybe sections of code redefines


# In[126]:


# NUMBER 5

# lasso regression to model temperature to reduce noise and fill in missing data

# SUCCESS for creating new fd of temps 
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# Assuming all_temps is your DataFrame
all_temps_new = all_temps.drop(all_temps.columns[:10], axis=1)

# Replace NaN values with the average of 2 days before and 2 days after
all_temps_new = all_temps_new.interpolate(method='linear', limit=2, limit_direction='both')

from sklearn.linear_model import LassoCV

# Your existing code...

# Perform a polynomial regression for each year
models = {}
fitted_temps = pd.DataFrame()
for year in all_temps_new.columns:
    # Prepare the data
    X = np.array(range(len(all_temps_new))).reshape(-1, 1)  # Day of the year
    y = all_temps_new[year].values  # Temperature

    # Remove rows with NaN values for model fitting
    mask = ~np.isnan(y)
    X_fit = X[mask]
    y_fit = y[mask]

    # Transform the features to higher degree features.
    poly = PolynomialFeatures(degree=2)
    X_poly_fit = poly.fit_transform(X_fit)

    # Create and fit the model
    alpha_value = 10  # Set your alpha value here
    model = Lasso(alpha=alpha_value)
    model.fit(X_poly_fit, y_fit)

    # Store the model
    models[year] = model


    # Print the optimal alpha value
    print(f'Alpha for {year}: {alpha_value}')


    # Store the model
    models[year] = model

    # Plot the original data
    plt.scatter(X, y, color='blue',s=1)

    # Generate predictions for the entire year
    X_poly = poly.transform(X)
    y_pred = model.predict(X_poly)

    # Plot the regression curve
    plt.plot(X, y_pred, color='red')

    # Set the title and show the plot
    plt.title(f'Regression Curve for {year}')
    plt.show()

    # Add the predictions to the new DataFrame
    fitted_temps[year] = pd.Series(y_pred)

# Ensure the new DataFrame has the same index as the original DataFrame
fitted_temps.index = all_temps_new.index


# In[127]:


# NUMBER 6

# Final Temperatures over the years

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


# Create an empty DataFrame to store the future predictions
temps_projection_all = pd.DataFrame()

for index, row in fitted_temps.iterrows():
    # Reshape the data
    X = np.array(range(len(row))).reshape(-1, 1)
    y = row.values.reshape(-1, 1)

    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Generate predictions for the existing data
    y_pred = model.predict(X)

    # Generate predictions for 20 future data points
    X_future = np.array(range(len(row), len(row) + 25)).reshape(-1, 1)
    y_future = model.predict(X_future)

    # Add the future predictions to the new DataFrame
    temps_projection_all = temps_projection_all.append(pd.Series(y_future.flatten(), name=index))
    ''' 
    # Plot the original data, the regression line, and the future predictions
    plt.figure()
    plt.scatter(X, y, color='black')
    plt.plot(X, y_pred, color='blue', linewidth=3)
    plt.plot(X_future, y_future, color='red', linewidth=3)
    plt.title(f'Linear Regression and Future Projections for Row {index}')
    plt.show()
    '''
# Print the new DataFrame with the future predictions
temps_projection_all.columns = [str(2020 + int(col)) for col in temps_projection_all.columns]

import pandas as pd
import matplotlib.pyplot as plt


plt.figure(figsize=(15, 6))

# Create a colormap
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, len(df_future.columns)))

# Loop through each column in the DataFrame
for i, column in enumerate(df_future.columns):
    plt.plot(df_future.index, df_future[column], label=column, color=colors[i])

plt.xlabel('Days of the year')
plt.ylabel('Temperature (Celcius)')
plt.title('Temperature throughout the years')
plt.legend()
plt.show()


# In[128]:


df_future


# In[129]:


temps_projection_all


# In[130]:


averaged_df['day_average']


# In[131]:


# NUMBER 7

# Final Efficiency over the years

import pandas as pd
import numpy as np
import matplotlib.cm as cm


# Initialize a new dataframe to store the results
df_efficiency = pd.DataFrame()

# Iterate over the columns in the original dataframe
for iteration, year in enumerate(temps_projection_all.columns):
    # Get the temperature data for the year
    year_temps = temps_projection_all[year]
    
    # Calculate the efficiency for each day and store it in the new dataframe
    df_efficiency[year] = np.round(
        initialEff + TC * (year_temps - standardTestTemp) - (iteration) * (degradeRate), 2) / 100

# Now df_efficiency contains the efficiency data

# Create a new figure
plt.figure()

# Get a colormap
cmap = cm.get_cmap('viridis')  # Change 'viridis' to any colormap you like

# Iterate over the columns in the dataframe
for iteration, year in enumerate(df_efficiency.columns):
    # Get the color for the line based on its position
    color = cmap(float(iteration) / len(df_efficiency.columns))
    
    # Plot the data for the year with the specified color
    plt.plot(df_efficiency[year], color=color, label=year)

# Add a legend
plt.legend()

plt.xlabel('Days of the year')
plt.ylabel('Solar Panel Efficiency (%)')
plt.title('Efficiency throughout the years')
plt.legend()
plt.show()

# Show the plot
plt.show()


# In[132]:


# NUMBER 8

#assuming low sellback price

hour_effective_output = df_efficiency.apply(lambda x: x * predicted_irradiance['day_average'])

# convert watt-hours/hour to kilowatt-hours/day
day_effective_output = 24*hour_effective_output/1000 

# now we need to customize the panel specifications and the other parameters

price_rate = 0.041 # GOV SMART EXPORT GUARANTEE export rate 18p/KWH RANGE IS AROUND 3-24 
number_of_panels = 10    # years
lifetime = 25
size = 1.5     # meters squared THIS CAN BE INACCURATE CUZ NOT ALL SURFACE AREA IS THE PV CELL 
revenue = []
for column in day_effective_output:
    revenue.append(price_rate*day_effective_output[column].sum())
    
# starting from the 21st year 2021
real_revenue = [x*size for x in revenue]

print(f"Revenue Over 25 Years per 1 panel: {sum(real_revenue)} pounds")

lifetime_revenue = round(number_of_panels*sum(real_revenue),2)
print(f"Total Revenue Over 25 Years If we assume {number_of_panels} panels {lifetime_revenue} pounds")

annual_revenue = round(lifetime_revenue/lifetime,2)
print(f"Revenue Average per 1 year for {number_of_panels} panels: {annual_revenue} pounds")


# In[133]:


# NUMBER 9 

#assuming high sellback price 

hour_effective_output = df_efficiency.apply(lambda x: x * predicted_irradiance['day_average'])

# convert watt-hours/hour to kilowatt-hours/day
day_effective_output = 24*hour_effective_output/1000 

# now we need to customize the panel specifications and the other parameters

price_rate = 0.15 # GOV SMART EXPORT GUARANTEE export rate 18p/KWH RANGE IS AROUND 3-24 
number_of_panels = 10    # years
lifetime = 25
size = 1.5     # meters squared THIS CAN BE INACCURATE CUZ NOT ALL SURFACE AREA IS THE PV CELL 
revenue = []
for column in day_effective_output:
    revenue.append(price_rate*day_effective_output[column].sum())
    
# starting from the 21st year 2021
real_revenue = [x*size for x in revenue]

print(f"Revenue Over 25 Years per 1 panel: {sum(real_revenue)} pounds")

lifetime_revenue = round(number_of_panels*sum(real_revenue),2)
print(f"Total Revenue Over 25 Years If we assume {number_of_panels} panels {lifetime_revenue} pounds")

annual_revenue = round(lifetime_revenue/lifetime,2)
print(f"Revenue Average per 1 year for {number_of_panels} panels: {annual_revenue} pounds")

x = range(lifetime)
plt.plot(x,real_revenue)

plt.xlabel('years')
plt.ylabel('revenue (pounds)')
plt.title('Revenue throughout the years')


# In[ ]:


# Number 10

# What if temperature remained as 2020 and did not decrease 

