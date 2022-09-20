## importing needed packages
import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

##### Calculating Descriptive Statistics ##### 

## creating data to work with; this is python's list containing some arbitrary numeric data
x = [8.0, 1, 2.5, 4, 28.0] ## creating a list where the variable x is equal to [8.0, 1, 2.5, 4, 28.0]
x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0] ## creating a list where the variable x_with_nan is equal to [8.0, 1, 2.5, math.nan, 4, 28.0]
x ## printing variable x 
x_with_nan ## printing variable x_with_nan 

## methods below check whether list has a 'nan'
math.isnan(np.nan), np.isnan(math.nan)
math.isnan(x_with_nan[3]), np.isnan(x_with_nan[3])

## creating a np.ndarray and pd.Series objects that correspond to x and x_with_nan:
y, y_with_nan = np.array(x), np.array(x_with_nan)
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
## the list as Numpy array 
y
y_with_nan
## the list as a Pandas series
z
z_with_nan

##### Measures of Central Tendency ##### 

#### Mean ####

## method of calculating mean without importing any packages
sum_len_mean_ = sum(x) / len(x)
sum_len_mean_

## built-in python statistic functions 
mean_ = statistics.mean(x)
mean_
fmean_ = statistics.fmean(x) ##fmean() is a faster alternative to mean(). It always returns a floating-point number.
fmean_

## However, if there are nan values among your data, then statistics.mean() and statistics.fmean() will return nan as the output:
mean_with_nan = statistics.mean(x_with_nan)
mean_with_nan
fmean_with_nan = statistics.fmean(x_with_nan)
fmean_with_nan

### NumPy Array ###

## if using numpy can use following method
np_mean = np.mean(y)
np_mean
## another method
np_mean_2 = y.mean() 
np_mean_2

## The function mean() and method .mean() from NumPy return the same result as statistics.mean(). This is also the case when there are nan values among your data:
np.mean(y_with_nan)
y_with_nan.mean()

## If you prefer to ignore nan values, then you can use 
np.nanmean(y_with_nan)
## nanmean() simply ignores all nan values

### Pandas Series  ###

## pandas.series objects also have the .mean() method
mean_ = z.mean()
mean_

## used simiarly to NumPy, however pandas ignore 'nan' values on default
z_with_nan.mean()

#### Weighted Mean #### 

# For example, say that you have a set in which 20% of all items are equal to 2, 50% of the items are equal to 4, and the remaining 30% of the items are equal to 8. You can calculate the mean of such a set like this:
0.2 * 2 + 0.5 * 4 + 0.3 * 8

## You can implement the weighted mean in pure Python by combining sum() with either range() or zip() 
x2 = [8.0, 1, 2.5, 4, 28.0]
w = [0.1, 0.2, 0.3, 0.25, 0.15]
wmean_range = sum(w[i] * x2[i] for i in range(len(x2))) / sum(w)
wmean_range
wmean_zip = sum(x2_ * w_ for (x2_, w_) in zip(x2, w)) / sum(w)
wmean_zip

## if have larger dataset NumPy is better solution; can use np.average() in order to get weighted mean of NumPy arrays or Pandas series 
y, z, w = np.array(x), pd.Series(x), np.array(w)
wmean = np.average(y, weights=w)
wmean
wmean = np.average(z, weights=w)
wmean

## can also use product w * y with np.sum() or .sum() to calculate weighted mean 
(w * y).sum() / w.sum()

## if dataset has 'nan' values be careful
w = np.array([0.1, 0.2, 0.3, 0.0, 0.2, 0.1])
(w * y_with_nan).sum() / w.sum()
np.average(y_with_nan, weights=w)
np.average(z_with_nan, weights=w)

#### Harmonic Mean #### 

## One variant of the pure Python implementation of the harmonic mean is this:
hmean = len(x) / sum(1 / item for item in x)
hmean

## You can also calculate this measure with statistics.harmonic_mean():
hmean = statistics.harmonic_mean(x)
hmean

## if have 'nan' in dataset may return 'nan'
statistics.harmonic_mean(x_with_nan)
statistics.harmonic_mean([1, 0, 2])
statistics.harmonic_mean([1, 2, -2]) ## This line will raise statistic-error due to having negative # 

## A third way to calculate the harmonic mean is to use scipy.stats.hmean():
scipy.stats.hmean(y)
scipy.stats.hmean(z)

#### Geometric Mean #### 

## can implement geometric mean in pure python 
gmean = 1
for item in x:
    gmean *= item
gmean **= 1 / len(x)
gmean

## statistics.geometric_mean() converts all values to floating-point numbers and returns their geometric mean
gmean = statistics.geometric_mean(x)
gmean

## If you pass data with nan values, then statistics.geometric_mean() will behave like most similar functions and return nan 
gmean = statistics.geometric_mean(x_with_nan)
gmean

## You can also get the geometric mean with scipy.stats.gmean() 
scipy.stats.gmean(y)
scipy.stats.gmean(z)

#### Median #### 

## one of many possible pure Python implementations of the median
n = len(x)
if n % 2:
    median_ = sorted(x)[round(0.5*(n-1))]
else:
    x_ord, index = sorted(x), round(0.5 * n)
    median_ = 0.5 * (x_ord[index-1] + x_ord[index])
median_

## Sorting: the elements of the dataset
## Finding: the middle element(s) in the sorted dataset

## You can get the median with statistics.median() 
median_ = statistics.median(x)
median_
median_ = statistics.median(x[:-1])
median_

## median_low() and median_high() are two more functions related to the median in the Python statistics library. They always return an element from the dataset
## If the number of elements is odd, then there’s a single middle value, so these functions behave just like median().
## If the number of elements is even, then there are two middle values. In this case, median_low() returns the lower and median_high() the higher middle value.
statistics.median_low(x[:-1])
statistics.median_high(x[:-1])

## median(), median_low(), and median_high() don’t return nan when there are nan values among the data points
statistics.median(x_with_nan)
statistics.median_low(x_with_nan)
statistics.median_high(x_with_nan)

## You can also get the median with np.median():
median_ = np.median(y)
median_
median_ = np.median(y[:-1])
median_

## However, if there’s a nan value in your dataset, then np.median() issues the RuntimeWarning and returns nan. 
## If this behavior is not what you want, then you can use nanmedian() to ignore all nan values
np.nanmedian(y_with_nan)
np.nanmedian(y_with_nan[:-1])

## Pandas Series objects have the method .median() that ignores nan values by default 
z.median()
z_with_nan.median()

#### Mode #### 

## pure python 
u = [2, 3, 2, 8, 12]
mode_ = max((u.count(item), item) for item in set(u))[1] ## don’t have to use set(u). Instead, you might replace it with just u and iterate over the entire list
mode_
## use u.count() and the # value with the highest occurence is the mode 

## You can obtain the mode with statistics.mode() and statistics.multimode():
mode_ = statistics.mode(u)
mode_
mode_ = statistics.multimode(u)
mode_

## If there’s more than one modal value, then mode() raises StatisticsError, while multimode() returns the list with all modes
v = [12, 15, 12, 15, 21, 15, 12]
statistics.mode(v)  # Raises StatisticsError
statistics.multimode(v)

## statistics.mode() and statistics.multimode() handle nan values as regular values and can return nan as the modal value 
statistics.mode([2, math.nan, 2])
statistics.multimode([2, math.nan, 2])
statistics.mode([2, math.nan, 0, math.nan, 5])
statistics.multimode([2, math.nan, 0, math.nan, 5])

##You can also get the mode with scipy.stats.mode():
u, v = np.array(u), np.array(v)
mode_ = scipy.stats.mode(u)
mode_
mode_ = scipy.stats.mode(v)
mode_
## returns mode and # of occurence 

## You can get the mode and its number of occurrences as NumPy arrays with dot notation
mode_.mode
mode_.count

## Pandas Series objects have the method .mode() that handles multimodal values well and ignores nan values by default
u, v, w = pd.Series(u), pd.Series(v), pd.Series([2, 2, math.nan])
u.mode()
v.mode()
w.mode()

##### Measure of Variability ##### 

### Variance ### 

## calculate variance with pure python 
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
var_

## the shorter and more elegant solution is to call the existing function statistics.variance()
var_ = statistics.variance(x)
var_

## variance() can avoid calculating the mean if you provide the mean explicitly as the second argument: 
statistics.variance(x, mean_)

## If you have nan values among your data, then statistics.variance() will return nan
statistics.variance(x_with_nan)

## You can also calculate the sample variance with NumPy. You should use the function np.var() or the corresponding method .var()
var_ = np.var(y, ddof=1)
var_
var_ = y.var(ddof=1)
var_

## If you have nan values in the dataset, then np.var() and .var() will return nan 
np.var(y_with_nan, ddof=1)
y_with_nan.var(ddof=1)

## If you want to skip nan values, then you should use np.nanvar()
np.nanvar(y_with_nan, ddof=1)
## np.nanvar() ignores nan values. It also needs you to specify ddof=1

## pd.Series objects have the method .var() that skips nan values by default
z.var(ddof=1)
z_with_nan.var(ddof=1) 

### Standard Deviation ### 

## calculate via pure python 
std_ = var_ ** 0.5
std_

## can also use statistics.stdev()
std_ = statistics.stdev(x)
std_

## You can use the function std() and the corresponding method .std() to calculate the standard deviation. 
## If there are nan values in the dataset, then they’ll return nan. 
## To ignore nan values, you should use np.nanstd(). 
## You use std(), .std(), and nanstd() from NumPy as you would use var(), .var(), and nanvar()
np.std(y, ddof=1)
y.std(ddof=1)
np.std(y_with_nan, ddof=1)
y_with_nan.std(ddof=1)
np.nanstd(y_with_nan, ddof=1)

## Don’t forget to set the delta degrees of freedom to 1!

## pd.Series objects also have the method .std() that skips nan by default 
z.std(ddof=1)
z_with_nan.std(ddof=1)

### Skewness ### 

## Once you’ve calculated the size of your dataset n, the sample mean mean_, and the standard deviation std_, you can get the sample skewness with pure Python
x = [8.0, 1, 2.5, 4, 28.0]
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
std_ = var_ ** 0.5
skew_ = (sum((item - mean_)**3 for item in x)
         * n / ((n - 1) * (n - 2) * std_**3))
skew_
## The skewness is positive, so x has a right-side tail

## You can also calculate the sample skewness with scipy.stats.skew()
y, y_with_nan = np.array(x), np.array(x_with_nan)
scipy.stats.skew(y, bias=False)
scipy.stats.skew(y_with_nan, bias=False)

## Pandas Series objects have the method .skew() that also returns the skewness of a dataset
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
z.skew()
z_with_nan.skew()

### Percentiles ### 

## If you want to divide your data into several intervals, then you can use statistics.quantiles()
x = [-5.0, -1.1, 0.1, 2.0, 8.0, 12.8, 21.0, 25.8, 41.0]
statistics.quantiles(x, n=2)
statistics.quantiles(x, n=4, method='inclusive')

## You can also use np.percentile() to determine any sample percentile in your dataset. 
## For example, this is how you can find the 5th and 95th percentiles
y = np.array(x)
np.percentile(y, 5)
np.percentile(y, 95)

## The percentile can be a number between 0 and 100 like in the example above, but it can also be a sequence of numbers
np.percentile(y, [25, 50, 75])
np.median(y)
## This code calculates the 25th, 50th, and 75th percentiles all at once.
## The first statement returns the array of quartiles. 
## The second statement returns the median, so you can confirm it’s equal to the 50th percentile, which is 8.0.

## If you want to ignore nan values, then use np.nanpercentile() instead 
y_with_nan = np.insert(y, 2, np.nan)
y_with_nan
np.nanpercentile(y_with_nan, [25, 50, 75])

## NumPy also offers you very similar functionality in quantile() and nanquantile(). If you use them, then you’ll need to provide the quantile values as the numbers between 0 and 1 instead of percentiles
np.quantile(y, 0.05)
np.quantile(y, 0.95)
np.quantile(y, [0.25, 0.5, 0.75])
np.nanquantile(y_with_nan, [0.25, 0.5, 0.75])

## pd.Series objects have the method .quantile()
z, z_with_nan = pd.Series(y), pd.Series(y_with_nan)
z.quantile(0.05)
z.quantile(0.95)
z.quantile([0.25, 0.5, 0.75])
z_with_nan.quantile([0.25, 0.5, 0.75])

### Range ###

## You can get it with the function np.ptp() 
np.ptp(y)
np.ptp(z)
np.ptp(y_with_nan)
np.ptp(z_with_nan)

## max() and min() from the Python standard library
## amax() and amin() from NumPy
## nanmax() and nanmin() from NumPy to ignore nan values
## .max() and .min() from NumPy
## .max() and .min() from Pandas to ignore nan values by default
np.amax(y) - np.amin(y)
np.nanmax(y_with_nan) - np.nanmin(y_with_nan)
y.max() - y.min()
z.max() - z.min()
z_with_nan.max() - z_with_nan.min()

## The interquartile range is the difference between the first and third quartile. Once you calculate the quartiles, you can take their difference:
quartiles = np.quantile(y, [0.25, 0.75])
quartiles[1] - quartiles[0]
quartiles = z.quantile([0.25, 0.75])
quartiles[0.75] - quartiles[0.25]

#### Summary of Descriptive Statistics #### 

## You can use scipy.stats.describe() like this
result = scipy.stats.describe(y, ddof=1, bias=False)
result

## nobs: the number of observations or elements in your dataset
## minmax: the tuple with the minimum and maximum values of your dataset
## mean: the mean of your dataset
## variance: the variance of your dataset
## skewness: the skewness of your dataset
## kurtosis: the kurtosis of your dataset
## You can access particular values with dot notation:
result.nobs
result.minmax[0]  # Min
result.minmax[1]  # Max
result.mean
result.variance
result.skewness
result.kurtosis

## Pandas Series objects have the method .describe()
result = z.describe()
result

## count: the number of elements in your dataset
## mean: the mean of your dataset
## std: the standard deviation of your dataset
## min and max: the minimum and maximum values of your dataset
## 25%, 50%, and 75%: the quartiles of your dataset

## You can access each item of result with its label 
result['mean']
result['std']
result['min']
result['max']
result['25%']
result['50%']
result['75%']

#### Measures of Correlation Between Pairs of Data #### 

## Positive correlation exists when larger values of 𝑥 correspond to larger values of 𝑦 and vice versa.
## Negative correlation exists when larger values of 𝑥 correspond to smaller values of 𝑦 and vice versa.
## Weak or no correlation exists if there is no such apparent relationship.

## create two Python lists and use them to get corresponding NumPy arrays and Pandas Series
x = list(range(-10, 11))
y = [0, 2, 2, 2, 2, 3, 3, 6, 7, 4, 7, 6, 6, 9, 4, 5, 5, 10, 11, 12, 14]
x_, y_ = np.array(x), np.array(y)
x__, y__ = pd.Series(x_), pd.Series(y_)

### Covariance ### 

## If the correlation is positive, then the covariance is positive, as well. A stronger relationship corresponds to a higher value of the covariance.
## If the correlation is negative, then the covariance is negative, as well. A stronger relationship corresponds to a lower (or higher absolute) value of the covariance.
## If the correlation is weak, then the covariance is close to zero.

## calculate covariance in pure python 
n = len(x)
mean_x, mean_y = sum(x) / n, sum(y) / n
cov_xy = (sum((x[k] - mean_x) * (y[k] - mean_y) for k in range(n))
          / (n - 1))
cov_xy

## NumPy has the function cov() that returns the covariance matrix 
cov_matrix = np.cov(x_, y_)
cov_matrix

## can check to see that this is true
x_.var(ddof=1)
y_.var(ddof=1)

## other two elements of the covariance matrix are equal and represent the actual covariance between x and y
cov_xy = cov_matrix[0, 1]
cov_xy
cov_xy = cov_matrix[1, 0]
cov_xy

## Pandas Series have the method .cov() that you can use to calculate the covariance:
cov_xy = x__.cov(y__)
cov_xy
cov_xy = y__.cov(x__)
cov_xy

### Correlation Coefficient ### 

## The value 𝑟 > 0 indicates positive correlation.
## The value 𝑟 < 0 indicates negative correlation.
## The value r = 1 is the maximum possible value of 𝑟. It corresponds to a perfect positive linear relationship between variables.
## The value r = −1 is the minimum possible value of 𝑟. It corresponds to a perfect negative linear relationship between variables.
## The value r ≈ 0, or when 𝑟 is around zero, means that the correlation between variables is weak.

## If you have the means (mean_x and mean_y) and standard deviations (std_x, std_y) for the datasets x and y, as well as their covariance cov_xy, then you can calculate the correlation coefficient with pure Python
var_x = sum((item - mean_x)**2 for item in x) / (n - 1)
var_y = sum((item - mean_y)**2 for item in y) / (n - 1)
std_x, std_y = var_x ** 0.5, var_y ** 0.5
r = cov_xy / (std_x * std_y)
r

## scipy.stats has the routine pearsonr() that calculates the correlation coefficient and the 𝑝-value
r, p = scipy.stats.pearsonr(x_, y_)
r
p

## Similar to the case of the covariance matrix, you can apply np.corrcoef() with x_ and y_ as the arguments and get the correlation coefficient matrix
corr_matrix = np.corrcoef(x_, y_)
corr_matrix

## You can get the correlation coefficient with scipy.stats.linregress() 
scipy.stats.linregress(x_, y_)

## To access particular values from the result of linregress(), including the correlation coefficient, use dot notation
result = scipy.stats.linregress(x_, y_)
r = result.rvalue
r

## Pandas Series have the method .corr() for calculating the correlation coefficient:
r = x__.corr(y__)
r
r = y__.corr(x__)
r

#### Working with 2D Data #### 

### Axes ### 

## Start by creating a 2D NumPy array:
a = np.array([[1, 1, 1],
              [2, 3, 1],
              [4, 9, 2],
              [8, 27, 4],
              [16, 1, 1]])
a 

## once have 2D dataset; can apply Python statistics functions and methods to it just as you would to 1D data
np.mean(a)
a.mean()
np.median(a)
a.var(ddof=1)

## axis=None says to calculate the statistics across all data in the array. The examples above work like this. This behavior is often the default in NumPy.
## axis=0 says to calculate the statistics across all rows, that is, for each column of the array. This behavior is often the default for SciPy statistical functions.
## axis=1 says to calculate the statistics across all columns, that is, for each row of the array.
np.mean(a, axis=0)
a.mean(axis=0)

## If you provide axis=1 to mean(), then you’ll get the results for each row:
np.mean(a, axis=1)
a.mean(axis=1)

## The parameter axis works the same way with other NumPy functions and methods:
np.median(a, axis=0)
np.median(a, axis=1)
a.var(axis=0, ddof=1)
a.var(axis=1, ddof=1)

## This is very similar when you work with SciPy statistics functions. 
## But remember that in this case, the default value for axis is 0
scipy.stats.gmean(a)  # Default: axis=0
scipy.stats.gmean(a, axis=0)

## If you specify axis=1, then you’ll get the calculations across all columns, that is for each row 
scipy.stats.gmean(a, axis=1)

## If you want statistics for the entire dataset, then you have to provide axis=None:
scipy.stats.gmean(a, axis=None)

## You can get a Python statistics summary with a single function call for 2D data with scipy.stats.describe(). 
## It works similar to 1D arrays, but you have to be careful with the parameter axis
scipy.stats.describe(a, axis=None, ddof=1, bias=False)
scipy.stats.describe(a, ddof=1, bias=False)  # Default: axis=0
scipy.stats.describe(a, axis=1, ddof=1, bias=False)

## You can get a particular value from the summary with dot notation 
result = scipy.stats.describe(a, axis=1, ddof=1, bias=False)
result.mean

#### DataFrames #### 

## Using the array a and create a DataFrame
row_names = ['first', 'second', 'third', 'fourth', 'fifth']
col_names = ['A', 'B', 'C']
df = pd.DataFrame(a, index=row_names, columns=col_names)
df

## If you call Python statistics methods without arguments, then the DataFrame will return the results for each column 
df.mean()
df.var()

##  If you want the results for each row, then just specify the parameter axis=1:
df.mean(axis=1)
df.var(axis=1)

## You can isolate each column of a DataFrame like
df['A']

## Now, you have the column 'A' in the form of a Series object and you can apply the appropriate methods
df['A'].mean()
df['A'].var()

## Sometimes, you might want to use a DataFrame as a NumPy array and apply some function to it. 
## It’s possible to get all data from a DataFrame with .values or .to_numpy()
df.values
df.to_numpy()

## Like Series, DataFrame objects have the method .describe() that returns another DataFrame with the statistics summary for all columns
df.describe()
## count: the number of items in each column
## mean: the mean of each column
## std: the standard deviation
## min and max: the minimum and maximum values
## 25%, 50%, and 75%: the percentiles

## You can access each item of the summary like
df.describe().at['mean', 'A']
df.describe().at['50%', 'B']

#### Visualizing Data #### 

### Box Plot ### 

## creating some data to represent with a box plot
np.random.seed(seed=0)
x = np.random.randn(1000)
y = np.random.randn(100)
z = np.random.randn(10)

## Now that you have the data to work with, you can apply .boxplot() to get the box plot
fig, ax = plt.subplots()
ax.boxplot((x, y, z), vert=False, showmeans=True, meanline=True,
           labels=('x', 'y', 'z'), patch_artist=True,
           medianprops={'linewidth': 2, 'color': 'purple'},
           meanprops={'linewidth': 2, 'color': 'red'})
plt.show()

## x is your data.
## vert sets the plot orientation to horizontal when False. The default orientation is vertical.
## showmeans shows the mean of your data when True.
## meanline represents the mean as a line when True. The default representation is a point.
## labels: the labels of your data.
## patch_artist determines how to draw the graph.
## medianprops denotes the properties of the line representing the median.
## meanprops indicates the properties of the line or dot representing the mean.

### Histogram ### 

## The function np.histogram() is a convenient way to get data for histograms 
hist, bin_edges = np.histogram(x, bins=10)
hist
bin_edges

## hist contains the frequency or the number of items corresponding to each bin.
## bin_edges contains the edges or bounds of the bin.

## What histogram() calculates, .hist() can show graphically 
fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=False)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()

## It’s possible to get the histogram with the cumulative numbers of items if you provide the argument cumulative=True to .hist()
fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=True)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()

### Pie Charts ### 

## data defining 
x, y, z = 128, 256, 1024

## creating pie chat with pie.()
fig, ax = plt.subplots()
ax.pie((x, y, z), labels=('x', 'y', 'z'), autopct='%1.1f%%')
plt.show()
## The first argument of .pie() is your data
## the second is the sequence of the corresponding labels. 
## autopct defines the format of the relative frequencies shown on the figure. 

### Bar Charts ### 

## data generating 
x = np.arange(21)
y = np.random.randint(21, size=21)
err = np.random.randn(21)

## You can create a bar chart with .bar() if you want vertical bars or .barh() if you’d like horizontal bars
fig, ax = plt.subplots()
ax.bar(x, y, yerr=err) ## .barh() for horizontal bars .bar() for vertical bars 
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

### XY Plots ### 

## data generate
x = np.arange(21)
y = 5 + 2 * x + 2 * np.random.randn(21)
slope, intercept, r, *__ = scipy.stats.linregress(x, y)
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'

## linregress returns several values. You’ll need the slope and intercept of the regression line, as well as the correlation coefficient r. Then you can apply .plot() to get the x-y plot
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=0, marker='s', label='Data points')
ax.plot(x, intercept + slope * x, label=line)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(facecolor='white')
plt.show()

### Heatmaps ### 

## You can create the heatmap for a covariance matrix with .imshow()
matrix = np.cov(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()

## You can obtain the heatmap for the correlation coefficient matrix following the same logic
matrix = np.corrcoef(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()