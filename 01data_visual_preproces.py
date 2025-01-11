##############################################################################
### Raw data visualization and normalization for preparation of NN training
### Data from LLC converter simulation on PSIM
### Author: Fanghao Tian
### Date: 2024-Oct-30
##############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

inputfolder = 'InputData'
outputfolder = 'OutputData'


## Section 1: 
# Load the raw data of PSIM simulation
# raw_PSIM = pd.read_csv(os.path.join(inputfolder, 'Original_PSIM.csv'))
# #find the column index of the data
# print(raw_PSIM.columns)

# #delete the columns that are "Unnamed
# raw_PSIM = raw_PSIM.drop(raw_PSIM.columns[raw_PSIM.columns.str.contains('unnamed',case = False)],axis = 1)
# #find the column index of the data
# print(raw_PSIM.columns)
# print(raw_PSIM.head())

# #create a new dataframe to store the data
# standard_data = pd.DataFrame(columns=['fn','Ln','Q','Vout_unit','Ir_rms_unit','Ir_d_unit','Vcs_max_unit'])
# fbase = 100e3
# standard_data['fn'] = raw_PSIM['f']/fbase
# standard_data['Ln'] = raw_PSIM['Lm']/raw_PSIM['Lr']
# standard_data['Q'] = np.pi**2*np.sqrt(raw_PSIM['Lr']/raw_PSIM['Cr'])/(8*raw_PSIM['n']**2*raw_PSIM['R'])
# standard_data['Vout_unit'] = raw_PSIM['Vo_avg']*raw_PSIM['n']/raw_PSIM['Vin']
# Ibase = raw_PSIM['n']*raw_PSIM['Vo_avg']/np.sqrt(raw_PSIM['Lr']/raw_PSIM['Cr'])
# standard_data['Ir_rms_unit'] = raw_PSIM['Ir_rms']/Ibase
# standard_data['Ir_d_unit'] = 0.5*(abs(raw_PSIM['Ir_fall']+raw_PSIM['Ir_rise']))/Ibase
# standard_data['Vcs_max_unit'] = raw_PSIM['Vcs_max']/raw_PSIM['Vo_avg']/raw_PSIM['n']

# print(standard_data.head())
# print(standard_data.max())
# print(standard_data.min())
# standard_data.to_csv(os.path.join(outputfolder, 'Standard_data.csv'), index=False)


##Section2: Data visualization

rawdata = pd.read_csv(os.path.join(outputfolder,'Standard_data_big.csv'))
# rawdata = rawdata[(rawdata.loc[:, 'TRUE'] == 1)]     #只选取你标记了1的数据
# rawdata.to_csv('LLC_Voltage_Gain_Model_onlyPO.csv')
# Only pick the columns we need
test_rawdata = pd.read_csv(os.path.join(outputfolder,'Standard_data_test.csv'))
# rawdata = rawdata[['f','Ln','Q','M']]
print(rawdata.head())

ln = rawdata['Ln']
q = rawdata['Q']
fn = rawdata['fn']
# add a new column "V_FHA" to the rawdata
rawdata['V_FHA'] = 1/np.sqrt((1+1/ln-1/(ln*fn**2))**2+q**2*(1/fn-fn)**2)

#add a new column "Irms_cal" to the rawdata

ln = test_rawdata['Ln']
q = test_rawdata['Q']
fn = test_rawdata['fn']
# add a new column "V_FHA" to the rawdata
test_rawdata['V_FHA'] = 1/np.sqrt((1+1/ln-1/(ln*fn**2))**2+q**2*(1/fn-fn)**2)

for idx, column in enumerate(rawdata.columns):
    #对部分分布不均的数据（大量数据集中在小值的情况）取log10
    # if column == 'Cr' or column == 'Lm' or column == 'R' or column == 'Vo':
    if column == 'Vout_unit' or column == 'Ir_rms_unit' or column == 'Ir_d_unit' or column == 'Vcs_max_unit' or column == 'V_FHA':
        rawdata[column] = np.log10(rawdata[column])

for idx, column in enumerate(test_rawdata.columns):
    #对部分分布不均的数据（大量数据集中在小值的情况）取log10
    # if column == 'Cr' or column == 'Lm' or column == 'R' or column == 'Vo':
    if column == 'Vout_unit' or column == 'Ir_rms_unit' or column == 'Ir_d_unit' or column == 'Vcs_max_unit' or column == 'V_FHA':
        test_rawdata[column] = np.log10(test_rawdata[column])

min_values = rawdata.min()
max_values = rawdata.max()

boundary_df = pd.DataFrame([min_values, max_values], index=['min', 'max']).transpose()
boundary_df.to_csv(os.path.join(outputfolder,'boundary_big.csv'), index_label='Column')

normalized_df = pd.DataFrame(columns=rawdata.columns)
normalized_test_df = pd.DataFrame(columns=rawdata.columns)




n_cols = len(rawdata.columns)
n_rows = int(np.ceil(n_cols / 2))  # Adjust the denominator to change layout
fig, axes = plt.subplots(n_rows, 2, figsize=(15, n_rows * 5))
axes = axes.flatten()  # Flatten in case of a single row


# Iterate through each column and its corresponding subplot
for idx, column in enumerate(rawdata.columns):
    #对部分分布不均的数据（大量数据集中在小值的情况）取log10
    # if column == 'Cr' or column == 'Lm' or column == 'R' or column == 'Vo':
    # if column == 'M':
    #     rawdata[column] = np.log10(rawdata[column])

    # Normalize the data to [0, 1]
    normalized_data = (rawdata[column] - rawdata[column].min()) / (rawdata[column].max() - rawdata[column].min())
    # normalized_data = rawdata[column]
    normalized_df[column] = normalized_data  # Store the normalized data
    
    normalized_data_test = (test_rawdata[column] - rawdata[column].min()) / (rawdata[column].max() - rawdata[column].min())
    # normalized_data_test = test_rawdata[column]
    normalized_test_df[column] = normalized_data_test  # Store the normalized data
    
    
    
    # Bin the data
    bins = np.arange(0, 1.06, 0.03)  # Adjust the step size if necessary
    binned_data = pd.cut(normalized_data, bins, include_lowest=True, right=False)
    
    # Count the number of data points in each bin
    bin_counts = binned_data.value_counts(sort=False)
    
    # Plot the distribution
    bin_counts.plot(kind='bar', ax=axes[idx])
    axes[idx].set_title(f'Distribution for {column}')
    axes[idx].set_xlabel('Normalized Data Interval')
    axes[idx].set_ylabel('Count')
    axes[idx].tick_params(axis='x', rotation=45)

# Adjust layout
plt.tight_layout()
normalized_df.to_csv(os.path.join(outputfolder,"normalized_big.csv"),index=False)
normalized_test_df.to_csv(os.path.join(outputfolder,"normalized_test_big.csv"),index=False)
# Save the figure
plt.savefig(os.path.join(outputfolder,"distribution_big.png"))






