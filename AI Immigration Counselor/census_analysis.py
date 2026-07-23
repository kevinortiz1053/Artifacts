import pandas as pd
succ_criteria_1 = pd.read_excel('succ_criteria_1.xlsx')
succ_criteria_2 = pd.read_excel('succ_criteria_2.xlsx')
succ_criteria_3 = pd.read_excel('succ_criteria_3.xlsx')
succ_criteria_4 = pd.read_excel('succ_criteria_4.xlsx')
succ_criteria_5 = pd.read_excel('succ_criteria_5.xlsx')
succ_criteria_6 = pd.read_excel('succ_criteria_6.xlsx')
succ_criteria_7 = pd.read_excel('succ_criteria_7.xlsx')
succ_criteria_8 = pd.read_excel('succ_criteria_8.xlsx')
succ_criteria_9 = pd.read_excel('succ_criteria_9.xlsx')
succ_criteria_10 = pd.read_excel('succ_criteria_10.xlsx')


# have to split my data into deciles because it was too big for one excel file
# I also couldn't work with the rpy2.robjects dataframe that it returned
whole_set = [succ_criteria_1, succ_criteria_2, succ_criteria_3, succ_criteria_4, succ_criteria_5, succ_criteria_6, succ_criteria_7, succ_criteria_8 ,succ_criteria_9, succ_criteria_10]
ws = [succ_criteria_1 + succ_criteria_2]
prac = [[1,1] + [2,2]]
print(prac)

educ_vals = succ_criteria_1.EDUC.unique()
print(educ_vals)


# input: 'person': dictionary -> { 'characteristic': 'value'}
  #'data_frame': data frame of all people
# returns dictionary with the probability of being in a specified quartile given
# a specified person profile
def person_counter(data_frame, person):
  quartile_predictions = {}
  
  total_count = 0
  for segment in data_frame:
    total_count = total_count + len(segment)
    
  neg_count = 0 # counts how many profiles did not match so we can then take the difference
  count_4q = 0 # counts profiles with income included for fourth qaurtile.
  count_3q = 0 # same as above for third quartile and continuing below
  count_2q = 0
  count_1q = 0
  
  for segment in data_frame:
    for ind in range(0, len(segment)): # to go through each row
      char_count = 0
      for char in person:
        char_count += 1
        if person[char] != segment[char][ind]:
          neg_count += 1
          break # just need one characteristic to not match, don't want to double count one person
        if char_count == len(person):
            if segment['INCTOT'][ind] > 50000: # each if statement checks which quartile the currently matched profile falls into
              count_4q += 1
            if segment['INCTOT'][ind] > 21600 and segment['INCTOT'][ind] <= 50000:
              count_3q += 1
            if segment['INCTOT'][ind] > 6000 and segment['INCTOT'][ind] <= 21600:
              count_2q += 1
            if segment['INCTOT'][ind] <= 6000:
              count_1q += 1
  count_without_inc = total_count - neg_count # count of profile w/o income included

  # Calculation is P(quartile | profile) = count of profile w/ income included / count of profile w/o income included
  quartile_predictions['First quartile'] = count_1q / count_without_inc
  quartile_predictions['Second quartile'] = count_2q / count_without_inc
  quartile_predictions['Third quartile'] = count_3q / count_without_inc
  quartile_predictions['Fourth quartile'] = count_4q / count_without_inc

  return quartile_predictions


per = {
  'EDUC': 11,
  'OCC': 3090,
  'SPEAKENG': 1
}


q_predictions = person_counter(whole_set, per)
print(q_predictions)

import operator

# Below function goes through each characteristic available and finds the most
# common value of each characteristic
# Input: in this case my data is segments of data sets put together in an array
# could change this if my input was just one big data set
def find_most_common_values(data):
  d = {}
  for segment in data:
    for ind in range(0, len(segment)):
      for char in segment:
        if char in d.keys():
          if segment[char][ind] in d[char].keys():
            d[char][segment[char][ind]] += 1
          else:
            d[char][segment[char][ind]] = 1
        else:
          d[char] = {
            segment[char][ind]: 1
          }
  final_d = {}
  for char in d:
    m = max(d[char].items(), key=operator.itemgetter(1))[0]
    final_d[char] = m
  return final_d

most_comm = find_most_common_values([five_yr_tier_1])
most_comm2 = find_most_common_values([five_yr_tier_2])
most_comm3 = find_most_common_values([five_yr_tier_3])
most_comm4 = find_most_common_values([five_yr_tier_4])
most_comm5 = find_most_common_values([five_yr_tier_5])
most_comm6 = find_most_common_values([five_yr_tier_6])
most_comm7 = find_most_common_values([five_yr_tier_7])
most_comm8 = find_most_common_values([five_yr_tier_8])
most_comm9 = find_most_common_values([five_yr_tier_9])
most_comm10 = find_most_common_values([five_yr_tier_10])

print(most_comm)
print(most_comm2)
print(most_comm3)
print(most_comm4)
print(most_comm5)
print(most_comm6)
print(most_comm7)
print(most_comm8)
print(most_comm9)
print(most_comm10)



def find_least_common_values(data):
  d = {}
  for segment in data:
    for ind in range(0, len(segment)):
      for char in segment:
        if char in d.keys():
          if segment[char][ind] in d[char].keys():
            d[char][segment[char][ind]] += 1
          else:
            d[char][segment[char][ind]] = 1
        else:
          d[char] = {
            segment[char][ind]: 1
          }
  final_d = {}
  for char in d:
    m = min(d[char].items(), key=operator.itemgetter(1))[0]
    final_d[char] = m
  return final_d


least_comm1 = find_least_common_values([five_yr_tier_1])
least_comm2 = find_least_common_values([five_yr_tier_2])
least_comm3 = find_least_common_values([five_yr_tier_3])
least_comm4 = find_least_common_values([five_yr_tier_4])
least_comm5 = find_least_common_values([five_yr_tier_5])
least_comm6 = find_least_common_values([five_yr_tier_6])
least_comm7 = find_least_common_values([five_yr_tier_7])
least_comm8 = find_least_common_values([five_yr_tier_8])
least_comm9 = find_least_common_values([five_yr_tier_9])
least_comm10 = find_least_common_values([five_yr_tier_10])




# input: each profile would be a dictionary with different characteristics
  # as keys and their associated values. Person may have a limited number of
  # characteristics, df_person will have all the characteristics
def check_profile_match(person_profile, df_person_profile):
  for char in person_file:
    if person_profile[char] != df_person_profile[char]:
      return 0
  return 1


#returns dictionary: {'characteristic': count of characteristic}
#name of col is the name of one characteristic which can have many
#different values so this functions counts how many of each value there are
def count_col(data_frame, name_of_col):
  col_data = data_frame[name_of_col]
  d = {}
  for row in col_data:
    key = str(row)
    if key in d.keys():
      d[key] = d[key] + 1
    else:
      d[key] = 1
  return d


#return dictionary: {'characteristic': group percentage}
#takes one characteristic and for each value it takes on, divides it by the total count
#giving you the probability of a specific characteristic value
def group_percentage(dict):
  sum = 0
  for key in dict:
      sum = sum + dict[key]
  d = {}
  for key in dict:
    d[key] = dict[key] / sum
  return d


#return dictionary: {'characteristic': sum of percentages by dictionary)
def characteristic_sum_by_group(arr_of_dicts):
  d = {}
  for dict in arr_of_dicts:
    for key in dict:
      if key in d.keys():
        d[key] = d[key] + dict[key]
      else:
        d[key] = dict[key]
  return d


#return dictionary: {'characteristic': given a group, probability of being in that group}
def group_probability(dict_of_group_sums, dict_of_group):
  d = {}
  for key in dict_of_group:
    d[key] = dict_of_group[key] / dict_of_group_sums[key]
  return d


#inputs: final arrays of characteristic probabilities per group, profile of person
#return dictionary: {'specified group': probability of being in that group}
#profile should be a dictionary: {'characteristic': characteristic value}
def group_prediction(characteristic_probabilities_arr, profile):
  d = {
    'top_group': 0,
    'second_group': 0,
    'third_group': 0,
    'bottom_group': 0
  }
  for characteristic in profile:
    characteristic_arr = characteristic_probabilities_arr[characteristic] #characteristic_arr is dictionary of all groups and all characteristic values of a single characteristic
    top_arr = characteristic_arr['top_group'] #top_arr is dictionary of all characteristic values of a specified characteristic and their prob of being in the top group
    second_arr = characteristic_arr['second_group']
    third_arr = characteristic_arr['third_group']
    bottom_arr = characteristic_arr['bottom_group']
    
    #get characteristic value
    c = profile[characteristic]
    c = str(c)
    #sum up percentages for each group, for each characteristic of the profile
    d['top_group'] = d['top_group'] + top_arr[c]
    d['second_group'] = d['second_group'] + second_arr[c]
    d['third_group'] = d['third_group'] + third_arr[c]
    d['bottom_group'] = d['bottom_group'] + bottom_arr[c]
  
  #Each characteristic value gives you a percentage of being in a specific group
  #Here we add up the percentages that each characteristic gives you and divide by
  #the number of them to get an average probability of being in a certain group given
  #x amount of characteristics values
  num_of_characteristics = len(profile)
  d['top_group'] = d['top_group'] / num_of_characteristics # Average probability of this profile landing you in the top group
  d['second_group'] = d['second_group'] / num_of_characteristics
  d['third_group'] = d['third_group'] / num_of_characteristics
  d['bottom_group'] = d['bottom_group'] / num_of_characteristics
  
  return d
    
################################ EDUCATION #################################

top_EDUC_count = count_col(top_quarter, 'EDUC')
second_EDUC_count = count_col(second_quarter, 'EDUC')
third_EDUC_count = count_col(third_quarter, 'EDUC')
bottom_EDUC_count = count_col(bottom_quarter, 'EDUC')

top_EDUC_percentages = group_percentage(top_EDUC_count)
second_EDUC_percentages = group_percentage(second_EDUC_count)
third_EDUC_percentages = group_percentage(third_EDUC_count)
bottom_EDUC_percentages = group_percentage(bottom_EDUC_count)

dict_arr_EDUC = [top_EDUC_percentages, second_EDUC_percentages, third_EDUC_percentages, bottom_EDUC_percentages]

sum_dicts_EDUC = characteristic_sum_by_group(dict_arr_EDUC)
top_EDUC_normalized = group_probability(sum_dicts_EDUC, top_EDUC_percentages)
second_EDUC_normalized = group_probability(sum_dicts_EDUC, second_EDUC_percentages)
third_EDUC_normalized = group_probability(sum_dicts_EDUC, third_EDUC_percentages)
bottom_EDUC_normalized = group_probability(sum_dicts_EDUC, bottom_EDUC_percentages)
final_EDUC_dict = {
  'top_group': top_EDUC_normalized,
  'second_group': second_EDUC_normalized,
  'third_group': third_EDUC_normalized,
  'bottom_group': bottom_EDUC_normalized
  }

####################### OCCUPATION ################################

top_OCC_count = count_col(top_quarter, 'OCC')
second_OCC_count = count_col(second_quarter, 'OCC')
third_OCC_count = count_col(third_quarter, 'OCC')
bottom_OCC_count = count_col(bottom_quarter, 'OCC')

top_OCC_percentages = group_percentage(top_OCC_count)
second_OCC_percentages = group_percentage(second_OCC_count)
third_OCC_percentages = group_percentage(third_OCC_count)
bottom_OCC_percentages = group_percentage(bottom_OCC_count)

dict_arr_OCC = [top_OCC_percentages, second_OCC_percentages, third_OCC_percentages, bottom_OCC_percentages]

sum_dicts_OCC = characteristic_sum_by_group(dict_arr_OCC)
top_OCC_normalized = group_probability(sum_dicts_OCC, top_OCC_percentages)
second_OCC_normalized = group_probability(sum_dicts_OCC, second_OCC_percentages)
third_OCC_normalized = group_probability(sum_dicts_OCC, third_OCC_percentages)
bottom_OCC_normalized = group_probability(sum_dicts_OCC, bottom_OCC_percentages)
final_OCC_dict = {
  'top_group': top_OCC_normalized,
  'second_group': second_OCC_normalized,
  'third_group': third_OCC_normalized,
  'bottom_group': bottom_OCC_normalized
  }
  
############################# SPEAKS ENGLISH ###############################

top_ENG_count = count_col(top_quarter, 'SPEAKENG')
second_ENG_count = count_col(second_quarter, 'SPEAKENG')
third_ENG_count = count_col(third_quarter, 'SPEAKENG')
bottom_ENG_count = count_col(bottom_quarter, 'SPEAKENG')

top_ENG_percentages = group_percentage(top_ENG_count)
second_ENG_percentages = group_percentage(second_ENG_count)
third_ENG_percentages = group_percentage(third_ENG_count)
bottom_ENG_percentages = group_percentage(bottom_ENG_count)

dict_arr_ENG = [top_ENG_percentages, second_ENG_percentages, third_ENG_percentages, bottom_EDUC_percentages]

sum_dicts_ENG = characteristic_sum_by_group(dict_arr_ENG)
top_ENG_normalized = group_probability(sum_dicts_ENG, top_ENG_percentages)
second_ENG_normalized = group_probability(sum_dicts_ENG, second_ENG_percentages)
third_ENG_normalized = group_probability(sum_dicts_ENG, third_ENG_percentages)
bottom_ENG_normalized = group_probability(sum_dicts_ENG, bottom_ENG_percentages)
final_ENG_dict = {
  'top_group': top_ENG_normalized,
  'second_group': second_ENG_normalized,
  'third_group': third_ENG_normalized,
  'bottom_group': bottom_ENG_normalized
  }
  

############################################################################


final_dict = {
  'EDUC': final_EDUC_dict,
  'OCC': final_OCC_dict,
  'SPEAKENG': final_ENG_dict
  }


profile = {
  'EDUC': 6,
  'OCC': 3255
}


yer = group_prediction(final_dict, profile)

print(yer)

common_job = 1
max = 0
for job in top_OCC_count:
  if job != '0':
    if top_OCC_count[job] > max:
      max = top_OCC_count[job]
      common_job = job
  
print(common_job)






