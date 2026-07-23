import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
import random
import pickle
import copy

# ------------------------------data loading---------------------------------

tier_4_episodes = pd.read_excel('tier_4_episodes.xlsx') # Top Tier
tier_3_episodes = pd.read_excel('tier_3_episodes.xlsx')
tier_2_episodes = pd.read_excel('tier_2_episodes.xlsx')
tier_1_episodes = pd.read_excel('tier_1_episodes.xlsx') # Lowest Tier
frames = [tier_1_episodes, tier_2_episodes, tier_3_episodes, tier_4_episodes]
all_episodes = pd.concat(frames)
print(all_episodes.iloc[19])

tier_3_data = pd.read_excel('tier_3_data.xlsx')
tier_2_data = pd.read_excel('tier_2_data.xlsx')

with open("tier_1_4_episodes.pickle", "wb") as f:
  pickle.dump(all_episodes, f)
  
with open("tier_3_data.pickle", "wb") as f:
  pickle.dump(tier_3_data, f)
  
with open("tier_2_data.pickle", "wb") as f:
  pickle.dump(tier_2_data, f)
               # ----------------------------------------------------------
  
with open("tier_1_4_episodes.pickle", "rb") as f:
  all_episodes = pickle.load(f)
  
with open("randomforest_4_v3.pickle", "rb") as f:
    # Load the variable
    model = pickle.load(f)
    
with open("RL_data_set_v2.pickle", "rb") as f:
    # Load the variable
    RL_data_set = pickle.load(f)
    
with open("tier_4_data.pickle", "rb") as f:
  tier_4_data = pickle.load(f)
  
with open("tier_3_data.pickle", "rb") as f:
  tier_3_data = pickle.load(f)
  
with open("tier_2_data.pickle", "rb") as f:
  tier_2_data = pickle.load(f)
    
with open("trained_q_table_RL4_PP_tier4_v6(7).pickle", "rb") as f:
  trained_q_table = pickle.load(f)
  
with open("reward_tracker_RL4_PP_mixed_v2(7).pickle", "rb") as f:
  reward_tracker = pickle.load(f)
  
with open("reward_every_5_MA_v3.pickle", "rb") as f:
  reward_every_5 = pickle.load(f)
  
with open("reward_every_other_5_MA_v3.pickle", "rb") as f:
  reward_every_other_5 = pickle.load(f)
  


# --------------------------------------------------------------------------


def get_all_available_actions(profile1, whole_data_set):
  # returns list of available actions: ['["char", new value]', '["char", new value]'...]
  # This method requires that the states all come in the same order, index below
  # profile = ("STATEFIP", "METRO", "OWNERSHPD", "MORTGAGE", "NCHILD", "SEX", "AGE", "MARST", "RACE", "HISPAN", "BPL", "CITIZEN", "YRSUSA1", "LANGUAGE", "SPEAKENG", "SCHOOL", "EDUC", "EMPSTAT", "OCC", "IND", "UHRSWORK", "POVERTY", "TRANWORK", "CARPOOL")
  # char_indexes = {"STATEFIP": 0, "METRO": 1, "OWNERSHPD": 2, "MORTGAGE": 3, "NCHILD": 4, "SEX": 5, "AGE": 6, "MARST": 7, "RACE": 8, "HISPAN": 9, "BPL": 10, "CITIZEN": 11, "YRSUSA1": 12, "LANGUAGE": 13, "SPEAKENG": 14, "SCHOOL": 15, "EDUC": 16, "EMPSTAT": 17, "OCC": 18, "IND": 19, "UHRSWORK": 20, "POVERTY": 21, "TRANWORK": 22, "CARPOOL": 23}
  
  actions = []
  profile2 = profile1.copy()
  
  #---------characterisitcs that can be incremented several different ways-----------
  
  # incrementing STATEFIP, keeping all, I'm interested in the movement of people
  state_fip_values = np.unique(whole_data_set["STATEFIP"])
  current_state_fip = profile2[0]
  for v in state_fip_values:
    if v != current_state_fip:
      action = ['STATEFIP', v]
      actions.append(str(action)) # make the action into a string so it can be a dictionary key
  
  #incrementing METRO, keeping all values
  # metro_values = np.unique(whole_data_set["METRO"])
  metro_values = [0, 1, 2, 3, 4]
  current_metro = profile2[1]
  for v in metro_values:
    if v != current_metro:
      action = ['METRO', v]
      actions.append(str(action))
      
  #incrementing OWNERSHPD, further constrained
  # ownershpd_values = np.unique(whole_data_set["OWNERSHPD"])
  ownershpd_values = [12, 13, 22]
  current_ownershpd = profile2[2]
  for v in ownershpd_values:
    if v != current_ownershpd:
      action = ['OWNERSHPD', v]
      actions.append(str(action))
      
   #incrementing MORTGAGE, costrained even further
  # mortgage_values = np.unique(whole_data_set["MORTGAGE"])
  mortgage_values = [1, 3]
  current_mortgage = profile2[3]
  for v in mortgage_values:
    if v != current_mortgage:
      action = ['MORTGAGE', v]
      actions.append(str(action))
      
    #incrementing MARST, keeping all
  #marst_values = np.unique(whole_data_set["MARST"])
  marst_values = [1, 3, 4, 6]
  current_marst = profile2[3]
  for v in marst_values:
    if v != current_marst:
      action = ['MARST', v]
      actions.append(str(action))
      
    #incrementing LANGUAGE. 99 options. Constrained it to the most popular ones
    # from my personal perspective
  # language_values = np.unique(whole_data_set["LANGUAGE"])
  language_values = [1, 2, 10, 11, 12, 18, 31, 43, 48, 49, 57]
  current_language = profile2[13]
  for v in language_values:
    if v != current_language:
      action = ['LANGUAGE', v]
      actions.append(str(action))
      
    #incrementing SPEAKENG, further constrained
  # speakeng_values = np.unique(whole_data_set["SPEAKENG"])
  speakeng_values = [1, 4, 6]
  current_speakeng = profile2[14]
  for v in speakeng_values:
    if v != current_speakeng:
      action = ['SPEAKENG', v]
      actions.append(str(action))
      
  # incrementing UHRSWORK. This has 0 - 99 options...Keeping 0, 35 - 45, 60 - 80
  # To decrease my grid world, picked those values because they have they 
  # appeared the most in the data set
  # uhrswork_values = np.unique(whole_data_set["UHRSWORK"])
  #uhrswork_values = [0, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
  uhrswork_values = [10, 20, 30, 40, 50, 60, 70, 80]
  #current_uhrswork = profile2[20]
  current_uhrswork = profile2[19]
  for v in uhrswork_values:
    if v != current_uhrswork:
      action = ['UHRSWORK', v]
      actions.append(str(action))
      
   # incrementing TRANWORK. Got rid of this because it was mostly car and had
   # a lot of different options. The increase in the grid didn't balance with the
   # information that could be attained/learned
      
  # incrementing CARPOOL. Keeping all since there's only a few options and I'm
  # interested if it would have any effect
  # carpool_values = np.unique(whole_data_set["CARPOOL"])
  carpool_values = [1, 2, 3, 4, 5]
  #current_carpool = profile2[23]
  current_carpool = profile2[21]
  for v in carpool_values:
    if v != current_carpool:
      action = ['CARPOOL', v]
      actions.append(str(action))
      
      
      
#---------characterisitcs that can be incremented only in a certain direction-----------
      
      
  # incrementing NCHILD. Got rid of this because it's a touchy subject to give
  # advice on
      
  # incrementing AGE, was thinking between keeping this and YRSUSA1. Chose YRAUSA
  # because someone could've immigrated at any age. More interested in how the years
  # in the US increase their chances of success
      
   # incrementing CITIZEN
  current_citizen = profile2[11]
  if current_citizen != 2:
    action = ['CITIZEN', 2]
    actions.append(str(action))
    
    # incrementing YRSUSA1
  #yrsusa1_values = np.unique(whole_data_set["YRSUSA1"])
  #current_yrsusa1 = profile2[6]
  #for v in yrsusa1_values:
   # if v == current_yrsusa1 + 1 or v == current_yrsusa1 + 2:
    #  action = ['YRSUSA1', v]
     # actions.append(str(action))
      
    # incrementing SCHOOL
  school_values = [1, 2]
  current_school = profile2[15]
  for v in school_values:
    if v != current_school:
      action = ['SCHOOL', v]
      actions.append(str(action))

   # incrementing EDUC
  educ_values = np.unique(whole_data_set["EDUC"])
  current_educ = profile2[16]
  for v in educ_values:
    if v == current_educ + 1 or v == current_educ + 2:
      action = ['EDUC', v]
      actions.append(str(action))
      
  # incrementing EMPSTAT
  current_empstat = profile2[17]
  if current_empstat != 1:
    action = ['EMPSTAT', 1]
    actions.append(str(action))
      
  # Increment OCC, if you know the occupation, you know the industry
  #occ_values = np.unique(whole_data_set["OCC"])
  occ_values = [0, 960, 1980, 2970, 3550, 4655, 5940, 7640, 9760, 9830, 9920]
  current_occ = profile2[18]
  for v in occ_values:
    if v != current_occ:
      action = ['OCC', v]
      actions.append(str(action))
    
  return actions
#  for v in occ_values:
 #   if current_occ >= 0 and current_occ <= 960 and v != current_occ: # Management, Business, and Finance
  #    action = ['OCC', v]
   #   actions.append(str(action))
    #if current_occ >= 1005 and current_occ <= 1980 and v != current_occ: # Computer, Engineering, and Science
     # action = ['OCC', v]
      #actions.append(str(action))
#    if current_occ >= 2001 and current_occ <= 2970 and v != current_occ: # Education, Legal, Community Service, Arts, and Media
 #     action = ['OCC', v]
  #    actions.append(str(action))
   # if current_occ >= 3000 and current_occ <= 3550 and v != current_occ: # Healthcare Practitioners and Technical
    #  action = ['OCC', v]
     # actions.append(str(action))
#    if current_occ >= 3601 and current_occ <= 4655 and v != current_occ: # Service Occupations
 #     action = ['OCC', v]
  #    actions.append(str(action))
   # if current_occ >= 4700 and current_occ <= 5940 and v != current_occ: # Sales and Office
    #  action = ['OCC', v]
     # actions.append(str(action))
#    if current_occ >= 6005 and current_occ <= 7640 and v != current_occ: # Natural Resources, Construction, and Maintenance
 #     action = ['OCC', v]
  #    actions.append(str(action))
   # if current_occ >= 7700 and current_occ <= 9760 and v != current_occ: # Production, Transportation, and Material Moving
    #  action = ['OCC', v]
     # actions.append(str(action))
#    if current_occ >= 9800 and current_occ <= 9830 and v != current_occ: # Military Specific
 #     action = ['OCC', v]
  #    actions.append(str(action))
   # if current_occ == 9920 and v != current_occ: # Unemployed
    #  action = ['OCC', v]
     # actions.append(str(action))
    
    
  # Increment IND, another folly of including this is that it could choose an
  # industry and occupation that don't coincide. Presumably all occupations
  # should coincide with their proper industry
  
 # return actions


def check_list(list):
  return len(set(list)) == 1

#def get_max_action(q_table, state):
  profile = str(state)
  #-------- checking if all the values are the same so we can pick a random value in this case--------
  action_values = []
  for action in list(q_table[profile].keys()):
    action_values.append(q_table[profile][action])
  c = check_list(action_values)
  if c == True:
    r = np.random.randint(len(q_table[profile].keys()))
    count = 0
    for action in list(q_table[profile].keys()):
      if count == r:
        print('all q values were the same')
        return action
      else:
        count += 1
  #---------------------------------------------------------------
  max1 = -1000
  max_action = None
  for action in list(q_table[profile].keys()):
    if q_table[profile][action] > max1:
      max1 = q_table[profile][action]
      max_action = action
  print('picking the actual highest action')
  return max_action


def get_max_action(q_table, state): # get_max_action_v2
  profile = str(state)
  max_value = max(q_table[profile].values())
  max_actions = [key for key, value in q_table[profile].items() if value == max_value]
  if len(max_actions) > 1:
    random_action = random.choice(max_actions)
    print("Multiple actions have the same maximum value. Randomly selected action:", random_action)
    return random_action
  else:
    max_action = max_actions[0]
    print("Action with the maximum value:", max_action)
  return max_action

# Define the action selection strategy
def select_action(state, epsilon, q_table5, actions_taken):
  # returns action = '["char", new_value]'
  profile = str(state)
  available_actions = list(q_table5[profile].keys())
  if np.random.rand() < epsilon:
    int1 = np.random.randint(len(available_actions)) # Explore
    print('Explore')
    if str(available_actions[int1]) in actions_taken:
      print('Explore Actions Taken: ', actions_taken)
      # q_copy = q_table5.deepcopy()
      q_copy = copy.deepcopy(q_table5)
      del q_copy[profile][str(available_actions[int1])]
      a = select_action(state, epsilon, q_copy, actions_taken)
      print('Explore** random action selected: ', a)
      return a
    else:
      print('Explore random action selected: ', str(available_actions[int1]))
      return str(available_actions[int1])
  else:
    print('Exploit ya heard')
    max_action = get_max_action(q_table5, state)
    if max_action in actions_taken:
      print('Exploit Actions Taken: ', actions_taken)
      # q_copy = q_table5.deepcopy()
      q_copy = copy.deepcopy(q_table5)
      del q_copy[profile][max_action]
      a = select_action(state, epsilon, q_copy, actions_taken)
      return a
    else:
      return max_action  # Exploit # may need to modify argmax to choose the maximal (state, action) pair, depends how my q-table is structured


#def select_action_v2(state, epsilon, q_table):
  # returns action = '["char", new_value]'
  profile = str(state)
  available_actions = list(q_table[profile].keys())
  action_list = []
  for k in range(0, 3):
    if np.random.rand() < epsilon:
      int1 = np.random.randint(len(available_actions)) # Explore
      max_action = str(available_actions[int1])
      action_list.append(max_action)
    else:
      max_action = get_max_action(q_table, state)
      action_list.append(max_action)  # Exploit # may need to modify argmax to choose the maximal (state, action) pair, depends how my q-table is structured
    if max_action in available_actions:
      available_actions.remove(max_action)
    
  return action_list


def take_action(state, action1):
  # This method requires that the states all come in the same order, index below
  state1 = state.copy()
  #char_indexes = {"STATEFIP": 0, "METRO": 1, "OWNERSHPD": 2, "MORTGAGE": 3, "NCHILD": 4, "SEX": 5, "AGE": 6, "MARST": 7, "RACE": 8, "HISPAN": 9, "BPL": 10, "CITIZEN": 11, "YRSUSA1": 12, "LANGUAGE": 13, "SPEAKENG": 14, "SCHOOL": 15, "EDUC": 16, "EMPSTAT": 17, "OCC": 18, "IND": 19, "UHRSWORK": 20, "POVERTY": 21, "TRANWORK": 22, "CARPOOL": 23}
  char_indexes = {"STATEFIP": 0, "METRO": 1, "OWNERSHPD": 2, "MORTGAGE": 3, "NCHILD": 4, "SEX": 5, "AGE": 6, "MARST": 7, "RACE": 8, "HISPAN": 9, "BPL": 10, "CITIZEN": 11, "YRSUSA1": 12, "LANGUAGE": 13, "SPEAKENG": 14, "SCHOOL": 15, "EDUC": 16, "EMPSTAT": 17, "OCC": 18, "UHRSWORK": 19, "TRANWORK": 20, "CARPOOL": 21}
  action1 = eval(action1)
  char = action1[0]
  new_val = action1[1]
  char_ind = char_indexes[char]
  state1[char_ind] = new_val # This is the next state
  state1[6] = state1[6] + 1 # Incrementing AGE - index: 6
  state1[12] = state1[12] + 1 # Incrementing YRSUSA1 - index: 12
  return state1 # Returns an array


#def take_action_v2(state, action_list1, state_tier, data_set):
  # This method requires that the states all come in the same order, index below
  state1 = state.copy()
  #char_indexes = {"STATEFIP": 0, "METRO": 1, "OWNERSHPD": 2, "MORTGAGE": 3, "NCHILD": 4, "SEX": 5, "AGE": 6, "MARST": 7, "RACE": 8, "HISPAN": 9, "BPL": 10, "CITIZEN": 11, "YRSUSA1": 12, "LANGUAGE": 13, "SPEAKENG": 14, "SCHOOL": 15, "EDUC": 16, "EMPSTAT": 17, "OCC": 18, "IND": 19, "UHRSWORK": 20, "POVERTY": 21, "TRANWORK": 22, "CARPOOL": 23}
  char_indexes = {"STATEFIP": 0, "METRO": 1, "OWNERSHPD": 2, "MORTGAGE": 3, "NCHILD": 4, "SEX": 5, "AGE": 6, "MARST": 7, "RACE": 8, "HISPAN": 9, "BPL": 10, "CITIZEN": 11, "YRSUSA1": 12, "LANGUAGE": 13, "SPEAKENG": 14, "SCHOOL": 15, "EDUC": 16, "EMPSTAT": 17, "OCC": 18, "UHRSWORK": 19, "TRANWORK": 20, "CARPOOL": 21}
  max_prob = 0
  
  action1 = eval(action_list1[0])
  char = action1[0]
  new_val = action1[1]
  char_ind = char_indexes[char]
  state1[char_ind] = new_val # This is the next state
  max_state = state1.copy()
  
  max_action = str(action_list1[0])
  for action1 in action_list1:
    action1 = eval(action1)
    char = action1[0]
    new_val = action1[1]
    char_ind = char_indexes[char]
    state1[char_ind] = new_val # This is the next state
    if state_tier == 'Tier 1':
      tiered_data2 = char_counter_2
    elif state_tier == 'Tier 2':
      tiered_data2 = char_counter_3
    else:
      tiered_data2 = char_counter_4
    prob = get_state_tiered_prob(state1, data_set, tiered_data2)
    if prob > max_prob:
      max_state = state1.copy()
      max_action = str(action1)
      max_prob = prob
  return max_state, max_action # Returns an array


# Define the update function for Q-values
def update_q_table(q_table3, state4, action, reward, next_state4, learning_rate, discount_factor):
  max_action_next_state = get_max_action(q_table3, next_state4)
  state4 = str(state4)
  next_state4 = str(next_state4)
  q_table3[state4][action] += learning_rate * (reward + discount_factor * q_table3[next_state4][max_action_next_state] - q_table3[state4][action])
  return q_table3


def q_table_check(q_table, state, whole_data_set): # generates/checks if the agent has seen this state before and what actions it can take in it
  profile = str(state)
  if profile not in q_table.keys():
    q_table[profile] = {}
    available_actions = get_all_available_actions(state, whole_data_set)
    for action in available_actions:
      q_table[profile][action] = 0
  # q_table = {'[state]':{'['char', new value]': value}}
  return q_table


def convert_state_for_model(state):
  d = {}
  #characteristics = ["STATEFIP", "METRO", "OWNERSHPD", "MORTGAGE", "NCHILD", "SEX", "AGE", "MARST", "RACE", "HISPAN", "BPL", "CITIZEN", "YRSUSA1", "LANGUAGE", "SPEAKENG", "SCHOOL", "EDUC", "EMPSTAT", "OCC", "IND", "UHRSWORK", "POVERTY", "TRANWORK", "CARPOOL"]
  characteristics = ["STATEFIP", "METRO", "OWNERSHPD", "MORTGAGE", "NCHILD", "SEX", "AGE", "MARST", "RACE", "HISPAN", "BPL", "CITIZEN", "YRSUSA1", "LANGUAGE", "SPEAKENG", "SCHOOL", "EDUC", "EMPSTAT", "OCC", "UHRSWORK", "TRANWORK", "CARPOOL"]
  for ind in range(len(state)):
    d[characteristics[ind]] = [state[ind]]
  final_d = pd.DataFrame(d)
  return final_d


#def get_income_tier(model, state):
  # model should be model that has already been fit to training data
  state1 = state.copy()
  state1 = convert_state_for_model(state1)
  #print("Model is predicting")
  income_tier = model.predict(state1)
  random_number = np.random.rand() # rand number to take into account our
  # predictive values we calculated for the income tier prediction
  if income_tier == 'Tier 1':
    if random_number <= 0.74:
      return 'Tier 1'
    elif random_number <= 0.91:
      return 'Tier 2'
    elif random_number <= 0.98:
      return 'Tier 3'
    else:
      return 'Tier 4'
    
  if income_tier == 'Tier 2':
    if random_number <= 0.16:
      return 'Tier 1'
    elif random_number <= 0.79:
      return 'Tier 2'
    elif random_number <= 0.97:
      return 'Tier 3'
    else:
      return 'Tier 4'
    
  if income_tier == 'Tier 3':
    if random_number <= 0.03:
      return 'Tier 1'
    elif random_number <= 0.14:
      return 'Tier 2'
    elif random_number <= 0.83:
      return 'Tier 3'
    else:
      return 'Tier 4'
    
  if income_tier == 'Tier 4':
    if random_number <= 0.01:
      return 'Tier 1'
    elif random_number <= 0.02:
      return 'Tier 2'
    elif random_number <= 0.08:
      return 'Tier 3'
    else:
      return 'Tier 4'
    

def get_income_tier(model, state):
  # model should be model that has already been fit to training data
  state1 = state.copy()
  state1 = convert_state_for_model(state1)
  #print("Model is predicting")
  income_tier = model.predict(state1)
  random_number = np.random.rand() # rand number to take into account our
  # predictive values we calculated for the income tier prediction
  if income_tier == 'Tier 1':
    if random_number <= 0.87:
      return 'Tier 1'
    elif random_number <= 0.97:
      return 'Tier 2'
    elif random_number <= 0.99:
      return 'Tier 3'
    else:
      return 'Tier 4'
    
  if income_tier == 'Tier 2':
    if random_number <= 0.21:
      return 'Tier 1'
    elif random_number <= 0.80:
      return 'Tier 2'
    elif random_number <= 0.93:
      return 'Tier 3'
    else:
      return 'Tier 4'
    
  if income_tier == 'Tier 3':
    if random_number <= 0.05:
      return 'Tier 1'
    elif random_number <= 0.24:
      return 'Tier 2'
    elif random_number <= 0.84:
      return 'Tier 3'
    else:
      return 'Tier 4'
    
  if income_tier == 'Tier 4':
    if random_number <= 0.04:
      return 'Tier 1'
    elif random_number <= 0.15:
      return 'Tier 2'
    elif random_number <= 0.31:
      return 'Tier 3'
    else:
      return 'Tier 4'
    
    
def get_char_name(char_ind):
  if char_ind == 0:
    return 'STATEFIP'
  if char_ind == 1:
    return 'METRO'
  if char_ind == 2:
    return 'OWNERSHPD'
  if char_ind == 3:
    return 'MORTGAGE'
  if char_ind == 4:
    return 'NCHILD'
  if char_ind == 5:
    return 'SEX'
  if char_ind == 6:
    return 'AGE'
  if char_ind == 7:
    return 'MARST'
  if char_ind == 8:
    return 'RACE'
  if char_ind == 9:
    return 'HISPAN'
  if char_ind == 10:
    return 'BPL'
  if char_ind == 11:
    return 'CITIZEN'
  if char_ind == 12:
    return 'YRSUSA1'
  if char_ind == 13:
    return 'LANGUAGE'
  if char_ind == 14:
    return 'SPEAKENG'
  if char_ind == 15:
    return 'SCHOOL'
  if char_ind == 16:
    return 'EDUC'
  if char_ind == 17:
    return 'EMPSTAT'
  if char_ind == 18:
    return 'OCC'
  #if char_ind == 19:
  #  return 'IND'
  if char_ind == 19:
    return 'UHRSWORK'
  #if char_ind == 20:
  #  return 'UHRSWORK'
  #if char_ind == 21:
  #  return 'POVERTY'
  #if char_ind == 22:
  #  return 'TRANWORK'
  if char_ind == 20:
    return 'TRANWORK'
  #if char_ind == 23:
  #  return 'CARPOOL'
  if char_ind == 21:
    return 'CARPOOL'
  

def get_char_count_dicts(whole_data_set2, tiered_data_set4, tiered_data_set3, tiered_data_set2):
  #characteristics = ["STATEFIP", "METRO", "OWNERSHPD", "MORTGAGE", "NCHILD", "SEX", "AGE", "MARST", "RACE", "HISPAN", "BPL", "CITIZEN", "YRSUSA1", "LANGUAGE", "SPEAKENG", "SCHOOL", "EDUC", "EMPSTAT", "OCC", "IND", "UHRSWORK", "POVERTY", "TRANWORK", "CARPOOL"]
  characteristics = ["STATEFIP", "METRO", "OWNERSHPD", "MORTGAGE", "NCHILD", "SEX", "AGE", "MARST", "RACE", "HISPAN", "BPL", "CITIZEN", "YRSUSA1", "LANGUAGE", "SPEAKENG", "SCHOOL", "EDUC", "EMPSTAT", "OCC", "UHRSWORK", "TRANWORK", "CARPOOL"]
  char_dict_w = {}
  char_dict_4 = {}
  char_dict_3 = {}
  char_dict_2 = {}
  for char in characteristics:
    char_dict_w[char] = {}
    char_dict_4[char] = {}
    char_dict_3[char] = {}
    char_dict_2[char] = {}
    for val in np.unique(whole_data_set2[char]):
      char_dict_w[char][val] = 0
      char_dict_4[char][val] = 0
      char_dict_3[char][val] = 0
      char_dict_2[char][val] = 0
      
  for char in characteristics:
    for val in whole_data_set2[char]:
      char_dict_w[char][val]  = char_dict_w[char][val] + 1
    
    for val in tiered_data_set4[char]:
      char_dict_4[char][val] = char_dict_4[char][val] + 1

    for val in tiered_data_set3[char]:
      char_dict_3[char][val] = char_dict_3[char][val] + 1
      
    for val in tiered_data_set2[char]:
      char_dict_2[char][val] = char_dict_2[char][val] + 1
        
  return char_dict_w, char_dict_4, char_dict_3, char_dict_2


def convert_tier_to_number(tier):
  if tier == 'Tier 1':
    return 1
  if tier == 'Tier 2':
    return 2
  if tier == 'Tier 3':
    return 3
  if tier == 'Tier 4':
    return 4


char_counter_whole, char_counter_4, char_counter_3, char_counter_2 = get_char_count_dicts(RL_data_set, tier_4_data, tier_3_data, tier_2_data)

print(len(tier_2_data))

def char_counter_t(char2, char_value, tiered_data_count1):
  # data_set should only be one column worth of data in array format
  if char_value in tiered_data_count1[char2].keys():
    counter = tiered_data_count1[char2][char_value]
  else:
    counter = 0
  return counter  

#def char_counter_t(col_data, char_value):
#  counter3 = 0
#  for val in col_data:
#    if val == char_value:
#      counter3 += 1
#      
#  return counter3
  

def char_counter_w(char2, char_value):
  # data_set should only be one column worth of data in array format
  if char_value in char_counter_whole[char2].keys():
    counter = char_counter_whole[char2][char_value]
  else:
    counter = 0
  return counter
    
    
def get_state_tiered_prob(state, whole_data_set, tiered_data_count_dict):
  # state = [1, 1, 2, 3..., 6]
  # profile = ("STATEFIP", "METRO", "OWNERSHPD", "MORTGAGE", "NCHILD", "SEX", "AGE", "MARST", "RACE", "HISPAN", "BPL", "CITIZEN", "YRSUSA1", "LANGUAGE", "SPEAKENG", "SCHOOL", "EDUC", "EMPSTAT", "OCC", "IND", "UHRSWORK", "POVERTY", "TRANWORK", "CARPOOL")
  char_probabilities = []
  for char_ind in range(0, len(state)):
    char = get_char_name(char_ind)
    #tiered_col_data = list(tiered_data[char])
    #whole_col_data = whole_data_set[char]
    tiered_count = char_counter_t(char, state[char_ind], tiered_data_count_dict)
    whole_count = char_counter_w(char, state[char_ind])
    if whole_count == 0:
      char_prob = 0
    else:
      char_prob = tiered_count / whole_count
    char_probabilities.append(char_prob)
  state_prob = round(stat.mean(char_probabilities), 3)
  return state_prob


#def calc_reward_points(tier_next_state, tier_state):
  bonus = 10
  if tier_next_state == 'Tier 1':
    reward = 5
  if tier_next_state == 'Tier 2':
    reward = 4
    if tier_state == 'Tier 1':
      reward = reward + bonus
  if tier_next_state == 'Tier 3':
    reward = 3
    if tier_state == 'Tier 2' or tier_state == 'Tier 1':
      reward = reward + bonus
  if tier_next_state == 'Tier 4':
    reward = 2
    if tier_state == 'Tier 3' or tier_state == 'Tier 2' or tier_state == 'Tier 1':
      reward = reward + bonus
    
  return reward


def calc_reward_points_v2(tier_next_state, tier_state):
  bonus = 500
  if tier_next_state == 'Tier 1':
    reward = 50
  if tier_next_state == 'Tier 2':
    reward = 100
    if tier_state == 'Tier 1':
      reward = reward + bonus
  if tier_next_state == 'Tier 3':
    reward = 150
    if tier_state == 'Tier 2' or tier_state == 'Tier 1':
      reward = reward + bonus
  if tier_next_state == 'Tier 4':
    reward = 200
    if tier_state == 'Tier 3' or tier_state == 'Tier 2' or tier_state == 'Tier 1':
      reward = reward + bonus
    
  return reward
    
    
def get_pos_or_neg(state, next_state, whole_data_set, tiered_data_count_dict1):
  # Assuming we would only check the probabilities of both states being in the
  # top tier. Don't see why we would check anything else right now
  state_prob = get_state_tiered_prob(state, whole_data_set, tiered_data_count_dict1)
  next_state_prob = get_state_tiered_prob(next_state, whole_data_set, tiered_data_count_dict1)
  print('State Probability: ', state_prob)
  print('Next State Probability: ', next_state_prob)
  if next_state_prob > state_prob:
    return 1 # postive
  elif next_state_prob < state_prob:
    return 0 # negative
  else: # when next_state_prob == state_prob
    return 2 # 0 reward since you didn't get better or worse
  
  
def get_pos_or_neg_v2(state2, next_state2, tier_next_state2):
  state2_copy = convert_state_for_model(state2)
  next_state2_copy = convert_state_for_model(next_state2)
  pred_prob_state = model.predict_proba(state2_copy)
  #print('pred prob state: ', pred_prob_state)
  pred_prob_next_state = model.predict_proba(next_state2_copy)
  # state_prob = get_state_tiered_prob(state, whole_data_set, tiered_data_count_dict1)
  # next_state_prob = get_state_tiered_prob(next_state, whole_data_set, tiered_data_count_dict1)
  #if tier_state2 == 'Tier 1':
   # state_prob = pred_prob_state[0][1]
    #next_state_prob = pred_prob_next_state[0][1]
  #elif tier_state2 == 'Tier 2':
   # state_prob = pred_prob_state[0][2]
    #next_state_prob = pred_prob_next_state[0][2]  
  if tier_next_state2 == 'Tier 1' or tier_next_state2 == 'Tier 2':
    state_prob = pred_prob_state[0][2]
    next_state_prob = pred_prob_next_state[0][2]   
  else:
    state_prob = pred_prob_state[0][3]
    next_state_prob = pred_prob_next_state[0][3]
    
  if next_state_prob > state_prob:
    return 1 # postive
  elif next_state_prob < state_prob:
    return 0 # negative
  else: # when next_state_prob == state_prob
    return 2 # 0 reward since you didn't get better or worse
    

def get_reward(state, next_state, whole_data_set, tier_next_state, tier_state, tier_4_data):
  #if tier_next_state == 'Tier 1':
   # tiered_data2 = char_counter_2
  #elif tier_next_state == 'Tier 2':
   # tiered_data2 = char_counter_3
  #else:
    #tiered_data2 = char_counter_4
    
  if tier_next_state == 'Tier 1' or tier_next_state == 'Tier 2':
    tiered_data2 = char_counter_3
  else:
    tiered_data2 = char_counter_4
    
  #pos_or_neg = get_pos_or_neg(state, next_state, whole_data_set, tiered_data2)
  pos_or_neg = get_pos_or_neg_v2(state, next_state, tier_next_state)
  if pos_or_neg == 1:
    reward = calc_reward_points_v2(tier_next_state, tier_state)
    return reward
  if pos_or_neg == 0:
    s = convert_tier_to_number(tier_state)
    next_s = convert_tier_to_number(tier_next_state)
    if next_s < s:
      return -500 # negates the bonus
    else:
      return -250 # when your increment made it worse but didn't lower the tier
  else:
    return 1 # When your increment had no change. Changed from 0
  

def del_statefip(q_table3, state3):
  profile = str(state3)
  q_table3_copy = copy.deepcopy(q_table3)
  for k in list(q_table3[profile]):
    if 'STATE' in k:
      del q_table3_copy[profile][k]
      
  return q_table3_copy


def train_agent(starting_state1, learning_rate, discount_factor, epsilon, q_table, whole_data_set, model, tier_4_data, reward_tracker1):
  # Could train the agent with an existing q-table. So the agent has "prior knowledge".
  # Or the q-table could be empty, a new one.
  state_counter = 1
  statefip_counter = 0
  print("First State: ", starting_state1)
  state = starting_state1.copy()
  acc_reward = 0
  actions_taken1 = []
  income_tier_state = get_income_tier(model, state)
  for z in range(0, 20):
    print('train_agent, on state number: ', state_counter, ' , move number: ', z + 1)
    q_table = q_table_check(q_table, state, whole_data_set)
    action = select_action(state, epsilon, q_table, actions_taken1) # returns characteristic to increment and it's new value: ['char', new value]
      # print('train_agent, state is: ' + str(state))
    #if 'STATE' in action:
     # statefip_counter += 1
    #if statefip_counter > 5 and 'STATE' in action:
     # q_copy1 = del_statefip(q_table, state)
      #action = select_action(state, epsilon, q_copy1, actions_taken1)
      #print('chose STATE action more than 5 times')
    next_state = take_action(state, action) # returns new state with the new char value from above
    actions_taken1.append(action)
      # print('train_agent, next_state is: ' + str(next_state))
    #income_tier_state = get_income_tier(model, state)
    #next_state, action = take_action_v2(state, action, income_tier_state, whole_data_set) # returns new state with the new char value from above
    income_tier_next_state = get_income_tier(model, next_state)
    reward = get_reward(state, next_state, whole_data_set, income_tier_next_state, income_tier_state, tier_4_data)
    print('State Tier: ', income_tier_state)
    print('State Next Tier: ', income_tier_next_state)
    print('Current State: ', state)
    print('train_agent, action is: ' + action)
    print('Reward: ', reward)
    acc_reward += reward
    q_table = q_table_check(q_table, next_state, whole_data_set)
    q_table = update_q_table(q_table, state, action, reward, next_state, learning_rate, discount_factor)
    # state_copy = state.copy()
    print('Q_table value: ', q_table[str(state)][action])
    state = next_state
    income_tier_state = income_tier_next_state
    print(' ')
  
  print("Last State: ", state)  
  reward_tracker1[1].append(acc_reward)
  reward_tracker1[0].append(reward_tracker1[0][len(reward_tracker1[0]) - 1] + 1)
  print(actions_taken1)
      
  return q_table, reward_tracker1, state


#q_table3[state][action] += learning_rate * (reward + discount_factor * q_table3[next_state][max_action_next_state] - q_table3[state][action])
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 0.2 # want to explore in the beginning, may want to make this smaller as time goes on
EPSILON = 0
epsilon = 0.2
epsilon = 0
q_table1 = {}


uhrswork_trans = pd.cut(all_episodes["UHRSWORK"], bins=[-9999999, 10, 20, 30, 40, 50, 60, 70, 80, 90, 1000], labels=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
all_episodes["UHRSWORK"] = uhrswork_trans
occ_trans = pd.cut(all_episodes["OCC"], bins=[-999999, 1, 961, 1981, 2971, 3551, 4656, 5941, 7641, 9761, 9831, 9921], labels=[0, 960, 1980, 2970, 3550, 4655, 5940, 7640, 9760, 9830, 9920])
all_episodes["OCC"] = occ_trans
characteristics1 = ["STATEFIP", "METRO", "OWNERSHPD", "MORTGAGE", "NCHILD", "SEX", "AGE", "MARST", "RACE", "HISPAN", "BPL", "CITIZEN", "YRSUSA1", "LANGUAGE", "SPEAKENG", "SCHOOL", "EDUC", "EMPSTAT", "OCC", "UHRSWORK", "TRANWORK", "CARPOOL"]
all_episodes = all_episodes[characteristics1]  


print(actions_taken1)
# Train the agent
reward_tracker = [[0], [0]] # Original reward_tracker
reward_every_other_5 = [[0], [0]] # Original reward_tracker
for i in range(15, 20):
  starting_state = list(all_episodes.iloc[i])
  for blah in range(0, 10):
    trained_q_table9, reward_tracker, last_state = train_agent(starting_state, LEARNING_RATE, DISCOUNT_FACTOR, EPSILON, trained_q_table, RL_data_set, model, tier_4_data, reward_tracker)
    print("=============================================EPISODE NUMBER IS: ", blah)


starting_state = list(all_episodes.iloc[0])
print(len(reward_tracker[1]))

#trained_q_table = {}
#last_states = []
for blah in range(0, 10):
  for i in range(0, len(all_episodes)):
    starting_state = list(all_episodes.iloc[i])
    trained_q_table9, reward_tracker, last_state = train_agent(starting_state, LEARNING_RATE, DISCOUNT_FACTOR, EPSILON, trained_q_table, RL_data_set, model, tier_4_data, reward_tracker)
  #last_states.append(last_state)
 
m = sum(reward_tracker[1]) / (len(reward_tracker[1]) - 1)
print(m)
 
last_state_counts = last_state_analysis(last_states)
print(last_state_counts)


def last_state_analysis(last_states_array):
  # ditionary is {"char": {"char_value": count}}
  final_dict = {"STATEFIP":{}, "METRO":{}, "OWNERSHPD":{}, "MORTGAGE":{}, "NCHILD":{}, "SEX":{}, "AGE":{}, "MARST":{}, "RACE":{}, "HISPAN":{}, "BPL":{}, "CITIZEN":{}, "YRSUSA1":{}, "LANGUAGE":{}, "SPEAKENG":{}, "SCHOOL":{}, "EDUC":{}, "EMPSTAT":{}, "OCC":{}, "UHRSWORK":{}, "TRANWORK":{}, "CARPOOL":{}}
  characteristics = ["STATEFIP", "METRO", "OWNERSHPD", "MORTGAGE", "NCHILD", "SEX", "AGE", "MARST", "RACE", "HISPAN", "BPL", "CITIZEN", "YRSUSA1", "LANGUAGE", "SPEAKENG", "SCHOOL", "EDUC", "EMPSTAT", "OCC", "UHRSWORK", "TRANWORK", "CARPOOL"]
  ind = 0
  for c in characteristics:
    for i in range(0, len(last_states_array)):
      value = last_states_array[i][ind]
      if value in final_dict[c]:
        final_dict[c][value] += 1
      else:
        final_dict[c][value] = 1
    ind += 1
   
  return final_dict
  
print(reward_tracker[1])
# trained_q_table = None
starting_state = list(all_episodes.iloc[8])
print(starting_state)

pred_prob_next_state = model.predict_proba(next_state2_copy)

t1_state = [6, 2, 0, 0, 0, 2, 84, 4, 1, 0, 457, 3, 45, 23, 6, 1, 2, 3, 0, 10, 0, 0] #list(all_episodes.iloc[2])
t1_state = convert_state_for_model(t1_state)
t1_prob = model.predict_proba(t1_state)
print('t1_prob: ', t1_prob)

t1_state_m = [6, 2, 0, 0, 0, 2, 84, 4, 1, 0, 457, 2, 45, 57, 6, 1, 7, 3, 3550, 50, 0, 2]
t1_state_m = convert_state_for_model(t1_state_m)
t1_prob_m = model.predict_proba(t1_state_m)
print('t1_prob_m: ', t1_prob_m)

t4_state = [53, 3, 13, 3, 2, 1, 35, 1, 6, 0, 521, 3, 10, 40, 5, 1, 11, 1, 1980, 40, 10, 2] #list(all_episodes.iloc[18])
t4_state = convert_state_for_model(t4_state)
t4_prob = model.predict_proba(t4_state)
print('t4_prob: ', t4_prob)

t4_state_m = [51, 3, 13, 3, 2, 1, 35, 1, 6, 0, 521, 2, 10, 11, 4, 1, 11, 1, 1980, 50, 10, 1]
t4_state_m = convert_state_for_model(t4_state_m)
t4_prob_m = model.predict_proba(t4_state_m)
print('t4_prob_m: ', t4_prob_m)

mixed_state = [26, 3, 13, 3, 2, 1, 43, 1, 7, 1, 200, 3, 20, 12, 5, 1, 6, 1, 7640, 40, 10, 1] #list(all_episodes.iloc[8])
mixed_state = convert_state_for_model(mixed_state)
mixed_prob = model.predict_proba(mixed_state)
print('mixed_prob: ', mixed_prob)

mixed_state_m = [27, 3, 13, 3, 2, 1, 43, 1, 7, 1, 200, 2, 20, 2, 4, 1, 11, 1, 960, 70, 10, 1]
mixed_state_m = convert_state_for_model(mixed_state_m)
mixed_prob_m = model.predict_proba(mixed_state_m)
print('mixed_prob_m: ', mixed_prob_m)


print(trained_q_table[str([11, 2, 13, 0, 0, 2, 29, 3, 1, 0, 150, 1, 29, 57, 3, 2, 6, 1, 9830, 80, 0, 1])])
a = get_max_action(trained_q_table, str([13, 3, 0, 1, 0, 2, 38, 3, 1, 0, 150, 2, 38, 1, 6, 2, 8, 1, 9760, 60, 0, 3]))
print(a)



# Open a file for writing
with open("trained_q_table_RL4_PP_mixed_v2(7).pickle", "wb") as f:
    # Pickle the variable
    pickle.dump(trained_q_table, f)

with open("reward_tracker_RL4_PP_mixed_v2(7).pickle", "wb") as f:
    # Pickle the variable
    pickle.dump(reward_tracker, f)
    
with open("reward_every_other_5_MA_v3.pickle", "wb") as f:
    # Pickle the variable
    pickle.dump(reward_every_other_5, f)

#------------------------------------------------

def get_y_best_fit(slope, intercept, x_vals):
  y_vals = []
  for v in x_vals:
    y_vals.append(slope*v + intercept)
  return y_vals
    
plt.figure()
#plt.scatter(reward_every_5_v2[0], reward_every_5_v2[1], color='blue', label='Every 5')
#plt.scatter(reward_every_other_5_v2[0], reward_every_other_5_v2[1], color='red', label='Every Other 5')
plt.scatter(reward_tracker[0], reward_tracker[1])
#a, b = np.polyfit(reward_every_5_v2[0], reward_every_5_v2[1], 1)
a, b = np.polyfit(reward_tracker[0], reward_tracker[1], 1)
#c, d = np.polyfit(reward_every_other_5_v2[0], reward_every_other_5_v2[1], 1)
y_best_fit = get_y_best_fit(a, b, reward_tracker[0])
#y_best_fit_every_5 = get_y_best_fit(a, b, reward_every_5_v2[0])
#y_best_fit_every_other_5 = get_y_best_fit(c, d, reward_every_other_5_v2[0])
plt.plot(reward_tracker[0], y_best_fit, color='blue') # best fit line
#plt.plot(reward_every_5_v2[0], y_best_fit_every_5, color='blue') # best fit line
#plt.plot(reward_every_other_5_v2[0], y_best_fit_every_other_5, color='red') # best fit line
plt.xlabel('Episode Number')
plt.ylabel('Reward')
plt.legend(loc='lower right')
plt.show()
print('Slope: ', a)
#print('Every other 5 slope: ', c)


plt.close()


def calculate_means(arr):
  means = []
  for i in range(0, len(arr), 10):
    chunk = arr[i:i+10]
    if chunk:
      mean = sum(chunk) / len(chunk)
      means.append(mean)
  return means

y = calculate_means(reward_tracker[1])
x = list(range(0, len(y)))


plt.figure()
plt.scatter(x,y)
a, b = np.polyfit(x, y, 1)
y_best_fit = get_y_best_fit(a, b, x)
plt.plot(x, y_best_fit, color='blue')
plt.xlabel('Training Episodes')
plt.title('Mixed Learning')
plt.show()
print('Slope: ', a)


def extrapolate_tiers(rew_track): # Input single array
  tier_1_rew = []
  tier_2_rew = []
  tier_3_rew = []
  tier_4_rew = []
  x_vals = list(range(int(len(rew_track) / 4)))
  print('yerr')
  for r in range(0, len(rew_track)):
    if (r % 4) == 0:
      tier_1_rew.append(rew_track[r])
    if (r % 4) == 1:
      tier_2_rew.append(rew_track[r])
    if (r % 4) == 2:
      tier_3_rew.append(rew_track[r])
    if (r % 4) == 3:
      tier_4_rew.append(rew_track[r])
  
  return x_vals, tier_1_rew, tier_2_rew, tier_3_rew, tier_4_rew


with open("reward_every_5(1)_v3.pickle", "rb") as f:
    reward_every_5 = pickle.load(f)
    

x_axis, y_tier1, y_tier2, y_tier3, y_tier4 = extrapolate_tiers(reward_every_5[1])
x_axis.append(len(x_axis))
print(len(x_axis))
print(len(y_tier2))
plt.figure()
plt.scatter(x_axis, y_tier4)
plt.show()

# -----------------------------Testing the agent------------------------------

def q_table_check_greedy(q_table2, state, whole_data_set): # generates/checks if the agent has seen this state before and what actions it can take in it
  profile = str(state)
  if profile not in q_table2.keys():
    q_table2[profile] = {}
    available_actions = get_all_available_actions(state, whole_data_set)
    for action in available_actions:
      q_table2[profile][action] = 0
  # q_table = {'[state]':{'['char', new value]': value}}
  return q_table2


epsilon = 0
def greedy_agent_run(state1, epsilon, trained_q_table1, RL_data_set1, model1): #, start_end_tracker_model1, start_end_tracker_cp1):
  first_income_tier = get_income_tier(model1, state1)
  print("First income tier: ", first_income_tier)
  print("First State: ", state1)
  state1_copy = state1.copy()
  #rew_tracker = []
  for z in range(0, 20):
    # print('greedy_agent_run, state is: ' + str(state1))
    trained_q_table1 = q_table_check_greedy(trained_q_table1, state1, RL_data_set1)
    action = select_action(state1, epsilon, trained_q_table1) # returns characteristic to increment and it's new value: ['char', new value]
    print('greedy_agent_run, action is: ' + action)
    next_state = take_action(state1, action) # returns new state with the new char value from above, in array format
    # print('greey_agent_run, next_state is: ' + str(next_state))
    state1 = next_state
    # print(' ')
  last_income_tier = get_income_tier(model1, state1)
  print('Last income tier: ', last_income_tier)
  print("Last State: ", state1)
  #start_end_tracker_model1.append([first_income_tier, last_income_tier])
  #start_end_tracker_cp1.append([state1_copy, next_state])
  #return start_end_tracker_model1, start_end_tracker_cp1
  return None

greedy_agent_run(starting_state, epsilon, trained_q_table, RL_data_set, model)
print(trained_q_table[str(starting_state)])

def prediction_prob(model1, state2, trained_q_table2, whole_data_set2):
  # Do it one at a time. Given a start state, go through 20 increments and track the actions taken
  # Can test the same starting state with different q-tables to compare what they've learned
  x_vals = []
  y_vals = []
  epsilon = 0
  print('Start: ', state2)
  for i in range(0, 20):
    x_vals.append(i + 1)
    state_copy = state2.copy()
    state_copy = convert_state_for_model(state_copy)
    pred_prob = model1.predict_proba(state_copy)
    m = max(pred_prob[0])
    m_ind = list(pred_prob[0]).index(m)
    if m_ind == 1:
      pred_prob = pred_prob[0][m_ind] + 1
    elif m_ind == 2:
      pred_prob = pred_prob[0][m_ind] + 2
    elif m_ind == 3:
      pred_prob = pred_prob[0][m_ind] + 3
    else:
      pred_prob = pred_prob[0][m_ind]
      
    y_vals.append(pred_prob)
    trained_q_table2 = q_table_check_greedy(trained_q_table2, state2, whole_data_set2)
    action = select_action(state2, epsilon, trained_q_table2)
    next_state = take_action(state2, action)
    state2 = next_state
    print(state2)
  return x_vals, y_vals


test_starting_state_t4 = list(all_episodes.iloc[19])
test_starting_state_t1 = list(all_episodes.iloc[0])

q_table1 = {}
x_probs, y_probs = prediction_prob(model, starting_state, trained_q_table, RL_data_set)

plt.figure()
plt.plot(x_probs, y_probs)
plt.xlabel('Increment Number')
plt.ylabel('Probability Confidence')
plt.show()

print(get_max_action(trained_q_table, test_starting_state_t4))
print(trained_q_table[str(test_starting_state_t4)]["['OCC', 5165]"])
print(test_starting_state_t4[18])
